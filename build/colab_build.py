#!/usr/bin/env python3
"""Build + push the MC3 image WITHOUT Docker, on a Google Colab CPU VM (Ubuntu, Python 3.12, no sudo).

    colab exec -s mc3-build -f build/colab_build.py          # real run, runs inside a Jupyter kernel
    python3 build/colab_build.py --dry-run                   # print the plan, no network, no writes
    python3 build/colab_build.py --app-tar SRC_DIR OUT.tar   # only build the deterministic /app layer

The image is assembled with `crane append` on top of the ROCm base image: the base layers are never
downloaded, only (re)used by digest (cross-repo mount on Docker Hub). Layers appended, in order:
    00-deps.tar        /opt/venv/lib/python3.14/site-packages/**      (cp314 wheels, pip --target)
    10-model-meta.tar  /models/Qwen3-VL-8B-Instruct/** minus *.safetensors
    11..14-shardN.tar  one safetensors shard each
    20-app.tar         /app/** (+ empty corpus/ output/ index/)       (separate: REF:heavy-noapp + app)
Tags: REF:tmp (chain), REF:heavy-noapp (deps+model+CMD/ENV), REF:heavy and REF:$MC3_TAG (+ app layer).

State lives under /content/mc3 (MC3_ROOT overrides it, for tests only). Phases are idempotent: a phase whose
marker /content/mc3/state/<phase>.done exists is skipped (MC3_FORCE=1 ignores markers).

Environment (all optional except MC3_REF for push/verify/app):
    MC3_PHASE   comma list of: tools,wheels,weights,tars,push,verify,app   (default: all but app)
    MC3_REF     e.g. docker.io/<user>/mc3            MC3_TAG   default v0
    MC3_BASE    base image                           MC3_MODEL_REPO / MC3_MODEL_REV   model + pinned revision
    Values may also come from /content/mc3/env.json ({"MC3_REF": "..."}); real env vars win.
Inputs: /content/mc3/src.tar (app/ + build/), /content/mc3/secrets/{dockerhub,ghcr}.{user,token}.
Secrets are only piped to `crane auth login --password-stdin`; they are never printed or put in argv.
"""
import glob
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------- constants

CRANE_VERSION = "v0.22.1"
CRANE_URL = ("https://github.com/google/go-containerregistry/releases/download/"
             f"{CRANE_VERSION}/go-containerregistry_Linux_x86_64.tar.gz")
DEFAULTS = {
    "MC3_TAG": "v0",
    "MC3_BASE": "rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0",
    "MC3_MODEL_REPO": "Qwen/Qwen3-VL-8B-Instruct",
    "MC3_MODEL_REV": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
}
MODEL_NAME = "Qwen3-VL-8B-Instruct"
SITE_PACKAGES = "opt/venv/lib/python3.14/site-packages"
BASE_GIB = 29.2          # uncompressed size of the base image (spec section 1)
LIMIT_GIB = 58.0         # abort above this (hard limit is 60 GiB)
GiB = 2 ** 30
PIP_PLATFORM_ARGS = [
    "--only-binary=:all:", "--python-version", "3.14",
    "--platform", "manylinux_2_28_x86_64", "--platform", "manylinux2014_x86_64",
    "--platform", "manylinux_2_17_x86_64",
]
PHASES = ["tools", "wheels", "weights", "tars", "push", "verify", "app"]
DEFAULT_PHASES = ["tools", "wheels", "weights", "tars", "push", "verify"]
CONTAINER_ENV = [
    "MC3_MODEL=/models/Qwen3-VL-8B-Instruct", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
    "PYTHONUNBUFFERED=1", "TOKENIZERS_PARALLELISM=false", "OMP_NUM_THREADS=8",
]
CONTAINER_CMD = ["python3", "/app/server.py"]
CONTAINER_WORKDIR = "/app"
MODEL_META_REQUIRED = [
    "config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors.index.json",
]
ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])

T0 = time.time()


def log(msg=""):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- config

class Cfg:
    def __init__(self):
        self.root = os.environ.get("MC3_ROOT", "/content/mc3")
        extra = {}
        p = os.path.join(self.root, "env.json")
        if os.path.isfile(p):
            try:
                with open(p) as fh:
                    extra = {str(k): str(v) for k, v in json.load(fh).items()}
            except Exception as e:  # a broken env.json must not be silent
                log(f"WARN cannot read {p}: {e}")

        def get(k):
            return os.environ.get(k) or extra.get(k) or DEFAULTS.get(k, "")

        self.ref = get("MC3_REF").rstrip("/")
        self.tag = get("MC3_TAG")
        self.base = get("MC3_BASE")
        self.repo = get("MC3_MODEL_REPO")
        self.rev = get("MC3_MODEL_REV")
        self.force = get("MC3_FORCE") == "1"
        phases = get("MC3_PHASE")
        self.phases = [x.strip() for x in phases.split(",") if x.strip()] if phases else list(DEFAULT_PHASES)

    # paths
    def p(self, *a):
        return os.path.join(self.root, *a)

    @property
    def crane(self):
        return self.p("bin", "crane")

    @property
    def model_dir(self):
        return self.p("stage-model", "models", MODEL_NAME)

    @property
    def sp_dir(self):
        return self.p("stage-deps", SITE_PACKAGES)

    def marker(self, name):
        return self.p("state", f"{name}.done")

    @property
    def registry(self):
        return parse_ref(self.ref)[0] if self.ref else ""


def parse_ref(ref):
    """'docker.io/u/mc3' -> ('docker.io', 'u/mc3'); 'ghcr.io/u/mc3' -> ('ghcr.io', 'u/mc3'); 'u/mc3' -> docker.io."""
    first, _, rest = ref.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        reg, repo = first, rest
    else:
        reg, repo = "docker.io", ref
    if reg in ("index.docker.io", "registry-1.docker.io"):
        reg = "docker.io"
    if reg == "docker.io" and "/" not in repo:
        repo = "library/" + repo
    return reg, repo


# ---------------------------------------------------------------- subprocess helpers

def run(cmd, cfg=None, input_bytes=None, capture=False, check=True, cwd=None):
    """Run a command. Streams merged stdout/stderr line by line (Jupyter does not show child fds) unless
    capture=True, where stdout is returned clean (stderr only shown on failure)."""
    env = dict(os.environ)
    if cfg:
        env["PATH"] = cfg.p("bin") + os.pathsep + env.get("PATH", "")
    shown = " ".join(cmd)
    log("$ " + (shown if len(shown) <= 400 else shown[:400] + f" ... [{len(cmd)} args]"))
    if capture:
        r = subprocess.run(cmd, input=input_bytes, capture_output=True, env=env, cwd=cwd)
        out = r.stdout.decode("utf-8", "replace")
        if check and r.returncode != 0:
            raise RuntimeError(f"command failed (exit {r.returncode}): {' '.join(cmd[:3])} ...\n"
                               + r.stderr.decode("utf-8", "replace")[-2000:])
        return r.returncode, out
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE if input_bytes is not None else None,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    if input_bytes is not None:
        pr.stdin.write(input_bytes)
        pr.stdin.close()
    lines = []
    for raw in iter(pr.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        lines.append(line)
        print("    " + line, flush=True)
    rc = pr.wait()
    if check and rc != 0:
        raise RuntimeError(f"command failed (exit {rc}): {' '.join(cmd[:3])} ...\n" + "\n".join(lines[-30:]))
    return rc, "\n".join(lines)


def crane_json(cfg, *args):
    _, out = run([cfg.crane, *args], cfg, capture=True)
    return json.loads(out)


def strip_tag(ref):
    head, _, last = ref.rpartition("/")
    return (head + "/" if head else "") + last.split(":")[0].split("@")[0]


def resolve_manifest(cfg, ref):
    """Return the single-image manifest for ref (follows an index to its linux/amd64 entry)."""
    m = crane_json(cfg, "manifest", ref)
    if "manifests" in m:
        pick = None
        for e in m["manifests"]:
            pl = e.get("platform", {})
            if pl.get("os") == "linux" and pl.get("architecture") == "amd64":
                pick = e["digest"]
                break
        if not pick:
            raise RuntimeError(f"{ref}: index without linux/amd64 manifest")
        m = crane_json(cfg, "manifest", f"{strip_tag(ref)}@{pick}")
    return m


# ---------------------------------------------------------------- phase bookkeeping

def phase_done(cfg, name):
    return os.path.exists(cfg.marker(name)) and not cfg.force


def mark(cfg, name, **info):
    os.makedirs(cfg.p("state"), exist_ok=True)
    with open(cfg.marker(name), "w") as fh:
        json.dump(dict(info, at=time.strftime("%Y-%m-%d %H:%M:%S")), fh)


def human(n):
    return f"{n / GiB:.2f} GiB" if n >= GiB else f"{n / 2**20:.1f} MiB"


# ---------------------------------------------------------------- deterministic tar

def _fmode(st_mode):
    return 0o755 if st_mode & 0o111 else 0o644


def collect(src_dir, arc_prefix, exclude=None, keep_root_dir=False):
    """[(arcname, abs_path, kind)] sorted by arcname. kind in dir|file|symlink. No leading '/' or './'."""
    src_dir = os.path.abspath(src_dir)
    prefix = arc_prefix.strip("/")
    out = []
    if keep_root_dir and prefix:
        out.append((prefix, src_dir, "dir"))
    for dp, dns, fns in os.walk(src_dir, followlinks=False):
        dns.sort()
        rel_dir = os.path.relpath(dp, src_dir)
        rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
        keep = []
        for d in dns:
            rel = f"{rel_dir}/{d}".lstrip("/")
            full = os.path.join(dp, d)
            if exclude and exclude(rel, True):
                continue
            arc = f"{prefix}/{rel}".lstrip("/")
            if os.path.islink(full):
                out.append((arc, full, "symlink"))
                continue
            out.append((arc, full, "dir"))
            keep.append(d)
        dns[:] = keep
        for f in sorted(fns):
            rel = f"{rel_dir}/{f}".lstrip("/")
            full = os.path.join(dp, f)
            if exclude and exclude(rel, False):
                continue
            arc = f"{prefix}/{rel}".lstrip("/")
            if os.path.islink(full):
                out.append((arc, full, "symlink"))
            elif os.path.isfile(full):
                out.append((arc, full, "file"))
            else:
                log(f"WARN skipping special file {full}")
    out.sort(key=lambda e: e[0])
    return out


def write_tar(out_path, entries, extra_dirs=()):
    """Deterministic layer tar: uid/gid 0, no names, mtime 0, sorted, PAX without extended attrs."""
    allents = list(entries) + [(d.strip("/"), None, "dir") for d in extra_dirs]
    allents.sort(key=lambda e: e[0])
    tmp = out_path + ".part"
    n_files = 0
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tf:
        for arc, path, kind in allents:
            ti = tarfile.TarInfo(arc)
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            ti.mtime = 0
            if kind == "dir":
                ti.type = tarfile.DIRTYPE
                ti.mode = 0o755
                tf.addfile(ti)
            elif kind == "symlink":
                ti.type = tarfile.SYMTYPE
                ti.linkname = os.readlink(path)
                ti.mode = 0o777
                tf.addfile(ti)
            else:
                st = os.stat(path)
                ti.size = st.st_size
                ti.mode = _fmode(st.st_mode)
                with open(path, "rb") as fh:
                    tf.addfile(ti, fh)
                n_files += 1
    os.replace(tmp, out_path)
    return n_files


APP_EXCLUDE_DIRS = {"__pycache__", "tests", ".pytest_cache", ".git"}
APP_EMPTY_DIRS = ["app/corpus", "app/output", "app/index"]


def _app_exclude(rel, is_dir):
    parts = rel.split("/")
    if any(p in APP_EXCLUDE_DIRS for p in parts):
        return True
    if parts[0] in ("corpus", "output", "index"):  # never ship local data; recreated empty
        return True
    return not is_dir and rel.endswith((".pyc", ".pyo"))


def build_app_tar(src_dir, out_path):
    """src_dir contains app/ (and optionally build/checks.py, shipped as /app/checks.py). Returns size."""
    app = os.path.join(src_dir, "app")
    if not os.path.isdir(app):
        raise RuntimeError(f"{app} not found")
    ents = collect(app, "app", exclude=_app_exclude, keep_root_dir=True)
    have = {e[0] for e in ents}
    chk = os.path.join(src_dir, "build", "checks.py")
    if os.path.isfile(chk) and "app/checks.py" not in have:
        ents.append(("app/checks.py", chk, "file"))
    extra = [d for d in APP_EMPTY_DIRS if d not in have]
    nf = write_tar(out_path, ents, extra_dirs=extra)
    size = os.path.getsize(out_path)
    log(f"{os.path.basename(out_path)}: {nf} files, {human(size)}")
    return size


def tar_listing(path, limit=12):
    with tarfile.open(path) as tf:
        names = tf.getnames()
    return names[:limit], len(names)


# ---------------------------------------------------------------- phase 1: tools

def phase_tools(cfg):
    for d in ("bin", "state", "secrets", "wheels", "tars"):
        os.makedirs(cfg.p(d), exist_ok=True)
    du = shutil.disk_usage(cfg.root)
    log(f"disk free: {human(du.free)} of {human(du.total)}")
    if not os.access(cfg.crane, os.X_OK):
        tgz = cfg.p("bin", "crane.tgz")
        log(f"downloading crane {CRANE_VERSION}")
        urllib.request.urlretrieve(CRANE_URL, tgz)
        with tarfile.open(tgz) as tf:
            m = tf.getmember("crane")
            m.name = "crane"
            tf.extract(m, cfg.p("bin"), filter="data")
        os.chmod(cfg.crane, 0o755)
        os.remove(tgz)
    run([cfg.crane, "version"], cfg)
    run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf-xet"], cfg)
    extract_src(cfg)
    mark(cfg, "tools")


def extract_src(cfg):
    tarp = cfg.p("src.tar")
    if not os.path.isfile(tarp):
        raise RuntimeError(f"{tarp} missing: `colab upload` the code tarball (app/ + build/) first")
    dst = cfg.p("src")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    with tarfile.open(tarp) as tf:
        tf.extractall(dst, filter="data")
    if not os.path.isdir(os.path.join(dst, "app")):
        raise RuntimeError(f"{tarp} has no app/ directory")
    log(f"src extracted to {dst}: {sorted(os.listdir(dst))}")


# ---------------------------------------------------------------- phase 2: wheels

def check_base_config(cfg):
    conf = crane_json(cfg, "config", cfg.base)
    c = conf.get("config", {})
    print("    base Env       :", json.dumps(c.get("Env")), flush=True)
    print("    base User      :", json.dumps(c.get("User")), flush=True)
    print("    base Cmd       :", json.dumps(c.get("Cmd")), flush=True)
    print("    base Entrypoint:", json.dumps(c.get("Entrypoint")), flush=True)
    print("    base WorkingDir:", json.dumps(c.get("WorkingDir")), flush=True)
    path = next((e for e in c.get("Env", []) if e.startswith("PATH=")), "")
    if "/opt/venv/bin" not in path:
        raise RuntimeError("ABORT: base image PATH has no /opt/venv/bin -> the venv is not /opt/venv, "
                           f"site-packages target would be wrong. PATH is: {path!r}")
    if c.get("Entrypoint"):
        raise RuntimeError(f"ABORT: base has Entrypoint {c['Entrypoint']}: our CMD would become its argument")
    return conf


def unzip_wheels(wheels, sp):
    """Fallback when `pip install --target` refuses foreign-python wheels: wheels are zips already in
    site-packages layout (purelib/platlib); *.data/scripts are skipped (console scripts are not needed)."""
    import zipfile
    for w in sorted(wheels):
        with zipfile.ZipFile(w) as z:
            names = z.namelist()
            dist = next(n.split("/")[0] for n in names if n.split("/")[0].endswith(".dist-info"))
            for n in names:
                top = n.split("/")[0]
                if top.endswith(".data"):
                    sub = n.split("/")[1] if "/" in n else ""
                    if sub in ("purelib", "platlib"):
                        rel = n.split("/", 2)[2] if n.count("/") >= 2 else ""
                        if rel:
                            z.extract(n, sp + ".tmpdata")
                            dst = os.path.join(sp, rel)
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            shutil.move(os.path.join(sp + ".tmpdata", n), dst)
                    continue
                z.extract(n, sp)
            with open(os.path.join(sp, dist, "INSTALLER"), "w") as fh:
                fh.write("pip\n")
    shutil.rmtree(sp + ".tmpdata", ignore_errors=True)


def phase_wheels(cfg):
    log("base image config")
    check_base_config(cfg)
    req = cfg.p("src", "app", "requirements.txt")
    wd = cfg.p("wheels")
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd)
    pip = [sys.executable, "-m", "pip"]
    run(pip + ["download", *PIP_PLATFORM_ARGS, "--no-deps", "-d", wd, "-r", req], cfg)
    # sdist-only pure-Python packages (antlr4-python3-runtime 4.9.3) -> py3-none-any wheel
    for sd in sorted(glob.glob(os.path.join(wd, "*.tar.gz")) + glob.glob(os.path.join(wd, "*.zip"))):
        run(pip + ["wheel", "-q", "--no-deps", "-w", wd, sd], cfg)
        os.remove(sd)
    wheels = sorted(glob.glob(os.path.join(wd, "*.whl")))
    log(f"{len(wheels)} wheels")
    for w in wheels:
        print("    " + os.path.basename(w), flush=True)

    stage = cfg.p("stage-deps")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(cfg.sp_dir)
    rc, _ = run(pip + ["install", "--no-deps", "--no-compile", *PIP_PLATFORM_ARGS,
                       "--target", cfg.sp_dir, *wheels], cfg, check=False)
    if rc != 0:
        log("pip install --target refused the foreign wheels -> unzip fallback")
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(cfg.sp_dir)
        unzip_wheels(wheels, cfg.sp_dir)
    # console scripts (bin/) point at the VM python: useless and misleading in the image
    shutil.rmtree(os.path.join(cfg.sp_dir, "bin"), ignore_errors=True)
    n_pyc = 0
    for dp, dns, _ in os.walk(cfg.sp_dir):
        for d in list(dns):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(dp, d))
                dns.remove(d)
                n_pyc += 1
    pths = glob.glob(os.path.join(cfg.sp_dir, "*.pth"))
    if pths:
        log(f"WARN .pth files present (check they are needed): {[os.path.basename(x) for x in pths]}")
    dists = sorted(d[:-10] for d in os.listdir(cfg.sp_dir) if d.endswith(".dist-info"))
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(cfg.sp_dir) for f in fs
                if not os.path.islink(os.path.join(dp, f)))
    log(f"installed {len(dists)} dists, {human(total)}, removed {n_pyc} __pycache__")
    # every requirements line must have landed
    want = []
    for ln in open(req):
        ln = ln.split("#")[0].strip()
        if ln and not ln.startswith("-"):
            want.append(re.split(r"[=<>!~ ]", ln)[0].lower().replace("-", "_").replace(".", "_"))
    have = {d.split("-")[0].lower().replace("-", "_").replace(".", "_") for d in dists}
    missing = [w for w in want if w not in have]
    if missing:
        raise RuntimeError(f"requirements not installed: {missing}")
    mark(cfg, "wheels", dists=len(dists), bytes=total)


# ---------------------------------------------------------------- phase 3: weights

def verify_model_dir(model_dir):
    """Same logic as build/checks.py weights (reimplemented: checks.py is not importable here)."""
    problems = []
    for f in MODEL_META_REQUIRED:
        p = os.path.join(model_dir, f)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            problems.append(f"missing/empty {f}")
    shards = []
    try:
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as fh:
            wm = json.load(fh)["weight_map"]
        shards = sorted(set(wm.values()))
        for s in shards:
            p = os.path.join(model_dir, s)
            if not os.path.isfile(p) or os.path.getsize(p) <= 0:
                problems.append(f"shard missing/empty {s}")
    except Exception as e:
        problems.append(f"index unreadable: {e}")
    if problems:
        raise RuntimeError("weights check failed: " + "; ".join(problems))
    total = sum(os.path.getsize(os.path.join(model_dir, s)) for s in shards)
    log(f"weights OK: {len(shards)} shards, {human(total)}")
    return shards


def phase_weights(cfg):
    from huggingface_hub import snapshot_download
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.makedirs(cfg.model_dir, exist_ok=True)
    t = time.time()
    snapshot_download(repo_id=cfg.repo, revision=cfg.rev, local_dir=cfg.model_dir,
                      allow_patterns=["*.json", "*.txt", "*.model"])
    log(f"metadata downloaded in {time.time() - t:.1f}s: {sorted(os.listdir(cfg.model_dir))}")
    t = time.time()
    snapshot_download(repo_id=cfg.repo, revision=cfg.rev, local_dir=cfg.model_dir,
                      allow_patterns=["*.safetensors"], max_workers=8)
    dt = time.time() - t
    sz = sum(os.path.getsize(os.path.join(cfg.model_dir, f)) for f in os.listdir(cfg.model_dir)
             if f.endswith(".safetensors"))
    log(f"shards downloaded in {dt:.1f}s ({sz / 2**20 / max(dt, 1e-6):.0f} MB/s)")
    shutil.rmtree(os.path.join(cfg.model_dir, ".cache"), ignore_errors=True)
    verify_model_dir(cfg.model_dir)
    mark(cfg, "weights", bytes=sz)


# ---------------------------------------------------------------- phase 4: tars

def tar_plan(cfg):
    """[(tar name, kind)] for the heavy image (app tar excluded)."""
    return [("00-deps.tar", "deps"), ("10-model-meta.tar", "meta")] + \
           [(f"{11 + i}-shard{i + 1}.tar", "shard") for i in range(4)]


def phase_tars(cfg):
    td = cfg.p("tars")
    shutil.rmtree(td, ignore_errors=True)
    os.makedirs(td)
    sizes = {}
    # 00-deps: site-packages content (not the site-packages dir itself: it exists in the base)
    ents = collect(cfg.sp_dir, SITE_PACKAGES)
    nf = write_tar(os.path.join(td, "00-deps.tar"), ents)
    sizes["00-deps.tar"] = os.path.getsize(os.path.join(td, "00-deps.tar"))
    log(f"00-deps.tar: {nf} files, {human(sizes['00-deps.tar'])}")
    # model: dirs for models/ and models/<name>/ live in the meta tar only
    mp = f"models/{MODEL_NAME}"
    meta = collect(cfg.model_dir, mp, exclude=lambda rel, d: rel.split("/")[0] == ".cache" or ((not d) and rel.endswith(".safetensors")),
                   keep_root_dir=True)
    meta.append(("models", None, "dir"))
    nf = write_tar(os.path.join(td, "10-model-meta.tar"), meta)
    sizes["10-model-meta.tar"] = os.path.getsize(os.path.join(td, "10-model-meta.tar"))
    log(f"10-model-meta.tar: {nf} files, {human(sizes['10-model-meta.tar'])}")
    shards = verify_model_dir(cfg.model_dir)
    if len(shards) != 4:
        log(f"WARN expected 4 shards, found {len(shards)}")
    for i, s in enumerate(shards):
        name = f"{11 + i}-shard{i + 1}.tar"
        write_tar(os.path.join(td, name), [(f"{mp}/{s}", os.path.join(cfg.model_dir, s), "file")])
        sizes[name] = os.path.getsize(os.path.join(td, name))
        log(f"{name}: {s}, {human(sizes[name])}")
    sizes["20-app.tar"] = build_app_tar(cfg.p("src"), os.path.join(td, "20-app.tar"))
    total = sum(sizes.values())
    est = BASE_GIB + total / GiB
    log(f"sum of new layers (uncompressed): {human(total)}; + base {BASE_GIB} GiB = {est:.2f} GiB (limit {LIMIT_GIB})")
    for n in sorted(sizes):
        names, cnt = tar_listing(os.path.join(td, n), 3)
        print(f"    {n:20s} {human(sizes[n]):>10s}  {cnt} entries, first: {names}", flush=True)
    if est > LIMIT_GIB:
        raise RuntimeError(f"ABORT: estimated uncompressed image {est:.2f} GiB > {LIMIT_GIB} GiB")
    with open(cfg.p("state", "tars.json"), "w") as fh:
        json.dump(sizes, fh)
    mark(cfg, "tars", total=total, est_gib=round(est, 2))


# ---------------------------------------------------------------- phase 5: push (and app)

def registry_login(cfg):
    reg = cfg.registry
    name = "ghcr" if reg == "ghcr.io" else "dockerhub"
    uf, tf_ = cfg.p("secrets", f"{name}.user"), cfg.p("secrets", f"{name}.token")
    for f in (uf, tf_):
        if not os.path.isfile(f):
            raise RuntimeError(f"missing credential file {f} (see build/README.md)")
    user = open(uf).read().strip()
    token = open(tf_, "rb").read().strip()
    host = "index.docker.io" if reg == "docker.io" else reg
    run([cfg.crane, "auth", "login", host, "-u", user, "--password-stdin"], cfg, input_bytes=token)
    del token


def mutate_args(cfg, ref, tag_ref):
    a = [cfg.crane, "mutate", ref, "--cmd", ",".join(CONTAINER_CMD), "--workdir", CONTAINER_WORKDIR]
    for e in CONTAINER_ENV:
        a += ["--env", e]
    return a + ["-t", tag_ref]


def n_layers(cfg, ref):
    return len(resolve_manifest(cfg, ref)["layers"])


def timed_append(cfg, base, tar, out_ref):
    size = os.path.getsize(tar)
    t = time.time()
    run([cfg.crane, "append", "-b", base, "-f", tar, "-t", out_ref], cfg)
    dt = time.time() - t
    log(f"append {os.path.basename(tar)}: {human(size)} in {dt:.1f}s = {size / 2**20 / max(dt, 1e-6):.0f} MB/s (uncompressed)")


def finalize_with_app(cfg, tag):
    """REF:heavy-noapp + 20-app.tar -> REF:<tag> with CMD/ENV/WORKDIR re-applied."""
    app_tar = cfg.p("tars", "20-app.tar")
    timed_append(cfg, f"{cfg.ref}:heavy-noapp", app_tar, f"{cfg.ref}:{tag}")
    run(mutate_args(cfg, f"{cfg.ref}:{tag}", f"{cfg.ref}:{tag}"), cfg)


def phase_push(cfg):
    need_ref(cfg)
    registry_login(cfg)
    base_n = n_layers(cfg, cfg.base)
    log(f"base has {base_n} layers")
    chain = f"{cfg.ref}:tmp"
    for i, (name, _) in enumerate(tar_plan(cfg)):
        mk = f"push-{name}"
        if phase_done(cfg, mk):
            log(f"skip {name} (marker)")
            continue
        tar = cfg.p("tars", name)
        if i == 0:
            src = cfg.base
        else:
            have = n_layers(cfg, chain)
            if have == base_n + i + 1:     # appended before the marker was written
                log(f"{name} already in {chain}")
                mark(cfg, mk)
                continue
            if have != base_n + i:
                raise RuntimeError(f"{chain} has {have} layers, expected {base_n + i} before {name}")
            src = chain
        if not os.path.isfile(tar):
            raise RuntimeError(f"{tar} missing and no marker: rerun phase tars")
        timed_append(cfg, src, tar, chain)
        mark(cfg, mk)
        os.remove(tar)  # save disk
    run(mutate_args(cfg, chain, f"{cfg.ref}:heavy-noapp"), cfg)
    app_tar = cfg.p("tars", "20-app.tar")
    if not os.path.isfile(app_tar):
        build_app_tar(cfg.p("src"), app_tar)
    finalize_with_app(cfg, cfg.tag)
    run([cfg.crane, "tag", f"{cfg.ref}:{cfg.tag}", "heavy"], cfg)
    mark(cfg, "push", tag=cfg.tag)


def phase_app(cfg):
    """Rebuild only the app layer from a fresh src.tar and tag REF:<MC3_TAG>."""
    need_ref(cfg)
    extract_src(cfg)
    os.makedirs(cfg.p("tars"), exist_ok=True)
    build_app_tar(cfg.p("src"), cfg.p("tars", "20-app.tar"))
    registry_login(cfg)
    finalize_with_app(cfg, cfg.tag)
    log(f"pushed {cfg.ref}:{cfg.tag}")


def need_ref(cfg):
    if not cfg.ref:
        raise RuntimeError("MC3_REF is not set (e.g. docker.io/<user>/mc3)")


# ---------------------------------------------------------------- phase 6: verify

def anon_manifest(ref, tag):
    reg, repo = parse_ref(ref)
    if reg == "docker.io":
        tok_url = ("https://auth.docker.io/token?service=registry.docker.io&scope="
                   + urllib.parse.quote(f"repository:{repo}:pull", safe=":"))
        host = "registry-1.docker.io"
    else:
        tok_url = f"https://{reg}/token?scope=" + urllib.parse.quote(f"repository:{repo}:pull", safe=":")
        host = reg
    with urllib.request.urlopen(tok_url, timeout=30) as r:
        tok = json.load(r).get("token") or ""
    req = urllib.request.Request(f"https://{host}/v2/{repo}/manifests/{tag}",
                                 headers={"Authorization": f"Bearer {tok}", "Accept": ACCEPT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers.get("Content-Type"), r.headers.get("Docker-Content-Digest"), json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, None, None, None


def phase_verify(cfg):
    need_ref(cfg)
    ref = f"{cfg.ref}:{cfg.tag}"
    man = resolve_manifest(cfg, ref)
    layers = man["layers"]
    comp = sum(l["size"] for l in layers)
    log(f"{ref}: {len(layers)} layers, compressed {human(comp)}")
    for l in layers[-8:]:
        print(f"    {l['digest'][:19]}  {human(l['size']):>10s}", flush=True)
    base = resolve_manifest(cfg, cfg.base)
    bl = [l["digest"] for l in base["layers"]]
    ours = [l["digest"] for l in layers]
    if ours[:len(bl)] != bl:
        raise RuntimeError("FAIL: base layer digests are not a prefix of the image layers")
    log(f"OK base layers intact: {len(bl)} base + {len(ours) - len(bl)} new")
    conf = crane_json(cfg, "config", ref)["config"]
    print("    Cmd       :", json.dumps(conf.get("Cmd")), flush=True)
    print("    WorkingDir:", json.dumps(conf.get("WorkingDir")), flush=True)
    print("    Env       :", json.dumps(conf.get("Env")), flush=True)
    bad = []
    if conf.get("Cmd") != CONTAINER_CMD:
        bad.append(f"Cmd {conf.get('Cmd')}")
    if conf.get("WorkingDir") != CONTAINER_WORKDIR:
        bad.append(f"WorkingDir {conf.get('WorkingDir')}")
    for e in CONTAINER_ENV:
        if e not in (conf.get("Env") or []):
            bad.append(f"Env {e}")
    if bad:
        raise RuntimeError("FAIL config: " + "; ".join(bad))
    sizes_p = cfg.p("state", "tars.json")
    if os.path.isfile(sizes_p):
        total = sum(json.load(open(sizes_p)).values())
        est = BASE_GIB + total / GiB
        log(f"uncompressed estimate: {BASE_GIB} GiB base + {human(total)} = {est:.2f} GiB (limit 60)")
        if est >= 60:
            raise RuntimeError("FAIL: image >= 60 GiB")
    else:
        log("WARN no tars.json: uncompressed estimate skipped")
    code, ctype, digest, m = anon_manifest(cfg.ref, cfg.tag)
    log(f"anonymous manifest GET: HTTP {code} {ctype} {digest}")
    if code != 200:
        raise RuntimeError(f"FAIL anonymous pull: HTTP {code} (repo private? make it public)")
    mark(cfg, "verify", digest=digest)


# ---------------------------------------------------------------- driver

PHASE_FUNCS = {"tools": phase_tools, "wheels": phase_wheels, "weights": phase_weights,
               "tars": phase_tars, "push": phase_push, "verify": phase_verify, "app": phase_app}


def plan(cfg):
    ref = cfg.ref or "<MC3_REF unset>"
    print(f"root      : {cfg.root}")
    print(f"phases    : {cfg.phases}   (all phases: {PHASES}; app is never in the default set)")
    print(f"ref/tag   : {ref} / {cfg.tag}")
    print(f"base      : {cfg.base}")
    print(f"model     : {cfg.repo} @ {cfg.rev}")
    print("credentials: read from secrets/{dockerhub|ghcr}.{user,token}, token only via stdin (never printed)")
    print("\nplan:")
    print(f"  tools   download crane {CRANE_VERSION} -> bin/; pip install huggingface_hub hf-xet; untar src.tar -> src/")
    print("  wheels  crane config BASE (abort unless PATH has /opt/venv/bin and no Entrypoint)")
    print("          pip download " + " ".join(PIP_PLATFORM_ARGS) + " --no-deps -d wheels -r src/app/requirements.txt")
    print("          sdists -> pip wheel --no-deps;  pip install --no-deps --no-compile <same flags> --target "
          f"stage-deps/{SITE_PACKAGES} wheels/*.whl  (fallback: unzip)")
    print("  weights snapshot_download(metadata: *.json *.txt *.model) then (*.safetensors); rm .cache; verify index")
    print(f"  tars    -> tars/ (uid0 gid0 mtime0 sorted PAX), abort if base {BASE_GIB} GiB + layers > {LIMIT_GIB} GiB")
    for name, kind in tar_plan(cfg):
        print(f"            {name:20s} {kind}")
    print(f"            {'20-app.tar':20s} app (app/** + build/checks.py as app/checks.py, +empty {APP_EMPTY_DIRS})")
    print(f"  push    crane auth login (stdin);  crane append -b BASE -f 00-deps.tar -t {ref}:tmp, then each tar onto :tmp")
    print("          (tar deleted after each successful append); crane mutate ->", f"{ref}:heavy-noapp")
    print("          mutate args:", " ".join(mutate_args(cfg, "REF:tmp", "REF:heavy-noapp")[2:]))
    print(f"          append 20-app.tar onto {ref}:heavy-noapp -> {ref}:{cfg.tag} (+mutate), crane tag -> heavy")
    print("  verify  manifest layers + sizes, base digests are a prefix, config Cmd/Env/WorkingDir,")
    print("          uncompressed estimate, anonymous token + manifest GET == 200")
    print(f"  app     (MC3_PHASE=app) re-extract src.tar, rebuild 20-app.tar, append onto {ref}:heavy-noapp -> :{cfg.tag}")


def main():
    argv = sys.argv[1:]
    if "--app-tar" in argv:  # CLI helper used by build/append_app.sh
        i = argv.index("--app-tar")
        build_app_tar(argv[i + 1], argv[i + 2])
        return 0
    cfg = Cfg()
    if "--dry-run" in argv:
        plan(cfg)
        return 0
    bad = [p for p in cfg.phases if p not in PHASES]
    if bad:
        raise SystemExit(f"unknown phase(s) {bad}; valid: {PHASES}")
    os.makedirs(cfg.p("state"), exist_ok=True)
    log(f"phases: {cfg.phases}  ref={cfg.ref or '-'} tag={cfg.tag}")
    timings = {}
    for ph in cfg.phases:
        if ph != "app" and phase_done(cfg, ph):
            log(f"== {ph}: skip (marker {cfg.marker(ph)})")
            continue
        log(f"== {ph}: start")
        t = time.time()
        PHASE_FUNCS[ph](cfg)  # every phase except `app` writes its own marker
        timings[ph] = time.time() - t
        log(f"== {ph}: done in {timings[ph]:.1f}s")
    log("timings: " + ", ".join(f"{k}={v:.0f}s" for k, v in timings.items()))
    return 0


if __name__ == "__main__" or "get_ipython" in globals():
    try:
        rc = main()
    except Exception as exc:  # show the failure in the kernel output and fail the exec
        print(f"BUILD FAILED: {type(exc).__name__}: {exc}", flush=True)
        if __name__ == "__main__" and "get_ipython" not in globals():
            sys.exit(1)
        raise
