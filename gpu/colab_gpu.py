"""Drive an MC3 validation session on a Colab GPU VM.  Run it repeatedly, one phase list per call:

    colab exec -s mc3-gpu --timeout 3600 --env MC3_PHASE=setup,weights -f gpu/colab_gpu.py
    colab exec -s mc3-gpu --timeout 3600 --env MC3_PHASE=bench         -f gpu/colab_gpu.py
    colab exec -s mc3-gpu --timeout 3600 --env MC3_PHASE=suite,report  -f gpu/colab_gpu.py

Phases (MC3_PHASE, comma list, run in the order given; `all` = setup,weights,bench,suite,report):
  setup    untar /content/mc3/src.tar -> /content/mc3/src, pip install the app requirements (pins kept,
           torch frozen by a constraints file, NO --no-deps because this is the CUDA env), nvidia-smi
  weights  snapshot_download MC3_MODEL_REPO @ MC3_MODEL_REV -> MC3_MODEL_DIR (default: Qwen3-VL-8B-Instruct @ pinned
           rev -> /content/mc3/models/Qwen3-VL-8B-Instruct; skipped when done)
  bench    gpu/bench.py -> /content/mc3/runs/<ts>/<model_slug>/bench.json
  suite    eval/run_suite.py on sample-corpus, then on every eval/suites/*/ that has questions.json
           -> /content/mc3/runs/<ts>/<model_slug>/<suite>/
  ab       eval/ab_suite.py (NOT part of `all`): several hypothesis configs vs a baseline in one process, the model
           loaded once and reader calls memoised in /content/mc3/cache -> <current run>/ab/ (ab_summary.json + table).
           MC3_SUITES=comma list; MC3_AB_CONFIGS=JSON list of {"name","env"} or a path relative to src (the first
           config is the baseline); MC3_AB_GROUPS='TUNE=sample,s1;HOLDOUT=h101' (optional); MC3_AB_SOFT / MC3_AB_LIMIT
  report   merge every JSON of the current run -> one table per model + /content/mc3/runs/latest.json
Other env: MC3_UPDATE=1 re-untar src.tar first; MC3_NEW_RUN=1 start a new runs/<ts> (both one-shot); MC3_RUN=<ts> pin the run;
MC3_SUITES=sample,s1,s2,s3,s4 restrict suites ("sample" = sample-corpus); MC3_BENCH_SUITE=1 let bench run the suite too;
MC3_BASE (default /content/mc3); MC3_SKIP_PIP=1.
A/B example:  colab exec -s mc3-gpu --timeout 7200 --env MC3_PHASE=ab --env MC3_SUITES=sample,s1,s2 \\
                  --env MC3_AB_CONFIGS=eval/ab_configs.json --env 'MC3_AB_GROUPS=TUNE=sample,s1;HOLDOUT=s2' -f gpu/colab_gpu.py

A/B of another VLM (the default stays Qwen3-VL-8B): every exec that touches the model (weights, bench, suite)
must name it, because the three variables below are one-shot -- they are removed from the kernel environment when the
exec ends, so a later exec without them falls back to the default model instead of a stale one:
    --env MC3_MODEL_REPO=google/gemma-4-12b-it --env MC3_MODEL_REV=<sha> [--env MC3_MODEL_DIR=/content/mc3/models/x]
MC3_MODEL_DIR defaults to /content/mc3/models/<repo name>; <model_slug> is that directory's name.

The script runs inside the Jupyter kernel, so every heavy step is a child process whose output is piped
back and print()-ed (a child writing to fd 1 would never reach `colab exec`). torch is never imported here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

DEFAULT_REPO = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_REV = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_ENV = ("MC3_MODEL_REPO", "MC3_MODEL_REV", "MC3_MODEL_DIR")   # one-shot, see main()
ALL_PHASES = ["setup", "weights", "bench", "suite", "report"]

BASE = Path(os.environ.get("MC3_BASE", "/content/mc3"))
SRC = BASE / "src"
SRC_TAR = BASE / "src.tar"
RUNS = BASE / "runs"
STATE = BASE / "state"


def resolve_model(env=None) -> tuple[str, str, Path]:
    """(repo, revision, dir) from MC3_MODEL_REPO / MC3_MODEL_REV / MC3_MODEL_DIR (empty = unset).
    Nothing set -> the shipped default Qwen3-VL-8B-Instruct at its pinned revision."""
    env = os.environ if env is None else env
    repo = (env.get("MC3_MODEL_REPO") or "").strip() or DEFAULT_REPO
    rev = (env.get("MC3_MODEL_REV") or "").strip()
    if not rev:
        rev = DEFAULT_REV if repo == DEFAULT_REPO else "main"
    d = (env.get("MC3_MODEL_DIR") or "").strip()
    return repo, rev, (Path(d) if d else BASE / "models" / repo.split("/")[-1])


REPO_ID, REVISION, MODEL_DIR = resolve_model()
MODEL_SLUG = re.sub(r"[^A-Za-z0-9._-]+", "_", MODEL_DIR.name) or "model"
os.environ["MC3_MODEL"] = str(MODEL_DIR)   # always overwrite: the kernel env may hold the previous exec's model


# ----------------------------------------------------------------------------- plumbing

def say(msg: str = "") -> None:
    print(msg, flush=True)


def banner(title: str) -> None:
    say("\n" + "=" * 8 + f" {title} " + "=" * max(4, 66 - len(title)))


def child_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (f"{SRC}/app", str(SRC), env.get("PYTHONPATH", "")) if p)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["MC3_MODEL"] = str(MODEL_DIR)
    if REPO_ID != DEFAULT_REPO:       # A/B models: no hub kernels, so the Qwen3.5 Gated DeltaNet runs the torch path like ROCm
        env.setdefault("USE_HUB_KERNELS", "NO")
    if extra:
        env.update(extra)
    return env


def sh(argv: list[str], cwd: Path | str | None = None, env: dict | None = None, log: Path | None = None,
       prefix: str = "") -> tuple[int, list[str]]:
    """Run a child, stream its merged stdout/stderr through print(), optionally tee to `log`.
    Returns (returncode, last 200 lines). Never raises."""
    shown = " ".join(str(a) for a in argv).replace("\n", " ; ")
    say(f"$ {shown[:240]}{'...' if len(shown) > 240 else ''}" + (f"   (cwd={cwd})" if cwd else ""))
    tail: list[str] = []
    fh = None
    try:
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log, "w")
        p = subprocess.Popen([str(a) for a in argv], cwd=str(cwd) if cwd else None, env=env or child_env(),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             errors="replace")
        for line in p.stdout:
            say(prefix + line.rstrip("\n"))
            tail.append(line.rstrip("\n"))
            if len(tail) > 200:
                tail.pop(0)
            if fh:
                fh.write(line)
                fh.flush()
        rc = p.wait()
    except Exception as e:
        say(f"[colab_gpu] ERROR launching {argv[0]}: {e!r}")
        rc = 127
    finally:
        if fh:
            fh.close()
    return rc, tail


def mark(name: str) -> Path:
    STATE.mkdir(parents=True, exist_ok=True)
    return STATE / name


def current_run(create: bool) -> Path | None:
    """runs/<ts> of this session: MC3_RUN > runs/CURRENT > new (main() rewrites CURRENT on MC3_NEW_RUN=1)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    cur = RUNS / "CURRENT"
    ts = os.environ.get("MC3_RUN", "").strip()
    if not ts and cur.exists():
        ts = cur.read_text().strip()
    if not ts:
        if not create:
            return None
        ts = time.strftime("%Y%m%d-%H%M%S")
    d = RUNS / ts
    if create:
        d.mkdir(parents=True, exist_ok=True)
        cur.write_text(ts)
    return d if d.exists() else None


# ----------------------------------------------------------------------------- src.tar

def extract_src(force: bool = False) -> bool:
    if not SRC_TAR.exists():
        say(f"[src] {SRC_TAR} missing -- run: colab upload -s mc3-gpu src.tar {SRC_TAR}")
        return (SRC / "app").exists()
    st = SRC_TAR.stat()
    stamp = f"{st.st_size}-{st.st_mtime_ns}"
    stamp_f = mark("src.stamp")
    if not force and (SRC / "app").exists() and stamp_f.exists() and stamp_f.read_text() == stamp:
        say(f"[src] up to date ({SRC})")
        return True
    tmp = BASE / "src.new"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(SRC_TAR) as tf:
        try:
            tf.extractall(tmp, filter="data")
        except TypeError:  # python < 3.12
            tf.extractall(tmp)
    if not (tmp / "app").exists():  # tar made with a top-level prefix directory
        subs = [p for p in tmp.iterdir() if p.is_dir() and (p / "app").exists()]
        if len(subs) == 1:
            tmp_inner = subs[0]
            shutil.rmtree(SRC, ignore_errors=True)
            shutil.move(str(tmp_inner), str(SRC))
            shutil.rmtree(tmp, ignore_errors=True)
            stamp_f.write_text(stamp)
            say(f"[src] extracted (hoisted prefix) -> {SRC}")
            return True
        say(f"[src] ERROR: no app/ in {SRC_TAR}")
        return False
    shutil.rmtree(SRC, ignore_errors=True)
    shutil.move(str(tmp), str(SRC))
    stamp_f.write_text(stamp)
    n = sum(1 for _ in SRC.rglob("*") if _.is_file())
    say(f"[src] extracted {n} files -> {SRC}")
    return True


# ----------------------------------------------------------------------------- setup

def parse_requirements(path: Path) -> list[str]:
    """requirement lines only: no comments, no pip options (--no-binary ...), no blanks."""
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split(" #")[0].strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        out.append(line)
    # spec pins that must be present even if the file drifts
    names = {r.split("==")[0].lower() for r in out}
    for pin in ("transformers==5.17.0", "accelerate==1.15.0"):
        if pin.split("==")[0] not in names:
            out.append(pin)
    return out


def _installed(dist: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return None


def phase_setup() -> bool:
    banner("setup")
    if not extract_src(force=False):
        return False
    say(f"python {sys.version.split()[0]} ({sys.executable})")
    sh(["nvidia-smi"])
    req_file = SRC / "app" / "requirements.txt"
    if not req_file.exists():
        say(f"[setup] ERROR {req_file} missing")
        return False
    reqs = parse_requirements(req_file)
    digest = hashlib.sha1("\n".join(reqs).encode()).hexdigest()[:12]
    done = mark(f"pip.{digest}.done")
    if os.environ.get("MC3_SKIP_PIP") == "1":
        say("[setup] MC3_SKIP_PIP=1 -> skipping pip")
    elif done.exists():
        say(f"[setup] requirements {digest} already installed")
    else:
        flat = BASE / "requirements.colab.txt"
        flat.write_text("\n".join(reqs) + "\n")
        # keep the preinstalled CUDA torch stack no matter what the resolver wants
        cons = BASE / "constraints.colab.txt"
        pins = [f"{d}=={v}" for d in ("torch", "torchvision", "torchaudio", "numpy", "pillow")
                if (v := _installed(d))]
        cons.write_text("\n".join(pins) + "\n")
        say(f"[setup] {len(reqs)} requirements, constraints: {', '.join(pins)}")
        pip = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
        rc, _ = sh(pip + ["-c", str(cons), "-r", str(flat)])
        failed = []
        if rc != 0:
            say("[setup] bulk install failed -> installing one by one to isolate the culprit")
            for r in reqs:
                rc1, _ = sh(pip + ["-c", str(cons), r])
                if rc1 != 0:
                    failed.append(r)
        if failed:
            say(f"[setup] FAILED requirements: {failed}")
        else:
            done.write_text(time.strftime("%FT%T"))
    code = ("import torch, transformers, accelerate; "
            "print('torch', torch.__version__, '| cuda', torch.version.cuda, '| hip', getattr(torch.version,'hip',None), "
            "'| gpu', torch.cuda.is_available() and torch.cuda.get_device_name(0)); "
            "print('transformers', transformers.__version__, '| accelerate', accelerate.__version__); "
            "import PIL, pypdf, pypdfium2, openpyxl; print('parsers ok')")
    rc, _ = sh([sys.executable, "-c", code])
    return rc == 0


# ----------------------------------------------------------------------------- weights

def _weights_complete() -> tuple[bool, str]:
    """Sharded checkpoints (model.safetensors.index.json) are checked shard by shard; a single-file checkpoint
    (Gemma 4: model.safetensors) just needs a non-empty *.safetensors."""
    try:
        missing = []
        idx_f = MODEL_DIR / "model.safetensors.index.json"
        if idx_f.exists():
            files = sorted(set(json.loads(idx_f.read_text())["weight_map"].values()))
            missing = [f for f in files if not (MODEL_DIR / f).exists()]
            what = f"{len(files)} shards present"
        else:
            files = sorted(p.name for p in MODEL_DIR.glob("*.safetensors") if p.stat().st_size > 0)
            if not files:
                missing.append("*.safetensors")
            what = f"{len(files)} safetensors file(s) present (no index.json)"
        for f in ("config.json", "tokenizer.json"):
            if not (MODEL_DIR / f).exists():
                missing.append(f)
        if not any((MODEL_DIR / f).exists() for f in ("preprocessor_config.json", "processor_config.json")):
            missing.append("preprocessor_config.json|processor_config.json")
        return (not missing), (f"missing {missing}" if missing else what)
    except Exception as e:
        return False, repr(e)


_DL = """
import sys, time
from huggingface_hub import snapshot_download
kind, repo, rev, dest = sys.argv[1:5]
kw = dict(repo_id=repo, revision=rev, local_dir=dest, max_workers=8)
if kind == "meta":
    kw["ignore_patterns"] = ["*.safetensors"]
else:
    kw["allow_patterns"] = ["*.safetensors"]
t0 = time.time()
p = snapshot_download(**kw)
print(f"[weights] {kind} done in {time.time()-t0:.0f}s -> {p}")
"""


def phase_weights() -> bool:
    banner("weights")
    say(f"[weights] model {REPO_ID} @ {REVISION} -> {MODEL_DIR}")
    ok, why = _weights_complete()
    done = mark(f"weights.{MODEL_SLUG}.done").exists() or (REPO_ID == DEFAULT_REPO and mark("weights.done").exists())
    if done and ok:
        say(f"[weights] already done ({why}) at {MODEL_DIR}")
        return True
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    env = child_env()
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    for kind in ("meta", "shards"):
        rc, _ = sh([sys.executable, "-u", "-c", _DL, kind, REPO_ID, REVISION, str(MODEL_DIR)], env=env)
        if rc != 0:
            say(f"[weights] {kind} download failed (rc={rc})")
            return False
    shutil.rmtree(MODEL_DIR / ".cache", ignore_errors=True)
    ok, why = _weights_complete()
    say(f"[weights] verify: {why}")
    sh(["du", "-sh", str(MODEL_DIR)])
    if ok:
        mark(f"weights.{MODEL_SLUG}.done").write_text(REVISION)
    return ok


# ----------------------------------------------------------------------------- bench / suite

def _need_ready() -> str | None:
    if not (SRC / "app").exists():
        return "src not extracted -- run phase setup"
    ok, why = _weights_complete()
    if not ok:
        return f"weights incomplete ({why}) -- run phase weights"
    return None


def phase_bench() -> bool:
    banner("bench")
    if (why := _need_ready()):
        say(f"[bench] SKIP: {why}")
        return False
    run = current_run(create=True)
    mdir = run / MODEL_SLUG
    mdir.mkdir(parents=True, exist_ok=True)
    out = mdir / "bench.json"
    cmd = [sys.executable, "gpu/bench.py", "--model", str(MODEL_DIR), "--out", str(out)]
    if os.environ.get("MC3_BENCH_SUITE") != "1":
        cmd.append("--skip-suite")
    rc, _ = sh(cmd, cwd=SRC, log=mdir / "bench.log")
    say(f"[bench] rc={rc} -> {out}  (rc 3 = ran but a gate failed)")
    return rc in (0, 3)


def suite_list() -> list[tuple[str, Path, Path]]:
    """(name, corpus_dir, questions.json): the sample, then eval/suites/*/ that have questions.json.
    A suite's corpus is <suite>/corpus, <suite>/mc3-corpus or the suite dir itself."""
    items = [("sample", SRC / "sample-corpus", SRC / "sample-questions.json")]
    sdir = SRC / "eval" / "suites"
    if sdir.is_dir():
        for d in sorted(p for p in sdir.iterdir() if p.is_dir() and (p / "questions.json").exists()):
            corpus = next((d / c for c in ("corpus", "mc3-corpus") if (d / c).is_dir()), d)
            items.append((d.name, corpus, d / "questions.json"))
    want = [s.strip() for s in os.environ.get("MC3_SUITES", "").split(",") if s.strip()]
    unknown = [w for w in want if w not in {i[0] for i in items}]
    if unknown:
        say(f"[suite] WARNING: MC3_SUITES names not found: {unknown} (available: {[i[0] for i in items]})")
    return [i for i in items if not want or i[0] in want]


def phase_suite() -> bool:
    banner("suite")
    if (why := _need_ready()):
        say(f"[suite] SKIP: {why}")
        return False
    if not (SRC / "eval" / "run_suite.py").exists():
        say("[suite] SKIP: eval/run_suite.py not in src.tar")
        return False
    run = current_run(create=True)
    mdir = run / MODEL_SLUG
    say(f"[suite] model {MODEL_SLUG} ({MODEL_DIR}) -> {mdir}")
    ok_all = True
    summary = {}
    for name, corpus, questions in suite_list():
        out = mdir / name
        out.mkdir(parents=True, exist_ok=True)
        say(f"\n--- suite {name}: corpus={corpus} questions={questions}")
        t0 = time.time()
        rc, tail = sh([sys.executable, "eval/run_suite.py", "--corpus", str(corpus), "--questions", str(questions),
                       "--out", str(out), "--reader", "real", "--model", str(MODEL_DIR)],
                      cwd=SRC, log=out / "suite_stdout.txt")
        total = next((l for l in reversed(tail) if l.startswith("TOTAL ")), None)
        summary[name] = {"rc": rc, "seconds": round(time.time() - t0, 1), "score_line": total}
        say(f"[suite] {name}: rc={rc} {total or 'no TOTAL line'}")
        ok_all &= (rc == 0)
    (mdir / "suite_summary.json").write_text(json.dumps(summary, indent=2))
    return ok_all


def ab_configs_arg() -> str | None:
    """MC3_AB_CONFIGS: an inline JSON string, or a path (relative to SRC, or absolute) to a JSON file."""
    raw = os.environ.get("MC3_AB_CONFIGS", "").strip()
    if not raw:
        return None
    if raw.startswith(("[", "{")):
        return raw
    p = Path(raw)
    return str(p if p.is_absolute() else SRC / p)


def phase_ab() -> bool:
    """eval/ab_suite.py: one model load, many configs, shared reader cache; prints the summary table."""
    banner("ab")
    if (why := _need_ready()):
        say(f"[ab] SKIP: {why}")
        return False
    if not (SRC / "eval" / "ab_suite.py").exists():
        say("[ab] SKIP: eval/ab_suite.py not in src.tar")
        return False
    configs = ab_configs_arg()
    if configs is None:
        say("[ab] SKIP: set MC3_AB_CONFIGS (JSON list of {name, env} or a path relative to src)")
        return False
    if not configs.lstrip().startswith(("[", "{")) and not Path(configs).exists():
        say(f"[ab] SKIP: MC3_AB_CONFIGS file {configs} not found")
        return False
    suites = [s.strip() for s in os.environ.get("MC3_SUITES", "").split(",") if s.strip()]
    if not suites:
        suites = [name for name, _, _ in suite_list()]
    run = current_run(create=True)
    out = run / "ab"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "eval/ab_suite.py", "--suites", ",".join(suites), "--configs", configs,
           "--reader", "real", "--model", str(MODEL_DIR), "--out", str(out), "--cache", str(BASE / "cache")]
    for g in (x.strip() for x in os.environ.get("MC3_AB_GROUPS", "").split(";")):
        if g:
            cmd += ["--group", g]
    for var, flag in (("MC3_AB_SOFT", "--soft"), ("MC3_AB_LIMIT", "--limit")):
        if os.environ.get(var, "").strip():
            cmd += [flag, os.environ[var].strip()]
    say(f"[ab] suites={suites} -> {out}")
    rc, _ = sh(cmd, cwd=SRC, log=out / "ab_stdout.txt")
    summ = out / "ab_summary.json"
    try:   # the child printed the table last; show it again after the (long) progress log
        lines = (out / "ab_stdout.txt").read_text(errors="replace").splitlines()
        start = max((i for i, l in enumerate(lines) if l.startswith("A/B summary")), default=None)
        if start is not None:
            say("\n" + "=" * 8 + " ab summary " + "=" * 8)
            say("\n".join(lines[start:]))
    except OSError as e:
        say(f"[ab] could not re-print the table: {e!r}")
    say(f"[ab] rc={rc} -> {summ}   download: colab download -s mc3-gpu {summ}")
    return rc == 0


def phase_harness() -> bool:
    """Real server.py (supervisor + child, real model) + app.py subprocesses per question, like the harness."""
    banner("harness")
    if (why := _need_ready()):
        say(f"[harness] SKIP: {why}")
        return False
    run = current_run(create=True)
    mdir = run / MODEL_SLUG
    ok_all = True
    for name, corpus, questions in suite_list():
        out = mdir / f"harness_{name}"
        out.mkdir(parents=True, exist_ok=True)
        vram = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                 "-lms", "500"], stdout=open(out / "vram.txt", "w"), stderr=subprocess.DEVNULL)
        rc, tail = sh([sys.executable, "eval/harness_sim.py", "--corpus", str(corpus), "--questions", str(questions),
                       "--out", str(out), "--reader", "real"], cwd=SRC, env=child_env(), log=out / "harness.txt")
        vram.terminate()
        try:
            peak = max(int(x) for x in (out / "vram.txt").read_text().split() if x.strip().isdigit())
        except ValueError:
            peak = -1
        total = next((l for l in reversed(tail) if l.startswith("TOTAL ")), None)
        say(f"[harness] {name}: rc={rc} {total or 'no TOTAL line'}  peak VRAM {peak} MiB")
        for l in tail:
            if l.startswith(("--index", "server status", "FAILURES", "  ", "harness_sim OK", "server ready")):
                say("    " + l[:300])
        ok_all &= (rc == 0)
    return ok_all


# ----------------------------------------------------------------------------- report

def _load_bench_module():
    """Fresh import of gpu/bench.py from SRC (kernel module cache must not serve a stale copy)."""
    f = SRC / "gpu" / "bench.py"
    if not f.exists():
        return None
    spec = importlib.util.spec_from_file_location("mc3_bench_fresh", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _score_line(txt: Path) -> tuple[int | None, int | None]:
    try:
        for line in reversed(txt.read_text().splitlines()):
            if line.startswith("TOTAL "):
                a, b = line.split()[1].split("/")
                return int(a), int(b)
    except Exception:
        pass
    return None, None


def _bench_summary(b: dict) -> dict:
    ph = b.get("phases", {})
    env = ph.get("env", {})
    sizes = ph.get("read", {}).get("sizes", {})
    return {
        "device": env.get("device_name"), "torch": env.get("torch"), "vram_total_gib": env.get("total_vram_gib"),
        "load_s": ph.get("load", {}).get("seconds"), "warmup_s": ph.get("warmup", {}).get("seconds"),
        "sdpa_force_math": ph.get("sdpa", {}).get("force_math"),
        "sdpa_skipped": ph.get("sdpa", {}).get("skipped"),
        "read_median_s": {k: v.get("median_s") for k, v in sizes.items()},
        "read8k_p95_s": sizes.get("8000", {}).get("p95_s"),
        "transcribe_max_s": ph.get("transcribe", {}).get("max_seconds"),
        "logprob_median_s": ph.get("logprob", {}).get("median_s"),
        "vram_peak_gib": b.get("vram", {}).get("peak_gib"),
        "gates": {k: v.get("status") for k, v in b.get("gates", {}).items()},
        "phase_errors": {k: v.get("error") or v.get("skip") for k, v in ph.items() if not v.get("ok", True)},
    }


def _is_suite_dir(d: Path) -> bool:
    return (d / "suite_stdout.txt").exists() or (d / "report.json").exists()


def _suite_summary(d: Path, bench_mod) -> dict | None:
    s: dict = {}
    tot, mx = _score_line(d / "suite_stdout.txt")
    s["score"], s["max"] = tot, mx
    rep = d / "report.json"
    if rep.exists() and bench_mod is not None:
        s.update(bench_mod.summarise_suite(d))
    return s if (tot is not None or rep.exists()) else None


def phase_report() -> bool:
    banner("report")
    run = current_run(create=False)
    if run is None:
        say("[report] no run directory yet -- run bench/suite first")
        return False
    bench_mod = _load_bench_module()
    raw: dict = {}
    for f in sorted(run.rglob("*.json")):
        try:
            if f.stat().st_size <= 400_000:
                raw[str(f.relative_to(run))] = json.loads(f.read_text())
        except Exception as e:
            raw[str(f.relative_to(run))] = {"unreadable": repr(e)}
    # layout: run/<model_slug>/{bench.json, <suite>/}; runs from before multi-model support have
    # run/bench.json and run/<suite>/ directly -> reported as model "(legacy)"
    models: dict = {}

    def model(name: str) -> dict:
        return models.setdefault(name, {"bench": None, "suites": {}})

    if raw.get("bench.json"):
        model("(legacy)")["bench"] = _bench_summary(raw["bench.json"])
    for d in sorted(p for p in run.iterdir() if p.is_dir()):
        if _is_suite_dir(d):
            if (s := _suite_summary(d, bench_mod)) is not None:
                model("(legacy)")["suites"][d.name] = s
            continue
        m = model(d.name)
        if raw.get(f"{d.name}/bench.json"):
            m["bench"] = _bench_summary(raw[f"{d.name}/bench.json"])
        for sd in sorted(p for p in d.iterdir() if p.is_dir() and _is_suite_dir(p)):
            if (s := _suite_summary(sd, bench_mod)) is not None:
                m["suites"][sd.name] = s
    models = {k: v for k, v in models.items() if v["bench"] or v["suites"]}
    primary = MODEL_SLUG if MODEL_SLUG in models else next(iter(models), None)
    summary: dict = {"run": run.name, "generated": time.strftime("%FT%T"), "models": models,
                     # single-model view kept for older consumers
                     "bench": models[primary]["bench"] if primary else None,
                     "suites": models[primary]["suites"] if primary else {}, "raw": raw}
    out = RUNS / "latest.json"
    out.write_text(json.dumps(summary, indent=2, default=str))

    say(f"run {run.name}")
    if not models:
        say("  (no bench or suite results in this run)")
    for name, m in models.items():
        say(f"\n  === model {name} ===")
        sb = m["bench"]
        if sb:
            say(f"  device {sb['device']}  torch {sb['torch']}  VRAM {sb['vram_total_gib']} GiB")
            say(f"  {'gate':<26}status")
            for k, v in sb["gates"].items():
                say(f"  {k:<26}{v}")
            say(f"  load {sb['load_s']} s | warmup {sb['warmup_s']} s | read8k p95 {sb['read8k_p95_s']} s | "
                f"transcribe max {sb['transcribe_max_s']} s | logprob {sb['logprob_median_s']} s | "
                f"peak {sb['vram_peak_gib']} GiB | force_math {sb['sdpa_force_math']}"
                + (f" | sdpa skipped: {sb['sdpa_skipped']}" if sb.get("sdpa_skipped") else ""))
            if sb["phase_errors"]:
                say(f"  phase problems: {sb['phase_errors']}")
        else:
            say("  (no bench.json for this model)")
        if m["suites"]:
            say(f"\n  {'suite':<22}{'score':>10}{'n':>5}{'p50 s':>8}{'p95 s':>8}{'max s':>8}")
            for n, s in m["suites"].items():
                sc = f"{s['score']}/{s['max']}" if s.get("score") is not None else "-"
                say(f"  {n:<22}{sc:>10}{s.get('n', '-')!s:>5}{s.get('p50_s', '-')!s:>8}{s.get('p95_s', '-')!s:>8}"
                    f"{s.get('max_s', '-')!s:>8}")
        else:
            say("  (no suite results for this model)")
    if len(models) > 1:
        names = sorted({n for m in models.values() for n in m["suites"]})
        if names:
            say("\n  A/B score (suite x model)")
            say(f"  {'suite':<14}" + "".join(f"{k[:24]:>26}" for k in models))
            for n in names:
                cells = []
                for m in models.values():
                    s = m["suites"].get(n)
                    cells.append(f"{s['score']}/{s['max']}" if s and s.get("score") is not None else "-")
                say(f"  {n:<14}" + "".join(f"{c:>26}" for c in cells))
    say(f"\n[report] wrote {out}   download: colab download -s mc3-gpu {out} runs/latest.json")
    return True


PHASES = {"setup": phase_setup, "weights": phase_weights, "bench": phase_bench, "suite": phase_suite,
          "harness": phase_harness, "ab": phase_ab,
          "report": phase_report}


def main() -> int:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 0
    wanted = [p.strip() for p in os.environ.get("MC3_PHASE", "").split(",") if p.strip()]
    if wanted == ["all"]:
        wanted = list(ALL_PHASES)
    bad = [p for p in wanted if p not in PHASES]
    if not wanted or bad:
        say(f"MC3_PHASE must be a comma list of {ALL_PHASES} or 'all' (got {os.environ.get('MC3_PHASE')!r}"
            f"{', unknown: ' + str(bad) if bad else ''}).  With colab exec pass it as: --env MC3_PHASE=setup,weights")
        return 2
    BASE.mkdir(parents=True, exist_ok=True)
    # one-shot flags: `colab exec --env` leaves them in the kernel's os.environ, so consume them
    update = os.environ.pop("MC3_UPDATE", "") == "1"
    new_run = os.environ.pop("MC3_NEW_RUN", "") == "1"
    if update:
        banner("update (MC3_UPDATE=1)")
        extract_src(force=True)
    if new_run and any(p in ("bench", "suite", "ab") for p in wanted):
        (RUNS).mkdir(parents=True, exist_ok=True)
        (RUNS / "CURRENT").write_text(time.strftime("%Y%m%d-%H%M%S"))
        (RUNS / (RUNS / "CURRENT").read_text()).mkdir(parents=True, exist_ok=True)
        say(f"[run] new run {(RUNS / 'CURRENT').read_text()}")
    say(f"[model] {REPO_ID} @ {REVISION[:12]} dir={MODEL_DIR} slug={MODEL_SLUG}")
    results = {}
    for p in wanted:
        t0 = time.time()
        try:
            results[p] = bool(PHASES[p]())
        except Exception as e:
            import traceback
            traceback.print_exc()
            say(f"[{p}] EXCEPTION {e!r}")
            results[p] = False
        say(f"[{p}] {'ok' if results[p] else 'FAILED'} in {time.time() - t0:.0f}s")
    banner("summary")
    say("  " + "  ".join(f"{p}={'ok' if v else 'FAIL'}" for p, v in results.items()))
    for k in MODEL_ENV + ("MC3_MODEL",):     # one-shot: `colab exec --env` would otherwise leak this model into the next exec
        os.environ.pop(k, None)
    return 0 if all(results.values()) else 1


# `colab exec` runs this file as a notebook cell (__name__ == "__main__", ipykernel loaded); importing it for a test does not run it
_rc = main() if (__name__ == "__main__" or "ipykernel" in sys.modules) else 0
if _rc:
    sys.exit(_rc)
