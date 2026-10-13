#!/usr/bin/env python3
"""GPU bench for the MC3 reader (CUDA on Colab, ROCm on the AMD notebook).

    python3 gpu/bench.py --model DIR --out runs/<ts>/bench.json [--corpus sample-corpus] [--skip-suite]

Phases (each in its own try/except; the JSON is rewritten after every phase, so a crash keeps what ran):
  env  load  sdpa  warmup  transcribe  read  logprob  gates  suite
Exit code: 0 = ran, all gates PASS; 3 = ran, at least one gate FAIL; 2 = bad arguments.
torch / transformers are imported lazily: `--help` and `import bench` stay light.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Spec §10 GPU gates
GATES = {
    "load_s": 90.0,
    "warmup_s": 30.0,
    "read8k_p95_s": 8.0,
    "transcribe_s_per_image": 30.0,
    "vram_peak_gib": 40.0,
}
TARGET_TOKENS = (4_000, 8_000, 14_000)
DEFAULT_QUESTION = "What is the maximum junction temperature of the TQ-40?"
PLANTED_VALUE = "94"
TRANSCRIBE_IMAGES = [
    ("specs/backplane_pinout.png", ["B14", "THERM_ALERT#"]),
    ("support/asset_label.jpg", ["REV-C2"]),
]


# ----------------------------------------------------------------------------- helpers

def norm(s: str) -> str:
    """Bench-side normalisation: no spaces / hyphens / underscores, uppercase."""
    return "".join(ch for ch in str(s) if not ch.isspace() and ch not in "-_").upper()


def pctl(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    return v[max(0, math.ceil(p * len(v)) - 1)]


def say(msg: str = "") -> None:
    print(msg, flush=True)


class Bench:
    """Collects phase results, prints the live table and writes the JSON after each phase."""

    def __init__(self, out_path: Path, args: argparse.Namespace):
        self.out_path = out_path
        self.data: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv[1:],
                           "model": args.model, "phases": {}, "gates": {}}
        self.reader = None
        self.sampler = None
        self.args = args

    def save(self) -> None:
        try:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2, default=str))
            os.replace(tmp, self.out_path)
        except Exception as e:  # never lose the run over a write error
            say(f"[bench] WARN could not write {self.out_path}: {e!r}")

    def phase(self, name: str, fn, *a, **kw) -> dict:
        t0 = time.time()
        say(f"\n=== phase {name} " + "=" * max(3, 60 - len(name)))
        try:
            res = fn(*a, **kw) or {}
            res.setdefault("ok", True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        res["wall_s"] = round(time.time() - t0, 3)
        self.data["phases"][name] = res
        flag = "ok " if res.get("ok") else "ERR"
        extra = res.get("error") or res.get("skip") or ""
        say(f"[{flag}] {name:<12} {res['wall_s']:>8.2f} s  {extra}")
        self.save()
        return res


def _sync() -> None:
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _deadline(seconds: float):
    from budget import Deadline  # app/budget.py (lazy; PYTHONPATH has app/)
    if hasattr(Deadline, "after"):
        return Deadline.after(seconds)
    return Deadline(time.time() + seconds)


def _text_of(out) -> str:
    """Reader results vary in shape (str / ReaderOut / dict): pull out the human text."""
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        return str(out.get("answer", out.get("text", "")))
    for attr in ("answer", "text"):
        if hasattr(out, attr):
            return str(getattr(out, attr))
    return str(out)


def _transcript_of(out) -> str:
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        return str(out.get("text", out.get("transcript", out)))
    return str(getattr(out, "text", out))


# ----------------------------------------------------------------------------- evidence packages

def _tok_counter(reader):
    """tokens(text) via the processor tokenizer; chars/3.5 when the reader exposes none."""
    tok = None
    try:
        proc = getattr(reader, "processor", None)
        tok = getattr(proc, "tokenizer", None) or getattr(reader, "tokenizer", None)
    except Exception:
        tok = None
    if tok is not None:
        def count(text: str) -> int:
            return len(tok(text, add_special_tokens=False)["input_ids"])
        return count, "processor.tokenizer"
    return (lambda text: int(len(text) / 3.5)), "chars/3.5 (estimate)"


_FILES = [("specs/tq40_datasheet_r2.pdf", "pdf"), ("specs/tq60_preliminary.pdf", "pdf"),
          ("support/rma_parts.xlsx", "xlsx"), ("engineering/ingest_service.py", "py")]
_PARAMS = ["Supply voltage", "Idle power", "Peak power", "Fan speed", "Coolant flow", "Inlet temperature",
           "Memory bandwidth", "Clock frequency", "PCIe lanes", "Shock rating", "Storage capacity", "Boot time"]
_UNITS = ["V", "W", "RPM", "L/min", "C", "GB/s", "MHz", "lanes", "g", "GB", "s"]


def build_package(count, target_tokens: int, question: str = DEFAULT_QUESTION, seed: int = 7) -> dict:
    """Synthetic evidence package of ~target_tokens made of realistic record lines with one planted
    answer (TQ-40 Tj max = 94 C) and a near-miss decoy (TQ-20 = 88 C)."""
    rng = random.Random(seed)

    def filler(i: int) -> str:
        f, _ = _FILES[i % len(_FILES)]
        prod = rng.choice(["TQ-40", "TQ-60", "TQ-20"])
        return (f"[R{i}] {f} p1.L{i}: Parameter {i} {rng.choice(_PARAMS)} ({prod}) ... "
                f"{rng.randint(1, 9999) / rng.choice([1, 10, 100])} {rng.choice(_UNITS)}")

    def assemble(n: int) -> tuple[str, int]:
        plant_at, decoy_at = n // 2, n // 4
        lines = []
        for i in range(n):
            if i == plant_at:
                lines.append(f"[R{i}] specs/tq40_datasheet_r2.pdf p1.L{i}: Maximum junction temperature (TQ-40) ... "
                             f"{PLANTED_VALUE} C")
            elif i == decoy_at:
                lines.append(f"[R{i}] specs/tq20_datasheet.pdf p1.L{i}: Maximum junction temperature (TQ-20) ... 88 C")
            else:
                lines.append(filler(i))
        text = "## specs/tq40_datasheet_r2.pdf (pdf)\n" + "\n".join(lines)  # Reader adds the EVIDENCE header
        return text, plant_at

    # calibrate tokens/line on a sample, then correct with real counts (max 6 rounds)
    sample, _ = assemble(120)
    per_line = max(1.0, count(sample) / 120.0)
    n = max(10, int(target_tokens / per_line))
    text, plant_at = assemble(n)
    toks = count(text)
    for _ in range(6):
        if abs(toks - target_tokens) <= 0.03 * target_tokens:
            break
        n = max(10, int(n * target_tokens / max(toks, 1)))
        text, plant_at = assemble(n)
        toks = count(text)
    return {"text": text, "tokens": toks, "records": n, "planted_rid": plant_at, "question": question}


# ----------------------------------------------------------------------------- phases

def phase_env(b: Bench) -> dict:
    info: dict = {"python": platform.python_version(), "platform": platform.platform()}
    import torch
    info["torch"] = torch.__version__
    info["backend_tag"] = ("rocm" if "+rocm" in torch.__version__ else "cuda" if "+cu" in torch.__version__ else "other")
    info["torch_hip"] = getattr(torch.version, "hip", None)
    info["torch_cuda"] = getattr(torch.version, "cuda", None)
    info["torch_has_rocm_tag"] = "+rocm" in torch.__version__
    info["torch_has_cu_tag"] = "+cu" in torch.__version__
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception as e:
        info["transformers"] = f"import failed: {e!r}"
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        info["device_name"] = p.name
        info["total_vram_gib"] = round(p.total_memory / 2**30, 2)
        info["arch"] = getattr(p, "gcnArchName", None) or f"sm_{p.major}{p.minor}"
        info["device_count"] = torch.cuda.device_count()
    else:
        info["ok"] = False
        info["error"] = "no GPU visible to torch (torch.cuda.is_available() is False)"
    say(f"  torch {info['torch']} ({info['backend_tag']})  transformers {info.get('transformers')}  "
        f"device {info.get('device_name')}  VRAM {info.get('total_vram_gib')} GiB")
    return info


def phase_load(b: Bench) -> dict:
    import torch
    from reader import Reader  # app/reader.py
    try:
        r = Reader(b.args.model, "cuda")
    except TypeError:
        r = Reader(b.args.model)
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    _sync()
    t0 = time.time()
    r.load()
    _sync()
    secs = time.time() - t0
    b.reader = r
    snap = b.sampler.sample_now() if b.sampler else {}
    res = {"seconds": round(secs, 3),
           "vram_after_load": {"torch_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                               "torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                               "sampler_used_mib": snap.get("used_mib")},
           "attn": getattr(r, "attn", None)}
    say(f"  load {secs:.1f} s  attn={res['attn']}  torch alloc {res['vram_after_load']['torch_allocated_mib']:.0f} MiB  "
        f"sampler {res['vram_after_load']['sampler_used_mib']} MiB")
    return res


def _need_reader(b: Bench) -> dict | None:
    if b.reader is None:
        return {"ok": False, "skip": "SKIP: reader not loaded (load phase failed)"}
    return None


def phase_sdpa(b: Bench) -> dict:
    if (s := _need_reader(b)):
        return s
    _sync()
    t0 = time.time()
    out = b.reader.self_test_sdpa()
    _sync()
    secs = time.time() - t0
    res = {"seconds": round(secs, 3), "force_math": bool(getattr(b.reader, "force_math", False)), "ran": out}
    skipped = getattr(b.reader, "sdpa_skipped", None)
    if skipped:  # deliberate skip (non-Qwen family without a vision-attention path): not a failure
        res["skipped"] = skipped
    elif out is False:  # Reader.self_test_sdpa() returns False when the test itself crashed (it never raises)
        res["ok"] = False
        res["error"] = "self_test_sdpa did not run (see stderr: 'sdpa self-test failed to run')"
    say(f"  self_test_sdpa {secs:.2f} s  force_math={res['force_math']}")
    return res


def phase_warmup(b: Bench) -> dict:
    if (s := _need_reader(b)):
        return s
    _sync()
    t0 = time.time()
    b.reader.warmup()
    _sync()
    secs = time.time() - t0
    tm = dict(getattr(b.reader, "timings", {}) or {})
    res = {"seconds": round(secs, 3), "read_s": tm.get("warmup_read_s"), "logprob_s": tm.get("warmup_logprob_s")}
    if tm.get("warmup_logprob_s") is None:  # Reader.warmup() swallows errors; the last timing is only set on success
        res["ok"] = False
        res["error"] = "warmup swallowed an error (no warmup_logprob_s timing; see stderr 'warmup failed')"
    say(f"  warmup {secs:.2f} s  (read {res['read_s']}, logprob {res['logprob_s']})")
    return res


def _corpus_path(b: Bench, rel: str) -> Path:
    for base in (Path(b.args.corpus), REPO / b.args.corpus):
        if (base / rel).exists():
            return base / rel
    raise FileNotFoundError(f"{rel} not under corpus {b.args.corpus}")


def phase_transcribe(b: Bench) -> dict:
    if (s := _need_reader(b)):
        return s
    from PIL import Image
    items = []
    say(f"  {'image':<32}{'sec':>8}{'words':>7}{'w/s':>7}  found")
    for rel, needles in TRANSCRIBE_IMAGES:
        item: dict = {"image": rel}
        try:
            img = Image.open(_corpus_path(b, rel)).convert("RGB")
            item["size"] = list(img.size)
            _sync()
            t0 = time.time()
            out = b.reader.transcribe(img, _deadline(60))
            _sync()
            secs = time.time() - t0
            text = _transcript_of(out)
            words = len(text.split())
            if not text.strip():
                item["ok"] = False
                item["error"] = "empty transcript (Reader.transcribe swallows errors; see stderr 'transcribe failed')"
            hay = norm(text)
            found = {n: (norm(n) in hay) for n in needles}
            item.update(seconds=round(secs, 3), tokens=words, words_per_s=round(words / secs, 2) if secs else None,
                        found=found, all_found=all(found.values()), text_head=text[:300],
                        timings=dict(getattr(b.reader, "timings", {}) or {}))
            say(f"  {rel:<32}{secs:>8.2f}{words:>7}{item['words_per_s'] or 0:>7.1f}  "
                + " ".join(f"{k}={'Y' if v else 'N'}" for k, v in found.items()))
        except Exception as e:
            item.update(ok=False, error=f"{type(e).__name__}: {e}")
            say(f"  {rel:<32} ERROR {item['error']}")
        items.append(item)
    secs = [i["seconds"] for i in items if "seconds" in i and i.get("ok", True)]
    bad = [i["image"] for i in items if not i.get("ok", True)]
    return {"ok": not bad, **({"error": f"failed: {bad}"} if bad else {}), "images": items, "max_seconds": max(secs) if secs else None,
            "all_found": all(i.get("all_found") for i in items) if items else False}


def _question(b: Bench) -> str:
    try:
        qs = json.loads((REPO / "sample-questions.json").read_text())["queries"]
        return qs[0]["query"]
    except Exception:
        return DEFAULT_QUESTION


def _timed_read(b: Bench, pkg: dict, images: list, max_new: int | None = None) -> dict:
    _sync()
    t0 = time.time()
    kw = {"max_new_tokens": max_new} if max_new else {}
    out = b.reader.read(pkg["text"], pkg["question"], images, _deadline(60), **kw)
    _sync()
    secs = time.time() - t0
    ans = _text_of(out)
    raw = (getattr(out, "raw", None) or "") if not isinstance(out, (str, dict)) else ""
    res = {"seconds": round(secs, 3), "answer": ans[:80],
           "new_tokens": getattr(out, "n_new_tokens", None) if not isinstance(out, (str, dict)) else None,
           "planted_ok": norm(ans) == norm(PLANTED_VALUE), "raw": raw[:200]}
    if raw.startswith("ERROR"):  # Reader.read() never raises; it returns raw="ERROR: ..."
        res["ok"] = False
        res["error"] = raw[:200]
    return res


def phase_read(b: Bench) -> dict:
    if (s := _need_reader(b)):
        return s
    from PIL import Image
    count, how = _tok_counter(b.reader)
    q = _question(b)
    res: dict = {"tokenizer": how, "sizes": {}}
    say(f"  token estimate via {how}")
    say(f"  {'package':<14}{'tokens':>8}{'rep':>4}{'sec':>8}{'new':>5}  planted")
    for target in TARGET_TOKENS:
        pkg = build_package(count, target, q)
        reps = b.args.reps if target == 8_000 else 2
        runs = []
        for k in range(reps):
            try:
                r = _timed_read(b, pkg, [])
            except Exception as e:
                r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            runs.append(r)
            say(f"  {'~%dk' % (target // 1000):<14}{pkg['tokens']:>8}{k + 1:>4}{r.get('seconds', 0):>8.2f}"
                f"{str(r.get('new_tokens')):>5}  {r.get('planted_ok', r.get('error'))}")
        secs = [r["seconds"] for r in runs if "seconds" in r and r.get("ok", True)]  # errored reads are fast: never count them
        prefill = None
        try:  # prefill-only: 1 new token -> separates prefill from decode (decode ms/token = remainder)
            prefill = _timed_read(b, pkg, [], max_new=1)["seconds"]
        except Exception:
            pass
        nt = next((r.get("new_tokens") for r in runs if r.get("new_tokens")), None)
        dec = None
        if prefill is not None and secs and nt and nt > 1:
            dec = round(max(0.0, statistics.median(secs) - prefill) / (nt - 1) * 1000, 1)
        say(f"  {'':<14}prefill-only {prefill} s  decode {dec} ms/token")
        res["sizes"][str(target)] = {"prefill_only_s": prefill, "decode_ms_per_token": dec,
            "package_tokens": pkg["tokens"], "records": pkg["records"], "runs": runs,
            "first_s": secs[0] if secs else None,
            "median_s": round(statistics.median(secs), 3) if secs else None,
            "p95_s": pctl(secs, 0.95), "max_s": max(secs) if secs else None,
            "planted_all": all(r.get("planted_ok") for r in runs) if runs else False}
    bad = [k for k, v in res["sizes"].items() if any(not r.get("ok", True) for r in v["runs"])]
    if bad:
        res["ok"] = False
        res["error"] = f"read errors at sizes {bad}: {next((r.get('error') for v in res['sizes'].values() for r in v['runs'] if r.get('error')), '')}"
    # 8k + one image attached
    try:
        pkg = build_package(count, 8_000, q)
        img = Image.open(_corpus_path(b, "support/asset_label.jpg")).convert("RGB")
        runs = [_timed_read(b, pkg, [img]) for _ in range(2)]
        secs = [r["seconds"] for r in runs]
        res["image_8k"] = {"package_tokens": pkg["tokens"], "runs": runs, "median_s": round(statistics.median(secs), 3),
                           "max_s": max(secs), "planted_all": all(r["planted_ok"] for r in runs)}
        say(f"  {'~8k + 1 image':<14}{pkg['tokens']:>8}{'':>4}{statistics.median(secs):>8.2f}  "
            f"{res['image_8k']['planted_all']}")
    except Exception as e:
        res["image_8k"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        say(f"  8k+image ERROR {res['image_8k']['error']}")
    return res


def phase_logprob(b: Bench) -> dict:
    if (s := _need_reader(b)):
        return s
    if not hasattr(b.reader, "logprob"):
        return {"ok": False, "skip": "SKIP: Reader has no logprob()"}
    count, _ = _tok_counter(b.reader)
    pkg = build_package(count, 8_000, _question(b))
    runs, vals = [], []
    for _ in range(3):
        _sync()
        t0 = time.time()
        v = b.reader.logprob(pkg["text"], pkg["question"], [], PLANTED_VALUE)
        _sync()
        runs.append(round(time.time() - t0, 3))
        vals.append(v if isinstance(v, (int, float)) else None)
    med = statistics.median(runs)
    say(f"  logprob @~8k: {runs} s  median {med:.2f} s  values {vals}")
    return {"seconds": runs, "median_s": round(med, 3), "values": vals, "package_tokens": pkg["tokens"]}


def phase_gates(b: Bench) -> dict:
    ph = b.data["phases"]
    torch = sys.modules.get("torch")
    info = b.sampler.stop() if b.sampler else {"peak_mib": 0.0, "samples": 0, "source": "none"}
    b.sampler = None
    torch_peak = None
    try:
        torch_peak = torch.cuda.max_memory_allocated() / 2**20 if torch is not None else None
    except Exception:
        pass
    peak_mib = max(x for x in (info.get("peak_mib"), torch_peak, 0.0) if x is not None)
    vram = {"sampler": info, "torch_max_memory_allocated_mib": round(torch_peak, 1) if torch_peak else None,
            "peak_mib": round(peak_mib, 1), "peak_gib": round(peak_mib / 1024, 2)}

    def g(name, value, limit, detail=""):
        if value is None:
            return {"status": "FAIL", "value": None, "limit": limit, "detail": "no data " + detail}
        return {"status": "PASS" if value <= limit else "FAIL", "value": round(value, 3), "limit": limit, "detail": detail}

    sizes = ph.get("read", {}).get("sizes", {})
    gates = {
        "load_s": g("load", ph.get("load", {}).get("seconds"), GATES["load_s"]),
        "warmup_s": g("warmup", ph.get("warmup", {}).get("seconds"), GATES["warmup_s"]),
        "read8k_p95_s": g("read", sizes.get("8000", {}).get("p95_s"), GATES["read8k_p95_s"], "p95 over 8k text-only reps"),
        "transcribe_s_per_image": g("tr", ph.get("transcribe", {}).get("max_seconds"), GATES["transcribe_s_per_image"],
                                    "slowest image"),
        "vram_peak_gib": g("vram", vram["peak_gib"] if peak_mib > 0 else None, GATES["vram_peak_gib"],
                           f"source={info.get('source')}"),
    }
    b.data["gates"] = gates
    b.data["vram"] = vram
    say(f"\n  {'gate':<26}{'value':>10}{'limit':>8}  status")
    for k, v in gates.items():
        say(f"  {k:<26}{str(v['value']):>10}{v['limit']:>8}  {v['status']} {v['detail']}")
    return {"gates": gates, "vram": vram, "all_pass": all(v["status"] == "PASS" for v in gates.values())}


def _free_reader(b: Bench) -> None:
    b.reader = None
    try:
        import gc
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def phase_suite(b: Bench) -> dict:
    script = REPO / "eval" / "run_suite.py"
    if not script.exists():
        return {"ok": False, "skip": "SKIP: eval/run_suite.py not present"}
    corpus = next((p for p in (Path(b.args.corpus), REPO / b.args.corpus) if p.exists()), Path(b.args.corpus))
    questions = REPO / "sample-questions.json"
    out_dir = b.out_path.parent / "suite_sample"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "app"), str(REPO), env.get("PYTHONPATH", "")])
    cmd = [sys.executable, str(script), "--corpus", str(corpus), "--questions", str(questions),
           "--out", str(out_dir), "--reader", "real", "--model", b.args.model]
    say("  $ " + " ".join(cmd))
    sys.path.insert(0, str(HERE))
    import vram_sampler
    samp = vram_sampler.start(str(b.out_path.parent / "vram_suite.csv"))
    tail: list[str] = []
    t0 = time.time()
    rc = None
    try:
        p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             bufsize=1)
        for line in p.stdout:
            sys.stdout.write("    | " + line)
            sys.stdout.flush()
            tail.append(line.rstrip("\n"))
            tail = tail[-80:]
        rc = p.wait()
    finally:
        vram = samp.stop()
    secs = time.time() - t0
    res: dict = {"returncode": rc, "seconds": round(secs, 1), "out_dir": str(out_dir), "vram": vram}
    for line in reversed(tail):
        if line.startswith("TOTAL "):
            res["score_line"] = line
            break
    res["summary"] = summarise_suite(out_dir)
    if rc != 0:
        res["ok"] = False
        res["error"] = f"run_suite exited {rc}"
    return res


def summarise_suite(out_dir: Path) -> dict:
    """Per-question seconds from run_suite's report.json (shape tolerant)."""
    f = out_dir / "report.json"
    if not f.exists():
        return {"report": None}
    try:
        data = json.loads(f.read_text())
    except Exception as e:
        return {"report": str(f), "error": repr(e)}
    rows = []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for k in ("questions", "results", "rows", "per_question", "queries"):
            if isinstance(data.get(k), list):
                rows = data[k]
                break
    secs = []
    for r in rows:
        if isinstance(r, dict):
            for k in ("seconds", "elapsed_s", "elapsed", "sec", "time_s"):
                if isinstance(r.get(k), (int, float)):
                    secs.append(float(r[k]))
                    break
    out = {"report": str(f), "n": len(rows)}
    if secs:
        out.update(p50_s=round(statistics.median(secs), 2), p95_s=round(pctl(secs, 0.95), 2), max_s=round(max(secs), 2))
    if isinstance(data, dict):
        for k in ("total", "score", "max", "refusals", "reasons"):
            if k in data:
                out[k] = data[k]
    return out


# ----------------------------------------------------------------------------- main

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MC3 GPU bench (load / sdpa / warmup / transcribe / read / logprob / VRAM / suite)")
    ap.add_argument("--model", required=True, help="Qwen3-VL-8B-Instruct directory")
    ap.add_argument("--out", default=None, help="bench.json path (default runs/<ts>/bench.json)")
    ap.add_argument("--corpus", default="sample-corpus", help="corpus dir (default sample-corpus)")
    ap.add_argument("--skip-suite", action="store_true", help="do not run eval/run_suite.py at the end")
    ap.add_argument("--reps", type=int, default=5, help="repetitions of the 8k read (p95 gate); others use 2")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out) if args.out else Path("runs") / time.strftime("%Y%m%d-%H%M%S") / "bench.json"
    for p in (str(REPO / "app"), str(REPO), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    b = Bench(out, args)
    say(f"[bench] model={args.model} out={out} corpus={args.corpus} skip_suite={args.skip_suite}")

    try:
        import vram_sampler
        b.sampler = vram_sampler.start(str(out.parent / "vram.csv"))
    except Exception as e:
        say(f"[bench] WARN vram sampler unavailable: {e!r}")

    b.phase("env", phase_env, b)
    b.phase("load", phase_load, b)
    b.phase("sdpa", phase_sdpa, b)
    b.phase("warmup", phase_warmup, b)
    b.phase("transcribe", phase_transcribe, b)
    b.phase("read", phase_read, b)
    b.phase("logprob", phase_logprob, b)
    gate_res = b.phase("gates", phase_gates, b)
    _free_reader(b)
    if args.skip_suite:
        b.data["phases"]["suite"] = {"ok": True, "skip": "skipped (--skip-suite)"}
    else:
        b.phase("suite", phase_suite, b)
    b.data["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    b.save()
    all_pass = bool(gate_res.get("all_pass"))
    say(f"\n[bench] wrote {out}   gates: {'ALL PASS' if all_pass else 'FAIL (see above)'}")
    return 0 if all_pass else 3


if __name__ == "__main__":
    sys.exit(main())
