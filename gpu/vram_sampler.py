"""Background GPU-memory sampler (stdlib only; torch is used only if the caller already imported it).

    from vram_sampler import start
    s = start("runs/x/vram.csv")      # thread samples every 2 s
    ...
    info = s.stop()                    # {"peak_mib": float, "samples": n, "source": str, ...}

Sources, all optional and independent (a missing tool is just skipped):
  torch    torch.cuda.memory_allocated / max_memory_allocated (CUDA and ROCm builds)
  nvidia   nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  amd      amd-smi metric --mem --json  (fallback: rocm-smi --showmeminfo vram --json)
`peak_mib` is the largest reading over every source, so on a driver-level source it is the number the
harness' sampler would see (allocator reserve + context), not just tensors. Nothing here may raise.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import threading
import time

INTERVAL_S = 2.0
CMD_TIMEOUT_S = 5.0
MAX_FAILS = 3  # consecutive failures before a command source is dropped
MIB = float(1 << 20)

COLUMNS = ["t_s", "torch_alloc_mib", "torch_max_alloc_mib", "nvidia_used_mib", "amd_used_mib", "used_mib"]


# ----------------------------------------------------------------------------- parsers (pure, unit-tested)

def parse_nvidia_smi(text: str) -> float | None:
    """`memory.used` csv,noheader,nounits: one integer MiB per GPU line -> max over GPUs."""
    vals = []
    for line in (text or "").splitlines():
        line = line.strip().split(",")[0].strip()
        try:
            vals.append(float(line))
        except ValueError:
            continue
    return max(vals) if vals else None


def _to_mib(value, unit: str) -> float | None:
    try:
        v = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    u = (unit or "").strip().lower()
    if u in ("b", "byte", "bytes"):
        return v / MIB
    if u in ("kb", "kib"):
        return v / 1024.0
    if u in ("gb", "gib"):
        return v * 1024.0
    return v  # MB / MiB / unknown -> amd-smi reports MiB-ish "MB"


def _is_used_vram_key(key: str, parent: str) -> bool:
    k, p = key.lower(), parent.lower()
    if "used" not in k:
        return False
    if "vis" in k or "gtt" in k:  # visible-vram / gtt counters are not VRAM use
        return False
    return "vram" in k or "vram" in p or p in ("mem_usage", "memory", "mem")


def _walk_used(node, parent: str, out: list) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            ks = str(k)
            if _is_used_vram_key(ks, parent):
                unit = ""
                if "(b)" in ks.lower():
                    unit = "B"
                if isinstance(v, dict):
                    mib = _to_mib(v.get("value"), v.get("unit", unit or "MB"))
                elif isinstance(v, str) and v.strip() and not v.strip().replace(".", "", 1).isdigit():
                    # "123 MB" style
                    parts = v.split()
                    mib = _to_mib(parts[0], parts[1] if len(parts) > 1 else "MB")
                else:
                    mib = _to_mib(v, unit or "MB")
                if mib is not None:
                    out.append(mib)
            else:
                _walk_used(v, ks, out)
    elif isinstance(node, list):
        for it in node:
            _walk_used(it, parent, out)


def parse_amd_json(text: str) -> float | None:
    """amd-smi `metric --mem --json` (list of per-GPU dicts, used_vram {value,unit}) or rocm-smi
    `--showmeminfo vram --json` ({"card0": {"VRAM Total Used Memory (B)": "..."}}) -> max used MiB."""
    try:
        data = json.loads(text)
    except Exception:
        return None
    vals: list[float] = []
    _walk_used(data, "", vals)
    return max(vals) if vals else None


# ----------------------------------------------------------------------------- readers

def _run(argv: list[str]) -> str | None:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=CMD_TIMEOUT_S, check=False)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def read_nvidia() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return parse_nvidia_smi(out) if out else None


def read_amd() -> float | None:
    if shutil.which("amd-smi"):
        out = _run(["amd-smi", "metric", "--mem", "--json"])
        v = parse_amd_json(out) if out else None
        if v is not None:
            return v
    if shutil.which("rocm-smi"):
        out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
        return parse_amd_json(out) if out else None
    return None


def read_torch() -> tuple[float, float] | None:
    """(allocated, max_allocated) MiB; only if torch is already imported and a device is up."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_allocated() / MIB, torch.cuda.max_memory_allocated() / MIB
    except Exception:
        return None


# ----------------------------------------------------------------------------- sampler

class Sampler:
    def __init__(self, path: str | None, interval: float = INTERVAL_S):
        self.path = path
        self.interval = interval
        self.t0 = time.time()
        self.n = 0
        self.peak = 0.0
        self.peak_by: dict[str, float] = {}
        self.last: dict = {}
        self._fails = {"nvidia": 0, "amd": 0}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._fh = None
        self._csv = None
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    # -- one reading
    def sample_now(self) -> dict:
        """Take one sample (also callable from the main thread, e.g. 'VRAM after load'). Never raises."""
        row = {"t_s": round(time.time() - self.t0, 2)}
        try:
            tt = read_torch()
            if tt:
                row["torch_alloc_mib"], row["torch_max_alloc_mib"] = round(tt[0], 1), round(tt[1], 1)
            for name, fn in (("nvidia", read_nvidia), ("amd", read_amd)):
                if self._fails[name] >= MAX_FAILS:
                    continue
                v = fn()
                if v is None:
                    self._fails[name] += 1
                else:
                    self._fails[name] = 0
                    row[f"{name}_used_mib"] = round(v, 1)
            vals = {k: v for k, v in row.items() if k != "t_s"}
            used = max(vals.values()) if vals else None
            if used is not None:
                row["used_mib"] = used
            with self._lock:
                self.n += 1
                for k, v in vals.items():
                    key = "torch" if k.startswith("torch") else k.split("_")[0]
                    if v > self.peak_by.get(key, 0.0):
                        self.peak_by[key] = v
                if used is not None and used > self.peak:
                    self.peak = used
                self.last = row
                if self._csv is not None:
                    self._csv.writerow([row.get(c, "") for c in COLUMNS])
                    self._fh.flush()
        except Exception as e:  # pragma: no cover - defensive
            self.error = repr(e)
        return row

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_now()
            self._stop.wait(self.interval)

    def current_mib(self) -> float | None:
        return self.last.get("used_mib")

    def start(self) -> "Sampler":
        try:
            if self.path:
                import os
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                self._fh = open(self.path, "w", newline="")
                self._csv = csv.writer(self._fh)
                self._csv.writerow(COLUMNS)
            self._thread = threading.Thread(target=self._loop, name="vram-sampler", daemon=True)
            self._thread.start()
        except Exception as e:
            self.error = repr(e)
        return self

    def stop(self) -> dict:
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=CMD_TIMEOUT_S * 3)
            self.sample_now()  # final reading so a short run still has data
        except Exception as e:  # pragma: no cover
            self.error = repr(e)
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
        with self._lock:
            src = "+".join(sorted(self.peak_by)) or "none"
            info = {"peak_mib": round(self.peak, 1), "samples": self.n, "source": src,
                    "peak_by_source_mib": {k: round(v, 1) for k, v in self.peak_by.items()}}
        if self.error:
            info["error"] = self.error
        return info


def start(path: str | None = None, interval: float = INTERVAL_S) -> Sampler:
    try:
        return Sampler(path, interval).start()
    except Exception:  # pragma: no cover
        return Sampler(None, interval)


if __name__ == "__main__":  # smoke: python3 gpu/vram_sampler.py [seconds]
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    s = start("/tmp/vram_smoke.csv", interval=1.0)
    time.sleep(secs)
    print(json.dumps(s.stop()))
