#!/usr/bin/env python3
"""Resident process (PID 1 of the container): loads the model, then serves ping/status/index/answer on a unix socket.

As PID 1 (or with MC3_SUPERVISE=1) it is only a supervisor: it runs ``server.py --child`` in a loop, reaps every
child, forwards SIGTERM and exits 0 on it.

Never exits on its own: a failed model load leaves ``mode="failed"`` and the server keeps answering through the
deterministic fallback; an unexpected exception inside ``daemon.serve`` restarts the accept loop. Only SIGTERM/SIGINT
stop it. Modes: loading | loaded | cpu | failed | stub.

Env: MC3_READER=auto|stub|none (config.READER), MC3_MODEL, MC3_SOCK, MC3_READY, MC3_INDEX_DIR,
MC3_STUB_PATH (dir holding stub_reader.py), MC3_STUB_GOLD (questions.json for the oracle stub),
MC3_STUB_MODE (oracle|lexical|adversarial|refuse), MC3_STATUS (json mirror of the state, default /tmp/mc3.status),
MC3_SERVER_LOG (default /tmp/mc3-server.log).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import sys
import subprocess
import threading
import time
import traceback
from pathlib import Path

import config
import daemon

HERE = Path(__file__).resolve().parent
STATUS_PATH = Path(os.environ.get("MC3_STATUS", "/tmp/mc3.status"))
LOG_PATH = os.environ.get("MC3_SERVER_LOG", "/tmp/mc3-server.log")
JOIN_PREV_S = 3.0          # wait this long for a timed-out previous worker before starting the next request
GPU_WAIT_S = 90.0          # torch says no GPU: keep asking this long (the ROCm runtime can come up late)
GPU_POLL_S = 5.0
VRAM_CAP_GIB = 44.0        # per-process allocator cap: the evaluator measures a [1, 48] GiB peak
FORCE_MATH_PKG_TOKENS = (4000, 6000)   # math SDPA is O(n^2) in memory: shrink the evidence packages

log = logging.getLogger("mc3.server")


def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger("mc3")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        handlers.append(logging.FileHandler(LOG_PATH))
    except OSError:
        pass
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)


def _json_safe(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


def _import_stub_reader():
    dirs = [HERE]
    if os.environ.get("MC3_STUB_PATH"):
        dirs.append(Path(os.environ["MC3_STUB_PATH"]).resolve())
    dirs.append(HERE.parent / "eval")
    for d in dirs:
        if (d / "stub_reader.py").exists():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            import stub_reader
            return stub_reader
    raise ImportError("stub_reader.py not found (set MC3_STUB_PATH)")


def make_stub_reader():
    mod = _import_stub_reader()
    mode = os.environ.get("MC3_STUB_MODE", "oracle")
    gold_path = os.environ.get("MC3_STUB_GOLD")
    if gold_path:
        return mod.StubReader.from_questions(json.loads(Path(gold_path).read_text(encoding="utf-8")), mode)
    return mod.StubReader(mode)


class State:
    """Server state shared by the boot code and the request handler."""

    def __init__(self):
        self.mode = "loading"
        self.reader = None
        self.index = None
        self.index_dir = Path(config.INDEX_DIR)
        self.corpus: Path | None = None
        self.worker: threading.Thread | None = None
        self.boot_seconds: float | None = None
        self.write_status()

    # -- status ---------------------------------------------------------------------------------

    def status(self) -> dict:
        return {"ok": True, "model": self.mode, "indexed": self.index is not None,
                "records": len(self.index.records) if self.index is not None else 0,
                "boot_s": self.boot_seconds, "pid": os.getpid()}

    def write_status(self) -> None:
        try:
            tmp = STATUS_PATH.with_name(f".{STATUS_PATH.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.status()), encoding="utf-8")
            os.replace(tmp, STATUS_PATH)
        except OSError:
            pass

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        log.info("mode=%s", mode)
        self.write_status()

    # -- boot -----------------------------------------------------------------------------------

    def boot(self) -> None:
        t0 = time.time()
        try:
            kind = config.READER
            if kind == "stub":
                self.reader = make_stub_reader()
                self.set_mode("stub")
            elif kind == "none":
                self.set_mode("cpu")
            else:
                self._boot_auto()
        except Exception as e:  # noqa: BLE001 - never exit: fall back to the deterministic path
            log.error("boot failed: %s\n%s", e, traceback.format_exc())
            self.reader = None
            self.set_mode("failed")
        self.boot_seconds = round(time.time() - t0, 2)
        self.write_status()

    @staticmethod
    def _wait_gpu(torch) -> bool:
        """torch.cuda.is_available() is False: retry every GPU_POLL_S for up to GPU_WAIT_S before CPU mode."""
        log.warning("torch.cuda.is_available() is False; HIP_VISIBLE_DEVICES=%r ROCR_VISIBLE_DEVICES=%r "
                    "CUDA_VISIBLE_DEVICES=%r /dev/kfd exists=%s; retrying for %.0fs",
                    os.environ.get("HIP_VISIBLE_DEVICES"), os.environ.get("ROCR_VISIBLE_DEVICES"),
                    os.environ.get("CUDA_VISIBLE_DEVICES"), os.path.exists("/dev/kfd"), GPU_WAIT_S)
        t0 = time.time()
        while time.time() - t0 < GPU_WAIT_S:
            time.sleep(GPU_POLL_S)
            try:
                if torch.cuda.is_available():
                    log.info("GPU appeared after %.0fs", time.time() - t0)
                    return True
            except Exception as e:  # noqa: BLE001
                log.warning("cuda probe failed: %s: %s", type(e).__name__, e)
        log.warning("no GPU after %.0fs", time.time() - t0)
        return False

    @staticmethod
    def _cap_vram(torch) -> None:
        try:
            total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
            frac = min(1.0, VRAM_CAP_GIB / total_gib)
            torch.cuda.set_per_process_memory_fraction(frac)
            log.info("per-process memory fraction %.3f (%.1f of %.1f GiB)", frac, frac * total_gib, total_gib)
        except Exception as e:  # noqa: BLE001 - the cap is a safety net, never a blocker
            log.warning("could not set the memory fraction: %s: %s", type(e).__name__, e)

    def _boot_auto(self) -> None:
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
        except Exception as e:  # noqa: BLE001 - no torch, broken ROCm runtime
            log.warning("torch unavailable (%s: %s)", type(e).__name__, e)
            cuda, torch = False, None
        if not cuda and torch is not None:
            cuda = self._wait_gpu(torch)
        if not cuda:
            log.warning("#" * 70 + "\n#  NO GPU: running in CPU mode, no model; answers come from the fallback  #\n" + "#" * 70)
            self.set_mode("cpu")
            return
        import reader as reader_mod
        for attempt in (1, 2):
            try:
                t = time.time()
                rd = reader_mod.Reader(config.MODEL_DIR)
                rd.load()
                log.info("model loaded in %.1fs (attempt %d)", time.time() - t, attempt)
                self._cap_vram(torch)
                try:
                    ok = rd.self_test_sdpa()
                    log.info("sdpa self-test ok=%s force_math=%s", ok, getattr(rd, "force_math", None))
                except Exception as e:  # noqa: BLE001
                    log.error("sdpa self-test crashed: %s", e)
                if getattr(rd, "force_math", False) is True:
                    config.PKG_TOKENS = FORCE_MATH_PKG_TOKENS
                    log.warning("force_math is on: PKG_TOKENS=%s", config.PKG_TOKENS)
                t = time.time()
                rd.warmup()
                log.info("warmup %.1fs", time.time() - t)
                self.reader = rd
                self.set_mode("loaded")
                return
            except Exception as e:  # noqa: BLE001
                log.error("model load attempt %d failed: %s\n%s", attempt, e, traceback.format_exc())
        log.error("!" * 70 + "\n!  MODEL LOAD FAILED: serving fallback answers only (mode=failed)  !\n" + "!" * 70)
        self.reader = None
        self.set_mode("failed")

    # -- requests -------------------------------------------------------------------------------

    def _wait_previous(self) -> None:
        w = self.worker
        if w is not None and w.is_alive():
            w.join(JOIN_PREV_S)

    def op_index(self, req: dict) -> dict:
        import pipeline
        from budget import Deadline
        t0 = time.time()
        self._wait_previous()
        corpus = Path(req["corpus"])
        index_dir = Path(req.get("index_dir") or config.INDEX_DIR)
        deadline = Deadline(float(req.get("deadline_ts") or time.time() + config.INDEX_BUDGET_S))
        self.index = pipeline.index_corpus(corpus, index_dir, self.reader, deadline,
                                           rebuild=bool(req.get("build")))
        self.corpus, self.index_dir = corpus, index_dir
        self.write_status()
        try:
            st = json.loads((index_dir / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
        return {"ok": True, "records": len(self.index.records), "vlm_done": st.get("vlm_done", 0),
                "vlm_pending": st.get("vlm_pending", 0), "seconds": round(time.time() - t0, 2)}

    def _ensure_index(self, req: dict) -> None:
        if self.index is not None:
            return
        import index as index_mod
        index_dir = Path(req.get("index_dir") or self.index_dir)
        self.index = index_mod.load(index_dir)
        self.index_dir = index_dir
        self.write_status()

    def _fallback_answer(self, query: str, why: str) -> dict:
        import fallback
        ans, cites = fallback.lookup(self.index_dir, query, deadline_ts=time.time() + config.FALLBACK_CAP_S)
        conf = 0.5 if ans and cites else 0.0
        return {"answer": ans, "citations": cites if ans else [], "confidence": conf, "trace": {"reason": why}}

    def op_answer(self, req: dict) -> dict:
        import pipeline
        from budget import Deadline, Trace
        query = str(req["query"])
        cap = time.time() + config.ANSWER_SOFT_S
        deadline = Deadline(min(float(req["deadline_ts"]), cap) if req.get("deadline_ts") else cap)
        corpus = Path(req["corpus"]) if req.get("corpus") else self.corpus
        if corpus is not None:
            self.corpus = corpus
        if "index_dir" in req:
            self.index_dir = Path(req["index_dir"])
        if self.reader is None:      # cpu / failed / none: deterministic path only
            return self._fallback_answer(query, f"MODE_{self.mode.upper()}")
        self._ensure_index(req)
        self._wait_previous()
        box: dict = {}
        trace = Trace()

        def work():
            try:
                box["res"] = pipeline.answer(self.index, self.reader, query, deadline, corpus, trace)
            except BaseException as e:  # noqa: BLE001
                box["err"] = e

        t = threading.Thread(target=work, name="answer", daemon=True)
        self.worker = t
        t.start()
        t.join(max(0.0, deadline.remaining()) + 0.5)
        if t.is_alive():
            log.warning("answer timed out for %r; worker left to finish", query[:80])
            return {"answer": "", "citations": [], "confidence": 0.0,
                    "trace": _json_safe({**trace.to_dict(), "reason": "TIMEOUT"})}
        if "err" in box:
            raise box["err"]
        res = box["res"]
        out = res.to_output()
        out["trace"] = _json_safe(res.trace)
        return out

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        t0 = time.time()
        try:
            if op == "status":
                return self.status()
            if op == "index":
                return self.op_index(req)
            if op == "answer":
                resp = self.op_answer(req)
                log.info("answer %s %.2fs reason=%s ans=%r cites=%s", req.get("query_id", "?"), time.time() - t0,
                         resp.get("trace", {}).get("reason"), resp.get("answer"), resp.get("citations"))
                return resp
            return {"error": f"unknown op {op!r}"}
        except Exception as e:  # noqa: BLE001 - one bad request must not kill the server
            log.error("op %s failed: %s\n%s", op, e, traceback.format_exc())
            if op == "answer":
                return {"answer": "", "citations": [], "confidence": 0.0,
                        "trace": {"error": f"{type(e).__name__}: {e}"}}
            return {"error": f"{type(e).__name__}: {e}"}


def _cleanup() -> None:
    for p in (Path(daemon.SOCK), daemon.READY):
        try:
            p.unlink()
        except OSError:
            pass


def _on_signal(signum, _frame) -> None:
    log.info("signal %s: exiting", signum)
    raise SystemExit(0)


def main() -> int:
    setup_logging()
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    atexit.register(_cleanup)
    log.info("server start pid=%d reader=%s sock=%s index_dir=%s", os.getpid(), config.READER, daemon.SOCK,
             config.INDEX_DIR)
    state = State()
    state.boot()
    while True:
        try:
            log.info("serving (mode=%s)", state.mode)
            daemon.serve(state.handle)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as e:  # noqa: BLE001 - restart the accept loop, never exit
            log.error("serve crashed: %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
            time.sleep(1.0)


def supervise() -> int:
    """PID 1: run ``server.py --child`` forever, reap every child (orphans included), exit 0 on SIGTERM/SIGINT.

    The server never exits on purpose; if the child dies anyway (OOM kill, segfault in the ROCm runtime) it is
    restarted after 2 s."""
    state = {"stop": False, "child": None}

    def forward(signum, _frame):
        state["stop"] = True
        child = state["child"]
        if child is not None:
            try:
                os.kill(child, signal.SIGTERM)
            except OSError:
                pass
            signal.alarm(10)    # a child that ignores SIGTERM is killed

    def kill_child(_signum, _frame):
        child = state["child"]
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except OSError:
                pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGALRM, kill_child)
    while not state["stop"]:
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child"])
        state["child"] = proc.pid
        if state["stop"]:   # the signal landed between the stop check and the assignment
            forward(signal.SIGTERM, None)
        while True:
            try:
                pid, status = os.waitpid(-1, 0)
            except ChildProcessError:
                break
            except InterruptedError:
                continue
            if pid == proc.pid:
                state["child"] = None
                print(f"[mc3-supervisor] server exited with status {status}", file=sys.stderr, flush=True)
                break
        state["child"] = None
        t_end = time.time() + 2.0
        while not state["stop"] and time.time() < t_end:
            time.sleep(0.1)
    signal.alarm(0)
    return 0


if __name__ == "__main__":
    try:
        if "--child" not in sys.argv[1:] and (os.getpid() == 1 or os.environ.get("MC3_SUPERVISE") == "1"):
            sys.exit(supervise())
        sys.exit(main())
    except (SystemExit, KeyboardInterrupt):
        sys.exit(0)
