#!/usr/bin/env python3
"""MC-3 entry point. The harness runs this script two ways (same argparse as the starter kit):

    python3 /app/app.py --index /app/corpus                      (once, before any graded clock)
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."   (once per question)

``--index`` parses and OCRs in this process (CPU), persists the index under ``MC3_INDEX_DIR`` and asks the
resident daemon (``server.py``) to finish the job (VLM transcription). ``--query`` is a thin stdlib client: one
JSON line to the daemon, deterministic fallback, then refusal. Both modes always exit 0 and ``--query`` always
leaves a valid ``<query-id>_output.json`` behind (a crashed writer must not look like a considered refusal,
but a missing file is worse than either).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import config
import daemon

CORPUS_PREFIXES = ("/app/corpus/", "./")
FALLBACK_CONFIDENCE = 0.5
REFUSAL = {"answer": "", "citations": [], "confidence": 0.0}


def _log(msg: str) -> None:
    try:
        print(f"[mc3] {msg}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a broken stderr must never stop the output file
        pass


# ---------------------------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------------------------

def _write_atomic_text(path: Path, text: str) -> None:
    """Whole file or nothing: temp file in the same directory, then ``os.replace``."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def write_atomic(path: Path, obj) -> None:
    # ensure_ascii: a lone surrogate from an odd file name must not turn into an encode error
    _write_atomic_text(path, json.dumps(obj, ensure_ascii=True))


def clean_citations(raw) -> list[str]:
    """Corpus-relative POSIX paths, deduplicated and sorted; absolute /app/corpus/ and ./ prefixes stripped."""
    out = set()
    for c in raw or []:
        c = str(c).strip()
        stripped = True
        while stripped:
            stripped = False
            for p in CORPUS_PREFIXES:
                if c.startswith(p):
                    c, stripped = c[len(p):], True
        if c:
            out.add(c)
    return sorted(out)


def _valid(resp) -> bool:
    return (isinstance(resp, dict) and "error" not in resp and isinstance(resp.get("answer"), str)
            and isinstance(resp.get("citations"), list) and all(isinstance(c, str) for c in resp["citations"]))


def _confidence(resp: dict) -> float:
    try:
        return float(resp.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------------------------
# --query
# ---------------------------------------------------------------------------------------------

def answer_query(corpus: Path, query_id: str, query: str, start: float) -> tuple[dict, str]:
    """(output dict, via) with via in daemon|fallback|refusal."""
    deadline_ts = start + config.ANSWER_SOFT_S
    resp = None
    try:
        resp = daemon.request(
            {"op": "answer", "query": query, "query_id": query_id, "corpus": str(corpus), "deadline_ts": deadline_ts},
            timeout=config.CLIENT_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 - any daemon trouble means fallback
        _log(f"{query_id}: daemon request failed: {type(e).__name__}: {e}")
    if _valid(resp):
        cites = clean_citations(resp["citations"])
        if bool(resp["answer"].strip()) != bool(cites):
            # an answer nobody can check, or citations without an answer, is never a correct response
            _log(f"{query_id}: daemon answer/citations incoherent ({resp['answer'][:60]!r}, {cites}); refusing")
            return dict(REFUSAL, citations=[]), "refusal"
        return ({"answer": resp["answer"], "citations": cites, "confidence": _confidence(resp)}, "daemon")
    if resp is not None:
        _log(f"{query_id}: daemon answer unusable: {str(resp)[:200]}")
    if time.time() - start > config.FALLBACK_LATEST_S:
        return dict(REFUSAL, citations=[]), "refusal"   # no time left for a safe fallback
    try:
        import fallback   # lazy: a broken fallback must not stop the refusal being written
        ans, cites = fallback.lookup(Path(config.INDEX_DIR), query,
                                     deadline_ts=time.time() + config.FALLBACK_CAP_S)
    except Exception as e:  # noqa: BLE001
        _log(f"{query_id}: fallback failed: {type(e).__name__}: {e}")
        ans, cites = "", []
    if ans and cites:
        return {"answer": ans, "citations": clean_citations(cites), "confidence": FALLBACK_CONFIDENCE}, "fallback"
    return dict(REFUSAL, citations=[]), "refusal"


def run_query(args: argparse.Namespace) -> int:
    start = time.time()
    result, via = dict(REFUSAL, citations=[]), "refusal"
    out = Path(config.OUTPUT_DIR) / (args.query_id + "_output.json")
    try:    # a valid file exists from the first second: SIGTERM/SIGKILL mid-query leaves a refusal, not nothing
        out.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(out, dict(REFUSAL, citations=[]))
    except BaseException as e:  # noqa: BLE001
        _log(f"{args.query_id}: early refusal write failed: {type(e).__name__}: {e}")
    try:
        if args.corpus is None or args.query is None:
            _log(f"{args.query_id}: --corpus and --query are required; writing a refusal")
        else:
            result, via = answer_query(args.corpus, args.query_id, args.query, start)
    except BaseException as e:  # noqa: BLE001 - incl. KeyboardInterrupt/SystemExit from inside: still write
        _log(f"{args.query_id}: crashed: {type(e).__name__}: {e}")
        result, via = dict(REFUSAL, citations=[]), "refusal"
    for attempt in (result, dict(REFUSAL, citations=[])):
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(out, attempt)
            break
        except BaseException as e:  # noqa: BLE001
            _log(f"{args.query_id}: write failed: {type(e).__name__}: {e}")
            via = "refusal"
    _log(f"{args.query_id} {time.time() - start:.2f}s via={via}")
    return 0


# ---------------------------------------------------------------------------------------------
# --index
# ---------------------------------------------------------------------------------------------

def _server_running() -> bool:
    """Any other process whose argv mentions server.py (it binds the socket only after the model loads)."""
    me = os.getpid()
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit() and int(p) != me]
    except OSError:
        return False
    for p in pids:
        try:
            with open(f"/proc/{p}/cmdline", "rb") as fh:
                argv = fh.read().split(b"\0")
        except OSError:
            continue
        if any(a.endswith(b"server.py") for a in argv[1:]):
            return True
    return False


def _model_loading() -> bool:
    """The server's status mirror says the model is still loading."""
    try:
        return json.loads(Path(config.STATUS_PATH).read_text(encoding="utf-8")).get("model") == "loading"
    except (OSError, ValueError, AttributeError):
        return False


def _wait_daemon(deadline: float, hard_deadline: float | None = None) -> bool:
    """Poll for READY; after 30 s without READY and without a server process, start one (safety net).

    Past ``deadline`` keep waiting up to ``hard_deadline`` while a server process is alive and still loading."""
    first = min(deadline, time.time() + 30.0)
    if daemon.wait_ready(max(0.0, first - time.time())):
        return True
    if time.time() < deadline and os.environ.get("MC3_AUTOSPAWN", "1") != "0" and not _server_running():
        server = os.environ.get("MC3_SERVER", "/app/server.py")
        if os.path.exists(server):
            _log(f"no server after 30s; spawning {server}")
            daemon.spawn_detached([server])
        else:
            _log(f"no server after 30s and {server} does not exist")
    if daemon.wait_ready(max(0.0, deadline - time.time())):
        return True
    if hard_deadline is None:
        return False
    waited = False
    while time.time() < hard_deadline and _model_loading() and _server_running():
        waited = True
        if daemon.wait_ready(min(5.0, max(0.0, hard_deadline - time.time()))):
            return True
    if waited:   # the model just finished (the status flips before the socket binds)
        return daemon.wait_ready(max(0.0, min(30.0, hard_deadline - time.time())))
    return False


def run_index(corpus: Path) -> int:
    t0 = time.time()
    index_dir = Path(config.INDEX_DIR)
    build_failed = False
    try:
        import index   # lazy: the --query path must stay stdlib-light
        import parse
        import vision

        parsed = parse.walk(corpus)
        t_parse = time.time() - t0
        t1 = time.time()
        try:
            ocr = vision.ocr_all(parsed, corpus, index_dir / "images", deadline_ts=t0 + config.OCR_BUDGET_S)
        except Exception as e:  # noqa: BLE001 - OCR is an enhancement, never a blocker
            _log(f"OCR failed: {type(e).__name__}: {e}")
            ocr = {}
        t_ocr = time.time() - t1
        idx = index.build(parsed, ocr)
        index.save(index_dir, idx)
        status = {"files": len(idx.files), "records": len(idx.records),
                  "images": sum(int(v.get("images", 0)) for v in ocr.values()),
                  "t_parse": round(t_parse, 3), "t_ocr": round(t_ocr, 3)}
        _write_atomic_text(index_dir / "status.json", json.dumps(status))
        _write_atomic_text(index_dir / "INDEXED", f"{time.time():.0f}\n")
        print(f"[mc3] indexed files={status['files']} records={status['records']} images={status['images']} "
              f"parse={t_parse:.1f}s ocr={t_ocr:.1f}s", flush=True)
    except BaseException as e:  # noqa: BLE001
        import traceback
        _log(f"index build failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        build_failed = True     # the daemon builds from the corpus itself (a partial index dir is ignored)

    try:
        deadline = t0 + config.INDEX_BUDGET_S
        hard = deadline + 180.0     # model still loading: wait up to INDEX_BUDGET_S + 180 s from the exec
        if _wait_daemon(deadline - 60, hard):
            # a late READY gets the remaining time (the daemon's VLM stage stops at deadline_ts - margin)
            idx_deadline = deadline if time.time() <= deadline - 60 else hard
            payload = {"op": "index", "corpus": str(corpus), "index_dir": str(index_dir),
                       "deadline_ts": idx_deadline}
            if build_failed:
                payload["build"] = True
            resp = daemon.request(payload, timeout=max(5.0, idx_deadline - time.time()))
            _log(f"daemon index: {resp}")
        else:
            _log("daemon not ready; queries will use the fallback unless it comes up")
    except BaseException as e:  # noqa: BLE001
        _log(f"daemon index step failed: {type(e).__name__}: {e}")
    return 0


# ---------------------------------------------------------------------------------------------

def _scan_flag(argv: list[str], flag: str) -> str | None:
    """Value of ``--flag V`` / ``--flag=V`` read by hand: V may start with "-" (argparse rejects that)."""
    for k, a in enumerate(argv):
        if a == flag and k + 1 < len(argv):
            return argv[k + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return None


def _recover_args(argv: list[str]) -> argparse.Namespace | None:
    """Namespace for a query whose command line argparse refused; None if there is no usable --query-id."""
    qid = _scan_flag(argv, "--query-id")
    if not qid:
        return None
    corpus = _scan_flag(argv, "--corpus")
    return argparse.Namespace(index=None, corpus=Path(corpus) if corpus else Path("/app/corpus"),
                              query_id=qid, query=_scan_flag(argv, "--query"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        args = ap.parse_args(raw)
    except SystemExit as e:     # bad command line (e.g. a question starting with "-"): never exit non-zero
        if "--index" in raw:
            return 0
        rec = _recover_args(raw)
        if rec is None:
            _log(f"unusable command line (argparse exit {e.code}); nothing to write")
            return 0
        _log(f"{rec.query_id}: argparse refused the command line; recovered it by hand")
        return run_query(rec)

    if args.index is not None:
        return run_index(args.index)

    if not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")
    return run_query(args)


if __name__ == "__main__":
    raise SystemExit(main())
