#!/usr/bin/env python3
"""Simulate the harness with subprocesses: start server.py, run `app.py --index`, then one `app.py --query` per question.

    python3 eval/harness_sim.py --corpus sample-corpus --questions sample-questions.json --out /tmp/sim \\
        [--reader oracle|lexical|adversarial|refuse] [--kill-daemon] [--as-user] [--keep]

The server runs with MC3_READER=stub (gold taken from the questions file), so this exercises socket, client, index
persistence, pipeline and scoring on CPU. Index goes to OUT/index, outputs to OUT/output; the socket lives in a short
/tmp dir (unix socket paths are limited to ~100 bytes).

--kill-daemon  SIGKILLs the server after indexing: every query must then go through the client fallback / refusal.
--as-user      only prints how to repeat the run as a non-owner user (needed to see the chmod 000 trap with root-like
               ownership; this script cannot sudo).
Exit 1 if any query took more than 30 s or any output file is not valid JSON with answer/citations/confidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import score as score_mod  # noqa: E402

QUERY_TIMEOUT_S = 30.0
INDEX_TIMEOUT_S = 600.0
READY_TIMEOUT_S = 60.0
VIA_RE = re.compile(r"via=(\w+)")


def ping(sock_path: str) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(sock_path)
            s.sendall(b'{"op":"ping"}\n')
            return bool(json.loads(s.recv(65536)).get("ok"))
    except (OSError, ValueError):
        return False


def valid_output(path: Path) -> tuple[bool, str]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable: {type(e).__name__}"
    if not isinstance(d, dict) or not {"answer", "citations", "confidence"} <= set(d):
        return False, "missing keys"
    if not isinstance(d["answer"], str) or not isinstance(d["citations"], list) \
            or not all(isinstance(c, str) for c in d["citations"]) or not isinstance(d["confidence"], (int, float)):
        return False, "bad types"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reader", default="oracle", choices=["oracle", "lexical", "adversarial", "refuse", "real"],
                    help="real = MC3_READER=auto with the real model (GPU); --index starts right away, before READY")
    ap.add_argument("--kill-daemon", action="store_true")
    ap.add_argument("--as-user", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the temp socket dir")
    args = ap.parse_args(argv)

    corpus, questions, out = args.corpus.resolve(), args.questions.resolve(), args.out.resolve()
    data = json.loads(questions.read_text(encoding="utf-8"))
    index_dir, out_dir = out / "index", out / "output"
    for d in (index_dir, out_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    sockdir = Path(tempfile.mkdtemp(prefix="mc3sim-", dir="/tmp"))
    sock, ready = str(sockdir / "s.sock"), str(sockdir / "s.ready")
    server_log = out / "server.log"

    real = args.reader == "real"
    env = dict(os.environ)
    if real:
        env.update({"MC3_READER": "auto", "MC3_SUPERVISE": "1"})
    else:
        env.update({"MC3_READER": "stub", "MC3_STUB_PATH": str(ROOT / "eval"), "MC3_STUB_GOLD": str(questions),
                    "MC3_STUB_MODE": args.reader})
    env.update({"MC3_SOCK": sock, "MC3_READY": ready,
                "MC3_INDEX_DIR": str(index_dir), "MC2_OUTPUT_DIR": str(out_dir),
                "MC3_STATUS": str(sockdir / "status.json"), "MC3_SERVER_LOG": str(out / "server-internal.log"),
                "MC3_AUTOSPAWN": "0", "PYTHONUNBUFFERED": "1"})

    if args.as_user:
        print("# repeat as a non-owner user so the chmod 000 trap is really unreadable (cannot sudo here):\n"
              f"#   sudo -u nobody -E env PATH=$PATH python3 {Path(__file__).name} --corpus {args.corpus} "
              f"--questions {args.questions} --out /tmp/sim_nobody --reader {args.reader}\n"
              "# (the corpus and the repo must be world-readable for that user)")

    srv_fh = open(server_log, "w")
    server = subprocess.Popen([sys.executable, str(APP / "server.py")], env=env, stdout=srv_fh, stderr=srv_fh)
    failures: list[str] = []
    rows: list[dict] = []
    try:
        t0 = time.time()
        while not real and time.time() - t0 < READY_TIMEOUT_S:
            if server.poll() is not None:
                failures.append(f"server exited early (rc={server.returncode})")
                break
            if Path(ready).exists() and ping(sock):
                break
            time.sleep(0.2)
        else:
            if not real:
                failures.append(f"server not READY in {READY_TIMEOUT_S:.0f}s")
        if real:
            time.sleep(1.0)   # like the harness: exec --index right after the container starts, model still loading
            if server.poll() is not None:
                failures.append(f"server exited early (rc={server.returncode})")
        print(f"server ready in {time.time() - t0:.1f}s" if not failures else f"FAILED: {failures}")

        if real and not failures:
            print(f"--index launched {time.time() - t0:.1f}s after server start (READY={Path(ready).exists()})")
        if not failures:
            ti = time.time()
            r = subprocess.run([sys.executable, str(APP / "app.py"), "--index", str(corpus)], env=env,
                               capture_output=True, text=True, timeout=INDEX_TIMEOUT_S)
            print(f"--index rc={r.returncode} in {time.time() - ti:.1f}s")
            for line in (r.stdout + r.stderr).splitlines():
                if "indexed" in line or "daemon index" in line:
                    print("   " + line)
            if r.returncode != 0:
                failures.append(f"--index rc={r.returncode}")

        if args.kill_daemon and server.poll() is None:
            server.kill()
            server.wait()
            print("server killed (SIGKILL): queries go through the client fallback")

        for q in data["queries"]:
            qid = f"query_{q['n']:02d}"
            t = time.time()
            try:
                r = subprocess.run([sys.executable, str(APP / "app.py"), "--corpus", str(corpus), "--query-id", qid,
                                    "--query", q["query"]], env=env, capture_output=True, text=True,
                                   timeout=QUERY_TIMEOUT_S)
                rc, err = r.returncode, r.stderr
            except subprocess.TimeoutExpired:
                rc, err = -9, "TIMEOUT"
            secs = time.time() - t
            m = VIA_RE.search(err)
            ok, why = valid_output(out_dir / f"{qid}_output.json")
            if secs > QUERY_TIMEOUT_S or rc != 0:
                failures.append(f"{qid}: {secs:.1f}s rc={rc}")
            if not ok:
                failures.append(f"{qid}: invalid output ({why})")
            rows.append({"id": qid, "secs": secs, "via": m.group(1) if m else "?", "valid": ok})
        if real:
            try:
                st = json.loads((sockdir / "status.json").read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                st = {"error": f"{type(e).__name__}: {e}"}
            print(f"server status after queries: {json.dumps(st)[:400]}  alive={server.poll() is None}")
            if server.poll() is not None or st.get("model") != "loaded":
                failures.append(f"server not alive/loaded after queries: {st.get('model')}")
            bad = [r["id"] for r in rows if r["via"] != "daemon"]
            if bad:
                failures.append(f"answered outside the daemon: {bad}")
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        srv_fh.close()
        if not args.keep:
            shutil.rmtree(sockdir, ignore_errors=True)

    sc = score_mod.score(data, out_dir)
    by_id = {r["id"]: r for r in sc["rows"]}
    print(f"\n{'id':<9} {'secs':>5} {'via':<8} {'pts':>3}  {'got':<70} expected")
    for r in rows:
        s = by_id[r["id"]]
        got = s["got"] or {}
        g = f"{got.get('answer', '?')!r} {got.get('citations', '?')}"
        e = f"{s['expected']['answer']!r} {s['expected']['citations']}"
        print(f"{r['id']:<9} {r['secs']:5.2f} {r['via']:<8} {s['pts']:>3}  {g[:70]:<70} {e}")
    print(f"TOTAL {sc['total']}/{sc['max']}   max query time {max((r['secs'] for r in rows), default=0):.2f}s   "
          f"reader={args.reader}{' kill-daemon' if args.kill_daemon else ''}")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        print(f"server log: {server_log}")
        return 1
    print("harness_sim OK (all outputs valid, all queries <= 30 s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
