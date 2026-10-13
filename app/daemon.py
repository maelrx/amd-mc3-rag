"""Resident model process behind a unix socket.

`--index` spawns this detached (setsid + double fork) so it outlives the exec
that started it; every `--query` process is then a thin client that sends one
JSON line and reads one JSON line back. If the socket is missing or dead, the
client falls back to answering in-process (slower, but never a crash).

Protocol: request  {"op": "answer", "query": str}   -> {"answer","citations","confidence"}
          request  {"op": "ping"}                    -> {"ok": true}
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

SOCK = os.environ.get("MC3_SOCK", "/tmp/mc3.sock")
READY = Path(os.environ.get("MC3_READY", "/tmp/mc3.ready"))


def spawn_detached(argv: list[str], log: str = "/tmp/mc3-daemon.log") -> None:
    """Start argv in its own session so it survives the parent process exiting (no raw fork:
    the caller may already run onnxruntime threads)."""
    import subprocess
    with open(log, "ab") as fh, open(os.devnull, "rb") as devnull:
        subprocess.Popen([sys.executable, *argv], stdin=devnull, stdout=fh, stderr=fh,
                         start_new_session=True, close_fds=True)


def wait_ready(timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if READY.exists() and ping():
            return True
        time.sleep(0.5)
    return False


def request(payload: dict, timeout: float = 28.0) -> dict | None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(SOCK)
            s.sendall((json.dumps(payload) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        return json.loads(buf) if buf else None
    except (OSError, ValueError):
        return None


def ping() -> bool:
    r = request({"op": "ping"}, timeout=2.0)
    return bool(r and r.get("ok"))


def serve(handler) -> None:
    """Accept one request at a time (the harness asks sequentially)."""
    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(8)
    READY.write_text(str(os.getpid()))
    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                conn.settimeout(5.0)  # a client that connects and sends nothing must not wedge the daemon
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                req = json.loads(buf)
                if req.get("op") == "ping":
                    resp = {"ok": True}
                else:
                    resp = handler(req)
            except Exception as e:  # noqa: BLE001 - a bad request must not kill the daemon
                resp = {"error": f"{type(e).__name__}: {e}"}
            try:
                conn.sendall((json.dumps(resp, ensure_ascii=True, default=str) + "\n").encode())
            except (OSError, ValueError, TypeError):
                pass  # client timed out and went away; the next request must still be served
