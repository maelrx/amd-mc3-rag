"""server.py as a subprocess (stub / no-reader modes) driven over its unix socket."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

import daemon
from conftest import APP, ROOT, SAMPLE_CORPUS, SAMPLE_QUESTIONS

SERVER_PY = APP / "server.py"
Q4 = "Which firmware version fixed ticket ORR-1847?"
Q9 = "The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?"
Q1 = "What is the maximum junction temperature of the TQ-40?"


class Srv:
    def __init__(self, tmp: Path, reader: str, preload_index: Path | None = None):
        self.tmp = tmp
        self.sock = str(tmp / "s.sock")
        self.index_dir = tmp / "index"
        env = dict(os.environ)
        env.update({"MC3_READER": reader, "MC3_STUB_PATH": str(ROOT / "eval"), "MC3_STUB_GOLD": str(SAMPLE_QUESTIONS),
                    "MC3_SOCK": self.sock, "MC3_READY": self.sock + ".ready", "MC3_INDEX_DIR": str(self.index_dir),
                    "MC2_OUTPUT_DIR": str(tmp / "out"), "MC3_STATUS": str(tmp / "status.json"),
                    "MC3_SERVER_LOG": str(tmp / "server.log"), "PYTHONUNBUFFERED": "1"})
        if preload_index is not None:
            shutil.copytree(preload_index, self.index_dir)
        self.log = open(tmp / "stderr.log", "w")
        self.proc = subprocess.Popen([sys.executable, str(SERVER_PY)], env=env, stdout=self.log, stderr=self.log)
        t0 = time.time()
        while time.time() - t0 < 30:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server died: {(tmp / 'stderr.log').read_text()}")
            if Path(self.sock + ".ready").exists() and self.ask({"op": "ping"}, 2).get("ok"):
                return
            time.sleep(0.1)
        raise RuntimeError("server not ready")

    def ask(self, payload, timeout=30.0) -> dict:
        return self.raw((json.dumps(payload) + "\n").encode(), timeout)

    def raw(self, data: bytes, timeout=30.0) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(self.sock)
            s.sendall(data)
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        return json.loads(buf)

    def index(self) -> dict:
        return self.ask({"op": "index", "corpus": str(SAMPLE_CORPUS), "index_dir": str(self.index_dir),
                         "deadline_ts": time.time() + 120}, 120)

    def answer(self, q: str, qid="query_01") -> dict:
        return self.ask({"op": "answer", "query": q, "query_id": qid, "corpus": str(SAMPLE_CORPUS),
                         "deadline_ts": time.time() + 22}, 30)

    def stop(self) -> int:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()
        return self.proc.returncode


@pytest.fixture(scope="module")
def stub_server(tmp_path_factory):
    srv = Srv(tmp_path_factory.mktemp("srv_stub"), "stub")
    yield srv
    srv.stop()


def test_ping_status_and_daemon_client(stub_server, monkeypatch):
    monkeypatch.setattr(daemon, "SOCK", stub_server.sock)
    monkeypatch.setattr(daemon, "READY", Path(stub_server.sock + ".ready"))
    assert daemon.ping() and daemon.wait_ready(5)
    st = stub_server.ask({"op": "status"})
    assert st["ok"] and st["model"] == "stub" and st["indexed"] is False
    assert json.loads((stub_server.tmp / "status.json").read_text())["model"] == "stub"


def test_index_then_answers(stub_server):
    r = stub_server.index()
    assert r["ok"] is True and r["records"] > 50 and r["vlm_done"] == 0 and "seconds" in r
    assert (stub_server.index_dir / "INDEXED").exists()
    assert stub_server.ask({"op": "status"})["indexed"] is True

    a = stub_server.answer(Q4)
    assert a["answer"] == "4.3.2" and a["citations"] == ["support/bug_database.csv"] and a["confidence"] > 0
    assert a["trace"]["reason"] == "OK"
    b = stub_server.answer(Q9)
    assert b["answer"] == "4.3.2" and b["citations"] == ["logs/prod_inference_2026-09-02.log", "support/bug_database.csv"]
    c = stub_server.answer("What is the unit price of the TQ-40 at 10,000 unit volume?")
    assert (c["answer"], c["citations"], c["confidence"]) == ("", [], 0.0)
    d = stub_server.answer("Which firmware version fixed ticket ORR-9999?")
    assert d["answer"] == "" and d["trace"]["reason"] == "FALSE_PREMISE"


def test_bad_requests_do_not_kill_the_server(stub_server):
    assert "error" in stub_server.raw(b"this is not json\n")
    assert "error" in stub_server.ask({"op": "frobnicate"})
    r = stub_server.ask({"op": "answer"})                     # no query at all
    assert r["answer"] == "" and r["citations"] == [] and r["confidence"] == 0.0 and "error" in r["trace"]
    r = stub_server.ask({"op": "index"})                      # no corpus
    assert "error" in r
    assert stub_server.ask({"op": "ping"})["ok"] is True
    assert stub_server.proc.poll() is None
    assert stub_server.answer(Q4)["answer"] == "4.3.2"        # still serving real requests


def test_answer_without_index_op_loads_persisted_index(stub_server, tmp_path):
    """`app.py --index` persists the files; a fresh server answers from disk without ever getting an index op."""
    stub_server.index()                                           # idempotent; guarantees INDEXED on disk
    assert (stub_server.index_dir / "INDEXED").exists()
    srv = Srv(tmp_path, "stub", preload_index=stub_server.index_dir)
    try:
        assert srv.ask({"op": "status"})["indexed"] is False
        a = srv.answer(Q4)
        assert a["answer"] == "4.3.2" and a["citations"] == ["support/bug_database.csv"]
        assert srv.ask({"op": "status"})["indexed"] is True
    finally:
        srv.stop()


def test_sigterm_exits_cleanly_and_removes_socket(tmp_path):
    srv = Srv(tmp_path, "stub")
    assert srv.proc.poll() is None
    rc = srv.stop()
    assert rc == 0
    assert not Path(srv.sock).exists()


def test_cpu_mode_without_reader_answers_via_fallback(tmp_path):
    srv = Srv(tmp_path, "none")
    try:
        assert srv.ask({"op": "status"})["model"] == "cpu"
        r = srv.index()
        assert r["ok"] and r["vlm_done"] == 0
        a = srv.answer(Q4)
        assert a["answer"] == "4.3.2" and a["citations"] == ["support/bug_database.csv"]
        assert a["trace"]["reason"] == "MODE_CPU"
        b = srv.answer(Q1)                                    # fallback is conservative: refuses
        assert (b["answer"], b["citations"], b["confidence"]) == ("", [], 0.0)
    finally:
        srv.stop()


def test_answer_op_times_out_with_a_refusal_and_keeps_serving(tmp_path, monkeypatch):
    """In-process: a reader that outlives the deadline yields TIMEOUT instead of blocking the socket."""
    import server
    from eval.stub_reader import StubReader
    from schema import ReaderOut

    monkeypatch.setattr(server, "STATUS_PATH", tmp_path / "status.json")

    class Slow(StubReader):
        def read(self, *a, **kw):
            time.sleep(2.5)
            return ReaderOut("")

    st = server.State()
    st.reader = Slow("refuse")
    st.set_mode("stub")
    st.handle({"op": "index", "corpus": str(SAMPLE_CORPUS), "index_dir": str(tmp_path / "idx"),
               "deadline_ts": time.time() + 60})
    t = time.time()
    r = st.handle({"op": "answer", "query": Q4, "corpus": str(SAMPLE_CORPUS), "deadline_ts": time.time() + 1.5})
    assert time.time() - t < 2.4
    assert (r["answer"], r["citations"], r["confidence"]) == ("", [], 0.0) and r["trace"]["reason"] == "TIMEOUT"
    st.worker.join(5)
    assert st.handle({"op": "status"})["indexed"] is True
