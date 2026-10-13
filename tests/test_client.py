"""app.py as the harness runs it: a subprocess per call, no torch, optional fake daemon on a unix socket."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

import daemon
from conftest import APP, SAMPLE_CORPUS

APP_PY = APP / "app.py"
Q4 = "Which firmware version fixed ticket ORR-1847?"
Q1 = "What is the maximum junction temperature of the TQ-40?"
TQ40 = "specs/tq40_datasheet_r2.pdf"


@pytest.fixture(scope="module")
def built_index(tmp_path_factory):
    """A real index of the sample corpus, built by `app.py --index` with no daemon around."""
    d = tmp_path_factory.mktemp("client_index")
    env = _env(d.parent / "unused_out", d, d / "none.sock")
    env["MC3_INDEX_BUDGET_S"] = "62"       # daemon wait = 2 s
    r = subprocess.run([sys.executable, str(APP_PY), "--index", str(SAMPLE_CORPUS)], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return d


def _env(out_dir, index_dir, sock, **extra):
    env = dict(os.environ)
    env.update({"MC2_OUTPUT_DIR": str(out_dir), "MC3_INDEX_DIR": str(index_dir), "MC3_SOCK": str(sock),
                "MC3_READY": str(Path(str(sock) + ".ready")), "MC3_AUTOSPAWN": "0"})
    env.update(extra)
    return env


def _query(env, qid, q, corpus=SAMPLE_CORPUS, timeout=40):
    t0 = time.time()
    r = subprocess.run([sys.executable, str(APP_PY), "--corpus", str(corpus), "--query-id", qid, "--query", q],
                       env=env, capture_output=True, text=True, timeout=timeout)
    return r, time.time() - t0


def _read_out(env, qid):
    return json.loads((Path(env["MC2_OUTPUT_DIR"]) / f"{qid}_output.json").read_text(encoding="utf-8"))


def _assert_shape(d):
    assert isinstance(d, dict) and set(d) == {"answer", "citations", "confidence"}
    assert isinstance(d["answer"], str) and isinstance(d["confidence"], float)
    assert isinstance(d["citations"], list) and all(isinstance(c, str) for c in d["citations"])


@pytest.fixture
def fake_daemon():
    """Run daemon.serve(handler) in this process on a private socket; the handler is swappable per test."""
    d = Path(tempfile.mkdtemp(prefix="mc3t"))
    sock = d / "s.sock"
    state = {"handler": lambda req: {}}
    saved = (daemon.SOCK, daemon.READY)
    daemon.SOCK, daemon.READY = str(sock), Path(str(sock) + ".ready")
    threading.Thread(target=daemon.serve, args=(lambda req: state["handler"](req),), daemon=True).start()
    for _ in range(100):
        if daemon.READY.exists():
            break
        time.sleep(0.05)
    yield state, sock
    daemon.SOCK, daemon.READY = saved


# ---------------------------------------------------------------- --index

def test_index_without_daemon(tmp_path):
    out, idx_dir = tmp_path / "out", tmp_path / "index"
    env = _env(out, idx_dir, tmp_path / "x.sock", MC3_INDEX_BUDGET_S="70")   # daemon wait = 10 s
    t0 = time.time()
    r = subprocess.run([sys.executable, str(APP_PY), "--index", str(SAMPLE_CORPUS)], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert time.time() - t0 < 60
    for name in ("files.json", "records.jsonl", "ids.json", "INDEXED", "status.json"):
        assert (idx_dir / name).is_file(), name
    status = json.loads((idx_dir / "status.json").read_text())
    assert status["files"] >= 13 and status["records"] > 50
    assert {"files", "records", "images", "t_parse", "t_ocr"} <= set(status)
    assert not list(idx_dir.glob("*.tmp")) and not list(idx_dir.glob(".*.tmp"))
    assert "indexed files=" in r.stdout


def test_index_asks_daemon_to_index(tmp_path, fake_daemon):
    state, sock = fake_daemon
    seen = []
    state["handler"] = lambda req: seen.append(req) or {"ok": True, "records": 1}
    env = _env(tmp_path / "out", tmp_path / "index", sock, MC3_INDEX_BUDGET_S="120")
    r = subprocess.run([sys.executable, str(APP_PY), "--index", str(SAMPLE_CORPUS)], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert len(seen) == 1 and seen[0]["op"] == "index"
    assert seen[0]["corpus"] == str(SAMPLE_CORPUS) and seen[0]["index_dir"] == str(tmp_path / "index")
    assert seen[0]["deadline_ts"] > time.time()


def test_index_bad_corpus_still_exits_zero(tmp_path):
    env = _env(tmp_path / "out", tmp_path / "index", tmp_path / "x.sock", MC3_INDEX_BUDGET_S="62")
    r = subprocess.run([sys.executable, str(APP_PY), "--index", str(tmp_path / "does-not-exist")], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0


# ---------------------------------------------------------------- --query, no daemon

def test_query_no_daemon_fallback_hit(tmp_path, built_index):
    env = _env(tmp_path / "out", built_index, tmp_path / "dead.sock")
    r, dt = _query(env, "query_04", Q4)
    assert r.returncode == 0, r.stderr
    d = _read_out(env, "query_04")
    _assert_shape(d)
    assert d["answer"] == "4.3.2" and d["citations"] == ["support/bug_database.csv"]
    assert "via=fallback" in r.stderr and dt < 10


def test_query_no_daemon_refuses_q1(tmp_path, built_index):
    env = _env(tmp_path / "out", built_index, tmp_path / "dead.sock")
    r, _ = _query(env, "query_01", Q1)
    assert r.returncode == 0, r.stderr
    d = _read_out(env, "query_01")
    _assert_shape(d)
    assert d == {"answer": "", "citations": [], "confidence": 0.0}
    assert "via=refusal" in r.stderr


def test_query_no_daemon_no_index(tmp_path):
    env = _env(tmp_path / "out", tmp_path / "no-index", tmp_path / "dead.sock")
    r, _ = _query(env, "query_02", Q4)
    assert r.returncode == 0
    assert _read_out(env, "query_02") == {"answer": "", "citations": [], "confidence": 0.0}


def test_output_dir_created_and_default_env(tmp_path):
    env = _env(tmp_path / "deep" / "er" / "out", tmp_path / "no-index", tmp_path / "dead.sock")
    r, _ = _query(env, "query_10", Q1)
    assert r.returncode == 0
    assert (tmp_path / "deep" / "er" / "out" / "query_10_output.json").is_file()


# ---------------------------------------------------------------- --query, fake daemon

def test_query_daemon_answer_strips_corpus_prefix(tmp_path, fake_daemon, built_index):
    state, sock = fake_daemon
    seen = []

    def handler(req):
        seen.append(req)
        return {"answer": "94", "citations": ["/app/corpus/" + TQ40, "./" + TQ40, "support/a.csv"],
                "confidence": 0.9, "trace": {}}

    state["handler"] = handler
    env = _env(tmp_path / "out", built_index, sock)
    t0 = time.time()
    r, _ = _query(env, "query_01", Q1)
    assert r.returncode == 0, r.stderr
    d = _read_out(env, "query_01")
    _assert_shape(d)
    assert d == {"answer": "94", "citations": [TQ40, "support/a.csv"], "confidence": 0.9}
    assert "via=daemon" in r.stderr
    req = seen[0]
    assert req["op"] == "answer" and req["query"] == Q1 and req["query_id"] == "query_01"
    assert req["corpus"] == str(SAMPLE_CORPUS)
    assert t0 < req["deadline_ts"] <= time.time() + 22


def test_query_daemon_refusal_is_honoured(tmp_path, fake_daemon, built_index):
    state, sock = fake_daemon
    state["handler"] = lambda req: {"answer": "", "citations": [], "confidence": 0.0}
    env = _env(tmp_path / "out", built_index, sock)
    r, _ = _query(env, "query_04", Q4)      # fallback would have answered; a valid daemon refusal must stand
    assert _read_out(env, "query_04") == {"answer": "", "citations": [], "confidence": 0.0}
    assert "via=daemon" in r.stderr


def test_query_daemon_confidence_defaults(tmp_path, fake_daemon, built_index):
    state, sock = fake_daemon
    state["handler"] = lambda req: {"answer": "94", "citations": [TQ40]}
    env = _env(tmp_path / "out", built_index, sock)
    _query(env, "query_01", Q1)
    assert _read_out(env, "query_01")["confidence"] == 0.0


def test_query_handler_raises_falls_back(tmp_path, fake_daemon, built_index):
    state, sock = fake_daemon

    def boom(req):
        raise RuntimeError("boom")

    state["handler"] = boom
    env = _env(tmp_path / "out", built_index, sock)
    r, _ = _query(env, "query_04", Q4)
    assert r.returncode == 0
    d = _read_out(env, "query_04")
    _assert_shape(d)
    assert (d["answer"], d["citations"]) == ("4.3.2", ["support/bug_database.csv"])
    r, _ = _query(env, "query_01", Q1)
    assert _read_out(env, "query_01") == {"answer": "", "citations": [], "confidence": 0.0}


@pytest.mark.parametrize("bad", [{"answer": 94, "citations": []}, {"answer": "94"},
                                 {"answer": "94", "citations": "x"}, {"answer": "94", "citations": [1]},
                                 {"answer": "94", "citations": [], "error": "x"}, [], "str"])
def test_query_malformed_daemon_reply(tmp_path, fake_daemon, built_index, bad):
    state, sock = fake_daemon
    state["handler"] = lambda req: bad
    env = _env(tmp_path / "out", built_index, sock)
    r, _ = _query(env, "query_01", Q1)
    assert r.returncode == 0
    assert _read_out(env, "query_01") == {"answer": "", "citations": [], "confidence": 0.0}


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_query_slow_daemon_times_out(tmp_path, fake_daemon, built_index):
    state, sock = fake_daemon
    state["handler"] = lambda req: time.sleep(4) or {"answer": "late", "citations": [TQ40], "confidence": 1.0}
    env = _env(tmp_path / "out", built_index, sock, MC3_CLIENT_TIMEOUT_S="2")
    r, dt = _query(env, "query_04", Q4)
    assert r.returncode == 0
    assert dt < 6, dt
    d = _read_out(env, "query_04")
    _assert_shape(d)
    assert d["answer"] == "4.3.2"
    time.sleep(2.2)     # let the sleeping handler finish so it does not leak into later tests


# ---------------------------------------------------------------- robustness

def test_write_is_atomic_no_tmp_left(tmp_path, fake_daemon, built_index):
    state, sock = fake_daemon
    state["handler"] = lambda req: {"answer": "94", "citations": [TQ40], "confidence": 0.9}
    env = _env(tmp_path / "out", built_index, sock)
    for i in range(3):
        _query(env, "query_01", Q1)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["query_01_output.json"]
    raw = (tmp_path / "out" / "query_01_output.json").read_text(encoding="utf-8")
    assert json.loads(raw)["answer"] == "94"


def test_unicode_answer_not_escaped(tmp_path, fake_daemon, built_index):
    state, sock = fake_daemon
    state["handler"] = lambda req: {"answer": "94°C", "citations": [TQ40], "confidence": 0.5}
    env = _env(tmp_path / "out", built_index, sock)
    _query(env, "query_01", Q1)
    # written with ensure_ascii (robust to odd file names); the decoded value must survive intact
    out = json.loads((tmp_path / "out" / "query_01_output.json").read_text(encoding="utf-8"))
    assert out["answer"] == "94°C"


def test_refusal_written_when_internals_crash(tmp_path, monkeypatch):
    """An exception (even KeyboardInterrupt) inside the answer path still leaves a refusal and returns 0."""
    import app
    import config

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    for exc in (RuntimeError("x"), KeyboardInterrupt(), SystemExit(3)):
        def boom(*a, **k):
            raise exc
        monkeypatch.setattr(app, "answer_query", boom)
        assert app.main(["--corpus", str(SAMPLE_CORPUS), "--query-id", "query_07", "--query", Q1]) == 0
        assert json.loads((tmp_path / "out" / "query_07_output.json").read_text()) == \
            {"answer": "", "citations": [], "confidence": 0.0}
        (tmp_path / "out" / "query_07_output.json").unlink()


def test_query_id_without_query_still_writes_refusal(tmp_path):
    env = _env(tmp_path / "out", tmp_path / "idx", tmp_path / "dead.sock")
    r = subprocess.run([sys.executable, str(APP_PY), "--query-id", "query_05"], env=env,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert _read_out(env, "query_05") == {"answer": "", "citations": [], "confidence": 0.0}


def test_no_args_is_argparse_error(tmp_path):
    r = subprocess.run([sys.executable, str(APP_PY)], env=_env(tmp_path, tmp_path, tmp_path / "s"),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "--query-id" in r.stderr


def test_output_dir_unwritable_does_not_crash(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    env = _env(blocker / "out", tmp_path / "idx", tmp_path / "dead.sock")
    r, _ = _query(env, "query_01", Q1)
    assert r.returncode == 0


def test_query_path_imports_stdlib_only():
    code = ("import sys; sys.path.insert(0, %r); import app, fallback; "
            "bad = [m for m in ('torch','transformers','PIL','numpy','parse','vision') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)" % str(APP))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
