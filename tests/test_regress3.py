"""Regressions from the third batch (ranking, drafts, parser guards, index/--query robustness, server, snap, units, OCR glue)."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest

import app as app_mod
import config
import daemon
import index as ix
import parse
import retrieve
import snap
import textnorm
import verify
import vision
from conftest import APP, SAMPLE_CORPUS
from schema import Package, ReaderOut

REFUSAL = {"answer": "", "citations": [], "confidence": 0.0}


def mk(files):
    """files: (path, kind, [texts], flags-dict | None)."""
    parsed = []
    for path, kind, recs, *rest in files:
        parsed.append({"path": path, "kind": kind, "status": "ok",
                       "records": [{"loc": f"L{i}", "text": t} for i, t in enumerate(recs, 1)],
                       "flags": rest[0] if rest else {}, "text": " ".join(recs), "images": []})
    return ix.build(parsed)


def rid(idx, file, needle):
    return next(r.rid for r in idx.records if r.file == file and not r.meta and needle in r.text)


# ---------------------------------------------------------------- A: ranking

def test_a_hop_target_beats_a_direct_decoy():
    idx = mk([("logs/prod.log", "log", ["fabric link flap on node ORR-1111 at 03:00"]),
              ("support/bugs.csv", "csv", ["ticket=ORR-1111; fixed_in=4.3.2"]),
              ("notes/release_notes.txt", "txt", ["firmware release notes: link flap fixed in release 4.3.2 for fabric"])])
    q = retrieve.analyze("The production log shows a fabric link flap, which firmware release fixed it?", idx)
    target, decoy = rid(idx, "support/bugs.csv", "4.3.2"), rid(idx, "notes/release_notes.txt", "4.3.2")
    rids = [rid(idx, "logs/prod.log", "ORR-1111"), target, decoy]
    out = ReaderOut("4.3.2", [decoy], [])
    plain = verify.verify(q, out, Package(rids=rids, text="", via={}), idx)
    hopped = verify.verify(q, out, Package(rids=rids, text="", via={target: rids[0]}), idx)
    assert plain.ok and plain.support_rid == decoy        # without the hop the decoy covers more words
    assert hopped.ok and hopped.support_rid == target     # the chain's end wins
    assert idx.record(rids[0]).file in {idx.record(b).file for b in hopped.bridge_rids}


def test_a_authority_beats_model_hint_on_a_coverage_tie():
    idx = mk([("specs/tq40_datasheet.pdf", "pdf", ["Maximum junction temperature 94 C"]),
              ("marketing/tq40_blurb.txt", "txt", ["Maximum junction temperature 94 C"])])
    q = retrieve.analyze("What is the maximum junction temperature of the TQ-40?", idx)
    pdf, txt = rid(idx, "specs/tq40_datasheet.pdf", "94"), rid(idx, "marketing/tq40_blurb.txt", "94")
    v = verify.verify(q, ReaderOut("94", [txt], []), Package(rids=[txt, pdf], text=""), idx)
    assert v.ok and v.support_rid == pdf


# ---------------------------------------------------------------- B: drafts

def _docx(path: Path, text: str) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml",
                   '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                   f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>")


def _build_dir(root: Path) -> ix.Index:
    return ix.build(parse.walk(root))


def test_b_draft_loses_to_non_draft_sibling(tmp_path):
    _docx(tmp_path / "roadmap_fy29_DRAFT.docx", "Launch in Q2 FY29")
    _docx(tmp_path / "roadmap_fy29.docx", "Launch in Q3 FY29")
    idx = _build_dir(tmp_path)
    draft, final = idx.file_by_path["roadmap_fy29_DRAFT.docx"], idx.file_by_path["roadmap_fy29.docx"]
    assert draft.draft and not final.draft
    assert draft.superseded_by == "roadmap_fy29.docx" and not draft.eligible
    assert final.eligible


def test_b_lone_draft_stays_eligible(tmp_path):
    _docx(tmp_path / "plan_DRAFT.docx", "Plan text")
    fe = _build_dir(tmp_path).file_by_path["plan_DRAFT.docx"]
    assert fe.draft and fe.eligible and fe.superseded_by is None


def test_b_draft_in_title_and_word_boundaries(tmp_path):
    (tmp_path / "memo.txt").write_text("\nMEMO (Draft)\nbody\n")
    (tmp_path / "memo_final.txt").write_text("Memo\nbody\n")
    (tmp_path / "drafting_guide.txt").write_text("How to write\nbody\n")        # 'drafting' is not 'draft'
    (tmp_path / "late.txt").write_text("one\ntwo\nthree\nthis is a draft of nothing\n")   # 4th line: not a title
    idx = _build_dir(tmp_path)
    assert idx.file_by_path["memo.txt"].draft and not idx.file_by_path["memo.txt"].eligible
    assert idx.file_by_path["memo_final.txt"].eligible
    assert not idx.file_by_path["drafting_guide.txt"].draft
    assert not idx.file_by_path["late.txt"].draft


def test_b_withdrawn_sibling_does_not_rescue_a_draft_from_staying_eligible():
    idx = mk([("a/spec_DRAFT.txt", "txt", ["x 1"], {"draft": True}),
              ("a/spec.txt", "txt", ["x 2"], {"withdrawn": True})])
    assert idx.file_by_path["a/spec_DRAFT.txt"].eligible


# ---------------------------------------------------------------- C: parser guards

def test_c_constants():
    assert parse.PDF_MAX_PAGES == 500 and parse.PDF_VISION_PAGES == 60


def test_c_csv_field_over_default_limit(tmp_path):
    (tmp_path / "big.csv").write_text("id,note\n1," + "x" * 300_000 + "\n2,short\n")
    rec = parse.extract(tmp_path, tmp_path / "big.csv")
    assert rec["status"] == "ok" and len(rec["records"]) == 3


def test_c_pdf_encrypted_pypdf_errors_fall_through(monkeypatch, tmp_path):
    import pypdf
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-1.4\n")
    for exc, expected in ((pypdf.errors.PdfStreamError("s"), False), (pypdf.errors.PdfReadError("r"), False),
                          (pypdf.errors.FileNotDecryptedError("d"), True)):
        def boom(*a, _e=exc, **k):
            raise _e
        monkeypatch.setattr(pypdf, "PdfReader", boom)
        assert parse._pdf_encrypted(p) is expected


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_c_fifo_is_an_error_without_opening(tmp_path):
    os.mkfifo(tmp_path / "pipe.txt")
    (tmp_path / "ok.txt").write_text("hello\n")
    signal.alarm(10)     # opening a FIFO with no writer would block forever
    try:
        recs = {r["path"]: r for r in parse.walk(tmp_path)}
    finally:
        signal.alarm(0)
    assert recs["pipe.txt"]["status"] == "error" and recs["pipe.txt"]["records"] == []
    assert recs["ok.txt"]["status"] == "ok"


def test_c_formula_pow_is_refused_and_arithmetic_works():
    assert parse._eval_formula("=2^3", {}) is None
    assert parse._eval_formula("=9**9**9", {}) is None
    assert parse._eval_formula("=1+2*3", {}) == 7
    assert parse._eval_formula("=SUM(1,2,3)", {}) == 6


def test_c_xlsx_row_cap_and_big_merge(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["part", "price"])
    for i in range(1, 20_301):
        ws.append([f"P{i}", i])
    ws.merge_cells("D1:Z600")          # 23 x 600 = 13,800 cells > 10,000: not propagated
    wb.save(tmp_path / "big.xlsx")
    t = time.time()
    rec = parse.extract(tmp_path, tmp_path / "big.xlsx")
    assert time.time() - t < parse.FILE_TIMEOUT_S
    assert rec["status"] == "ok"
    rows = [int(r["loc"].split("!R")[1]) for r in rec["records"] if "!R" in r["loc"]]
    assert max(rows) <= parse.XLSX_MAX_ROWS and len(rec["records"]) <= parse.MAX_RECORDS
    assert not any("D500" in r["text"] for r in rec["records"])


def test_c_record_cap_notes_the_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(parse, "MAX_RECORDS", 5)
    (tmp_path / "long.txt").write_text("\n".join(f"line {i}" for i in range(20)))
    rec = parse.extract(tmp_path, tmp_path / "long.txt")
    assert len(rec["records"]) == 5 and rec["records"][0]["text"] == "line 0" and "truncated" in rec["note"]


def test_c_per_file_timeout_discards_records_and_restores_timer(tmp_path, monkeypatch):
    def slow(text, py=False):
        t = time.time()
        while time.time() - t < 5:
            pass
        return [{"loc": "L1", "text": "never", "stale": False}]
    monkeypatch.setattr(parse, "_text_records", slow)
    monkeypatch.setattr(parse, "FILE_TIMEOUT_S", 0.3)
    (tmp_path / "a.txt").write_text("hi\n")
    before = signal.getsignal(signal.SIGALRM)
    t = time.time()
    rec = parse.extract(tmp_path, tmp_path / "a.txt")
    assert time.time() - t < 3
    assert rec["status"] == "error" and rec["error"] == "timeout" and rec["records"] == []
    assert signal.getsignal(signal.SIGALRM) == before and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_c_timeout_guard_is_skipped_off_the_main_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(parse, "FILE_TIMEOUT_S", 0.05)
    (tmp_path / "a.txt").write_text("hi\n")
    box = {}
    t = threading.Thread(target=lambda: box.update(rec=parse.extract(tmp_path, tmp_path / "a.txt")))
    t.start()
    t.join(10)
    assert box["rec"]["status"] == "ok"


def test_c_ids_dedup_keeps_order_and_is_linear():
    assert textnorm.ids("AB-12 CD-34 AB-12 EF-56 CD-34") == ["AB-12", "CD-34", "EF-56"]
    t = time.time()
    got = textnorm.ids(" ".join(f"AB-{i}" for i in range(30_000)))
    assert len(got) == 30_000 and time.time() - t < 5


# ---------------------------------------------------------------- D: index robustness

def test_d_surrogate_filename_survives_save_and_load(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    bad = os.fsdecode(b"bad_\xff_name.txt")
    with open(os.path.join(os.fsencode(corpus), os.fsencode(bad)), "wb") as fh:
        fh.write(b"serial number SN-1234567 here\n")
    (corpus / "ok.txt").write_text("fine\n")
    idx = ix.build(parse.walk(corpus))
    assert bad in idx.file_by_path
    ix.save(tmp_path / "idx", idx)
    assert (tmp_path / "idx" / "records.jsonl").read_bytes().isascii()
    back = ix.load(tmp_path / "idx")
    assert bad in back.file_by_path and any(r.file == bad and "SN-1234567" in r.text for r in back.records)
    new = ix.append_records(tmp_path / "idx", [back.records[0].__class__(rid=0, file=bad, loc="vlm.L1", text="x", src="vlm")])
    assert new == [len(back.records)]
    assert ix.load(tmp_path / "idx").records[-1].file == bad


def test_d_index_build_failure_still_asks_the_daemon_to_build(tmp_path, monkeypatch):
    """The client-side build raises (index dir under a regular file): the index op carries build=True."""
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    sent = []
    monkeypatch.setattr(config, "INDEX_DIR", str(blocker / "idx"))
    monkeypatch.setattr(app_mod, "_wait_daemon", lambda *a, **k: True)
    monkeypatch.setattr(daemon, "request", lambda payload, timeout=0: sent.append(payload) or {"ok": True})
    assert app_mod.run_index(SAMPLE_CORPUS) == 0
    assert len(sent) == 1 and sent[0]["op"] == "index" and sent[0]["build"] is True
    assert sent[0]["corpus"] == str(SAMPLE_CORPUS)


def test_d_successful_client_build_does_not_force_a_daemon_build(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setattr(app_mod, "_wait_daemon", lambda *a, **k: True)
    monkeypatch.setattr(daemon, "request", lambda payload, timeout=0: sent.append(payload) or {"ok": True})
    assert app_mod.run_index(SAMPLE_CORPUS) == 0
    assert len(sent) == 1 and "build" not in sent[0]


def test_d_daemon_builds_from_missing_or_partial_index_dir(tmp_path):
    import pipeline
    for setup in ("missing", "partial", "rebuild"):
        d = tmp_path / setup / "idx"
        if setup == "partial":      # INDEXED marker but no records.jsonl: unreadable, so rebuilt
            d.mkdir(parents=True)
            (d / "INDEXED").write_text("1\n")
            (d / "files.json").write_text("[]")
        idx = pipeline.index_corpus(SAMPLE_CORPUS, d)
        assert len(idx.records) > 50 and (d / "INDEXED").exists() and (d / "records.jsonl").exists()
    # rebuild=True ignores a (stale but loadable) INDEXED index
    d = tmp_path / "rebuild" / "idx"
    (d / "records.jsonl").write_text("")
    idx = pipeline.index_corpus(SAMPLE_CORPUS, d, rebuild=True)
    assert len(idx.records) > 50


def test_d_server_index_op_passes_build_flag(tmp_path, monkeypatch):
    import server
    from eval.stub_reader import StubReader
    monkeypatch.setattr(server, "STATUS_PATH", tmp_path / "status.json")
    seen = {}
    import pipeline
    real = pipeline.index_corpus

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)
    monkeypatch.setattr(pipeline, "index_corpus", spy)
    st = server.State()
    st.reader = StubReader("refuse")
    r = st.handle({"op": "index", "corpus": str(SAMPLE_CORPUS), "index_dir": str(tmp_path / "idx"),
                   "deadline_ts": time.time() + 60, "build": True})
    assert r["ok"] and seen["rebuild"] is True


# ---------------------------------------------------------------- E: --query / --index robustness

def _env(out_dir, index_dir, sock, **extra):
    env = dict(os.environ)
    env.update({"MC2_OUTPUT_DIR": str(out_dir), "MC3_INDEX_DIR": str(index_dir), "MC3_SOCK": str(sock),
                "MC3_READY": str(Path(str(sock) + ".ready")), "MC3_AUTOSPAWN": "0"})
    env.update(extra)
    return env


@pytest.fixture
def fake_daemon(tmp_path):
    sock = tmp_path / "s.sock"
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


def test_e_refusal_exists_before_the_daemon_answers(tmp_path, fake_daemon):
    state, sock = fake_daemon
    gate = threading.Event()
    state["handler"] = lambda req: gate.wait(20) and {"answer": "94", "citations": ["a.pdf"], "confidence": 0.9}
    env = _env(tmp_path / "out", tmp_path / "idx", sock)
    p = subprocess.Popen([sys.executable, str(APP / "app.py"), "--corpus", str(SAMPLE_CORPUS), "--query-id",
                          "query_01", "--query", "anything"], env=env, stderr=subprocess.DEVNULL)
    out = tmp_path / "out" / "query_01_output.json"
    try:
        for _ in range(100):
            if out.exists():
                break
            time.sleep(0.05)
        assert json.loads(out.read_text()) == REFUSAL          # killed now, the file is already valid
    finally:
        gate.set()
        p.wait(30)
    assert json.loads(out.read_text())["answer"] == "94"


def test_e_query_starting_with_dash_is_recovered(tmp_path, fake_daemon):
    state, sock = fake_daemon
    seen = []
    state["handler"] = lambda req: seen.append(req) or {"answer": "7", "citations": ["a.txt"], "confidence": 0.9}
    env = _env(tmp_path / "out", tmp_path / "idx", sock)
    r = subprocess.run([sys.executable, str(APP / "app.py"), "--corpus", str(SAMPLE_CORPUS), "--query", "-40C?",
                        "--query-id", "query_02"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert seen and seen[0]["query"] == "-40C?" and seen[0]["query_id"] == "query_02"
    assert json.loads((tmp_path / "out" / "query_02_output.json").read_text())["answer"] == "7"


def test_e_unparsable_command_lines_exit_zero(tmp_path):
    env = _env(tmp_path / "out", tmp_path / "idx", tmp_path / "dead.sock")
    r = subprocess.run([sys.executable, str(APP / "app.py"), "--query-id=query_03", "--query"], env=env,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and json.loads((tmp_path / "out" / "query_03_output.json").read_text()) == REFUSAL
    r = subprocess.run([sys.executable, str(APP / "app.py"), "--index"], env=env, capture_output=True, text=True,
                       timeout=30)
    assert r.returncode == 0


def test_e_scan_flag_and_recover_args():
    argv = ["--corpus", "/c", "--query", "-x y", "--query-id", "query_09"]
    assert app_mod._scan_flag(argv, "--query") == "-x y"
    assert app_mod._scan_flag(["--query-id=query_1"], "--query-id") == "query_1"
    assert app_mod._scan_flag(["--query"], "--query") is None
    rec = app_mod._recover_args(argv)
    assert rec.query_id == "query_09" and rec.query == "-x y" and str(rec.corpus) == "/c"
    assert app_mod._recover_args(["--query", "q"]) is None


@pytest.mark.parametrize("resp", [{"answer": "94", "citations": [], "confidence": 0.9},
                                  {"answer": "", "citations": ["a.pdf"], "confidence": 0.9},
                                  {"answer": "  ", "citations": ["a.pdf"], "confidence": 0.9}])
def test_e_incoherent_daemon_answer_is_a_refusal(resp, monkeypatch):
    monkeypatch.setattr(daemon, "request", lambda payload, timeout=0: resp)
    out, via = app_mod.answer_query(SAMPLE_CORPUS, "query_01", "q", time.time())
    assert out == REFUSAL and via == "refusal"


def test_e_wait_daemon_keeps_waiting_while_the_model_loads(tmp_path, monkeypatch):
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"model": "loading"}))
    monkeypatch.setattr(config, "STATUS_PATH", str(status))
    monkeypatch.setattr(app_mod, "_server_running", lambda: True)
    calls = []

    def ready(timeout):
        calls.append(timeout)
        return len(calls) >= 5
    monkeypatch.setattr(daemon, "wait_ready", ready)
    assert app_mod._wait_daemon(time.time() - 1, time.time() + 100) is True
    assert len(calls) >= 5


def test_e_wait_daemon_gives_up_when_not_loading_or_no_server(tmp_path, monkeypatch):
    status = tmp_path / "status.json"
    monkeypatch.setattr(config, "STATUS_PATH", str(status))
    monkeypatch.setattr(daemon, "wait_ready", lambda timeout: False)
    monkeypatch.setattr(app_mod, "_server_running", lambda: True)
    status.write_text(json.dumps({"model": "failed"}))
    t = time.time()
    assert app_mod._wait_daemon(time.time() - 1, time.time() + 100) is False
    status.write_text(json.dumps({"model": "loading"}))
    monkeypatch.setattr(app_mod, "_server_running", lambda: False)
    assert app_mod._wait_daemon(time.time() - 1, time.time() + 100) is False
    assert app_mod._wait_daemon(time.time() - 1) is False
    assert time.time() - t < 5


# ---------------------------------------------------------------- F: server

class _FakeCuda:
    def __init__(self, available_after: int, total_gib: float = 80.0):
        self.calls, self.after, self.total = 0, available_after, total_gib * 2**30
        self.fraction = None

    def is_available(self):
        self.calls += 1
        return self.calls > self.after

    def get_device_properties(self, i):
        return type("P", (), {"total_memory": self.total})()

    def set_per_process_memory_fraction(self, f):
        self.fraction = f


def _fake_torch(monkeypatch, cuda):
    monkeypatch.setitem(sys.modules, "torch", type(sys)("torch"))
    sys.modules["torch"].cuda = cuda


class _FakeReader:
    force_math = True

    def __init__(self, path):
        pass

    def load(self):
        pass

    def self_test_sdpa(self):
        return False

    def warmup(self):
        pass


def test_f_gpu_retry_memory_cap_and_force_math(tmp_path, monkeypatch):
    import reader
    import server
    cuda = _FakeCuda(available_after=3)
    _fake_torch(monkeypatch, cuda)
    monkeypatch.setattr(reader, "Reader", _FakeReader)
    monkeypatch.setattr(server, "STATUS_PATH", tmp_path / "s.json")
    monkeypatch.setattr(server, "GPU_POLL_S", 0.01)
    monkeypatch.setattr(config, "PKG_TOKENS", config.PKG_TOKENS)      # restored by monkeypatch
    monkeypatch.setattr(config, "READER", "auto")
    st = server.State()
    st.boot()
    assert st.mode == "loaded" and cuda.calls == 4
    assert cuda.fraction == pytest.approx(44.0 / 80.0)
    assert config.PKG_TOKENS == (4000, 6000)


def test_f_small_gpu_caps_at_one_and_no_gpu_means_cpu(tmp_path, monkeypatch):
    import reader
    import server
    cuda = _FakeCuda(available_after=0, total_gib=24.0)
    _fake_torch(monkeypatch, cuda)
    monkeypatch.setattr(reader, "Reader", _FakeReader)
    monkeypatch.setattr(server, "STATUS_PATH", tmp_path / "s.json")
    monkeypatch.setattr(config, "PKG_TOKENS", config.PKG_TOKENS)
    monkeypatch.setattr(config, "READER", "auto")
    server.State().boot()
    assert cuda.fraction == 1.0

    never = _FakeCuda(available_after=10**9)
    _fake_torch(monkeypatch, never)
    monkeypatch.setattr(server, "GPU_WAIT_S", 0.1)
    monkeypatch.setattr(server, "GPU_POLL_S", 0.02)
    st = server.State()
    st.boot()
    assert st.mode == "cpu" and never.calls >= 3


def test_f_supervisor_restarts_child_and_forwards_sigterm(tmp_path, monkeypatch):
    sock = str(tmp_path / "s.sock")
    status = tmp_path / "status.json"
    env = dict(os.environ)
    env.update({"MC3_SUPERVISE": "1", "MC3_READER": "none", "MC3_SOCK": sock, "MC3_READY": sock + ".ready",
                "MC3_INDEX_DIR": str(tmp_path / "idx"), "MC3_STATUS": str(status),
                "MC3_SERVER_LOG": str(tmp_path / "server.log"), "PYTHONUNBUFFERED": "1"})
    sup = subprocess.Popen([sys.executable, str(APP / "server.py")], env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

    def child_pid(timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                st = json.loads(status.read_text())
                if st.get("model") == "cpu" and st["pid"] != sup.pid and Path(sock + ".ready").exists() \
                        and os.path.exists(f"/proc/{st['pid']}"):
                    return st["pid"]
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.1)
        raise AssertionError("server child not ready")
    try:
        first = child_pid()
        assert first != sup.pid
        os.kill(first, signal.SIGKILL)
        for _ in range(100):
            if not os.path.exists(f"/proc/{first}") or open(f"/proc/{first}/stat").read().split()[2] == "Z":
                break
            time.sleep(0.1)
        os.unlink(sock + ".ready") if os.path.exists(sock + ".ready") else None
        status.unlink()
        second = child_pid(40)
        assert second != first and sup.poll() is None
        sup.send_signal(signal.SIGTERM)
        assert sup.wait(20) == 0
        assert not os.path.exists(f"/proc/{second}")
    finally:
        if sup.poll() is None:
            sup.kill()
            sup.wait()


# ---------------------------------------------------------------- G: snap / textnorm

@pytest.mark.parametrize("span,expected", [
    ("100 kHz", "100"), ("85 µs", "85"), ("85µs", "85"), ("45 dBA", "45"), ("2.2 kΩ", "2.2"), ("30 psi", "30"),
    ("3 years", "3"), ("95 °C max", "95"), ("−20", "-20"), ("−20 C", "-20"), ("-20", "-20"),
    ("B14", "B14"), ("REV-C2", "REV-C2"), ("4.3.2", "4.3.2"), ("Q3 FY27", "Q3 FY27"), ("14B", "14B"),
    ("15%", "15%"), ("ORR-FAN-2214-B", "ORR-FAN-2214-B"), ("NB‑23", "NB-23"), ("1,200 units", "1,200"),
])
def test_g_clean_tail_units_and_dashes(span, expected):
    assert snap.clean(span) == expected


def test_g_tail_unit_values_are_not_identifiers():
    for s in ("100 kHz", "85 µs", "2.2 kΩ", "3 years", "95 °C max"):
        assert textnorm.is_identifier(s) is False
    for s in ("B14", "REV-C2", "6.9.1", "Q3 FY27"):
        assert textnorm.is_identifier(s) is True


def test_g_find_span_accepts_a_glued_leading_v_for_dotted_versions():
    for hay in ("fixed in v6.9.1.", "fixed in V6.9.1, ok", "(v6.9.1)"):
        a, b = textnorm.find_span(hay, "6.9.1")
        assert hay[a:b] == "6.9.1"
    assert textnorm.find_span("xv6.9.1", "6.9.1") is None
    assert textnorm.find_span("v6.9.12", "6.9.1") is None
    assert textnorm.find_span("v6.9", "6.9.1") is None
    assert snap.find("6.9.1", "firmware v6.9.1 shipped") == "6.9.1"
    assert textnorm.find_span("v7", "7") is None            # only dotted versions


# ---------------------------------------------------------------- H: numbers in context, unit lexicon

def test_h_number_found_in_the_path_conditions_the_record():
    idx = mk([("v2.0/release_manifest.txt", "txt", ["manifest header", "build_id=BLD-4465-C; status=active"]),
              ("v3.0/release_manifest.txt", "txt", ["manifest header", "build_id=BLD-9999-D; status=active"])])
    q = retrieve.analyze("In release 2.0, which build id is active?", idx)
    assert q.numbers == ["2.0"]
    r2, r3 = rid(idx, "v2.0/release_manifest.txt", "BLD-4465"), rid(idx, "v3.0/release_manifest.txt", "BLD-9999")
    assert verify._check(q, idx, r2, "BLD-4465-C") is None
    assert verify._check(q, idx, r3, "BLD-9999-D") == "COND_MISMATCH"


@pytest.mark.parametrize("text,unit", [
    ("What is the memory in GB?", "GB"), ("capacity in GiB", "GiB"), ("size in gigabytes", "GB"),
    ("size in mebibytes", "MiB"), ("size in MB", "MB"), ("in TB", "TB"), ("in TiB", "TiB"),
    ("fan speed in RPM", "RPM"), ("clock in kHz", "kHz"), ("clock in MHz", "MHz"), ("clock in GHz", "GHz"),
    ("rate in Hz", "Hz"), ("flow in L/min", "L/min"), ("bandwidth in TB/s", "TB/s"), ("in seconds/frame", "s"),
])
def test_h_question_unit_forms(text, unit):
    assert retrieve.analyze(text).unit == unit


@pytest.mark.parametrize("row,asked,ok", [
    ("memory=16 GiB", "What is the memory of the NB-40 in GB?", False),
    ("memory=16 GB", "What is the memory of the NB-40 in GB?", True),
    ("memory=16 GB", "What is the memory of the NB-40 in GiB?", False),
    ("memory=16 MiB", "What is the memory of the NB-40 in MB?", False),
    ("memory=16 GiB", "What is the memory of the NB-40 in GiB?", True),
    ("flow=12 L/min", "What is the coolant flow of the NB-40 in L/min?", True),
    ("flow=12 L/min", "What is the coolant flow of the NB-40 in RPM?", False),
])
def test_h_gb_is_not_gib(row, asked, ok):
    idx = mk([("specs/nb40.txt", "txt", ["NB-40 data", f"NB-40 {row}"])])
    q = retrieve.analyze(asked, idx)
    r = rid(idx, "specs/nb40.txt", row)
    span = row.split("=")[1].split()[0]
    assert (verify._check(q, idx, r, span) is None) is ok


# ---------------------------------------------------------------- I: OCR glue

@pytest.mark.parametrize("raw,split", [
    ("BOARD REVISIONREV-A4", "BOARD REVISION REV-A4"), ("SERIALCS-836046-T", "SERIAL CS-836046-T"),
    ("SerialCS-836046-T", "Serial CS-836046-T"), ("REV-C2", "REV-C2"), ("SERIAL CS-836046-T", "SERIAL CS-836046-T"),
    ("noise 45 dBA", "noise 45 dBA"), ("storage eMMC 64", "storage eMMC 64"), ("B14  THERM_ALERT#", "B14  THERM_ALERT#"),
])
def test_i_split_glue(raw, split):
    assert vision.split_glue(raw) == split
    assert vision.split_glue(split) == split          # idempotent


def test_i_ocr_image_indexes_the_split_form_and_keeps_the_raw(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image
    line = lambda t, y: {"box": [0, y, 200, y + 20], "text": t, "conf": 0.99}   # noqa: E731
    monkeypatch.setattr(vision, "_run", lambda e, img: [line("BOARD REVISIONREV-A4", 0), line("SERIALCS-836046-T", 40)])
    monkeypatch.setattr(vision, "get_ocr", lambda: object())
    res = vision.ocr_image(Image.new("RGB", (300, 100), "white"), "k", tmp_path)
    assert res["layout"] == ["BOARD REVISION REV-A4", "SERIAL CS-836046-T"]
    assert res["raw_layout"] == ["BOARD REVISIONREV-A4", "SERIALCS-836046-T"]
    assert [ln["text"] for ln in res["lines"]] == ["BOARD REVISIONREV-A4", "SERIALCS-836046-T"]
    cached = json.loads((tmp_path / "k.ocr.json").read_text())
    assert cached["raw_layout"] == res["raw_layout"] and cached["layout"] == res["layout"]


def test_i_ocr_all_splits_rows_from_old_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "get_ocr", lambda: object())
    monkeypatch.setattr(vision, "ocr_file", lambda corpus, rel, cd: {
        "lines": [], "rot": 0, "layout": ["BOARD REVISIONREV-A4"], "pairs": ["SERIAL: SERIALCS-836046-T"]})
    parsed = [{"path": "l.png", "images": [("file", "l.png")]}]
    out = vision.ocr_all(parsed, tmp_path, tmp_path / "c", time.time() + 10)["l.png"]
    assert out["lines"] == ["BOARD REVISION REV-A4"] and out["pairs"] == ["SERIAL: SERIAL CS-836046-T"]


# ---------------------------------------------------------------- J: config

def test_j_config_tunables():
    assert config.RARE_DF == 5 and config.PKG_FILES == (8, 12)
