"""H6 warrant check: pipeline hook per mode, Reader.yes_no with a fake model, stub oracle, calibration grid."""

import json
from types import SimpleNamespace

import pytest

import config
import pipeline
import reader
import warrant
from budget import Deadline, Trace
from conftest import SAMPLE_CORPUS, SAMPLE_QUESTIONS
from eval import calib_warrant
from eval.stub_reader import StubReader

QDATA = json.loads(SAMPLE_QUESTIONS.read_text(encoding="utf-8"))
Q4 = "Which firmware version fixed ticket ORR-1847?"
GOLD4 = "4.3.2"


@pytest.fixture(scope="module")
def sample_idx(tmp_path_factory):
    return pipeline.index_corpus(SAMPLE_CORPUS, tmp_path_factory.mktemp("warrant_idx"))


class ScoreReader(StubReader):
    """Oracle reads; yes_no returns a fixed score (or raises)."""

    def __init__(self, score=2.0, boom=False):
        super().__init__("oracle", {Q4: {"expected_answer": GOLD4, "expected_citations": ["support/bug_database.csv"]}})
        self.score, self.boom, self.yn_args = score, boom, []

    def yes_no(self, evidence_text, question, answer, images=None, deadline=None):
        self.calls.append("yes_no")
        self.yn_args.append((evidence_text, question, answer))
        if self.boom:
            raise RuntimeError("boom")
        return self.score


def ask(idx, rd):
    tr = Trace()
    res = pipeline.answer(idx, rd, Q4, Deadline.after(20), SAMPLE_CORPUS, tr)
    return res, res.trace.get("notes", {})


@pytest.fixture
def mode(monkeypatch):
    def set_(m, theta=0.0):
        monkeypatch.setattr(config, "WARRANT", m)
        monkeypatch.setattr(config, "WARRANT_THETA", theta)
    return set_


def test_defaults_are_off():
    assert config.WARRANT == "0" and config.WARRANT_THETA == 0.0
    assert warrant.mode() == "0"


def test_off_makes_no_yes_no_calls(sample_idx, mode):
    mode("0")
    rd = ScoreReader()
    res, notes = ask(sample_idx, rd)
    assert res.answer == GOLD4 and "yes_no" not in rd.calls and "warrant" not in notes


def test_log_records_note_and_keeps_answer(sample_idx, mode):
    mode("log")
    rd = ScoreReader(score=-7.0)
    res, notes = ask(sample_idx, rd)
    assert res.answer == GOLD4 and res.citations and notes["warrant"] == -7.0
    assert rd.calls.count("yes_no") == 1
    ev, q, a = rd.yn_args[0]
    assert q == Q4 and a == GOLD4 and "4.3.2" in ev and "[R" in ev     # support record line is in the evidence


def test_enforce_refuses_below_theta_only(sample_idx, mode):
    mode("1", 0.0)
    res, notes = ask(sample_idx, ScoreReader(score=-0.5))
    assert res.answer == "" and res.citations == [] and res.trace["reason"] == "WARRANT" and notes["warrant"] == -0.5
    res, _ = ask(sample_idx, ScoreReader(score=0.0))                     # score == theta passes
    assert res.answer == GOLD4
    mode("1", 3.0)
    res, _ = ask(sample_idx, ScoreReader(score=2.9))
    assert res.trace["reason"] == "WARRANT"


def test_enforce_none_score_keeps_answer(sample_idx, mode):
    mode("1")
    res, notes = ask(sample_idx, ScoreReader(score=None))
    assert res.answer == GOLD4 and notes["warrant"] is None


def test_exception_keeps_answer(sample_idx, mode):
    for m in ("log", "1"):
        mode(m)
        res, notes = ask(sample_idx, ScoreReader(boom=True))
        assert res.answer == GOLD4 and "boom" in notes["warrant_error"]


def test_reader_without_yes_no_keeps_answer(sample_idx, mode):
    mode("1")

    rd = StubReader("oracle", {Q4: {"expected_answer": GOLD4, "expected_citations": ["support/bug_database.csv"]}})
    assert warrant.check(object(), None, None, None, None, None, None, Trace()) is None   # no yes_no at all
    res, _ = ask(sample_idx, rd)            # oracle stub: +5 for the gold answer
    assert res.answer == GOLD4


def test_evidence_has_support_name_and_heading(sample_idx):
    import retrieve
    q = retrieve.analyze(Q4, sample_idx)
    pkg = retrieve.assemble(sample_idx, q, 1)
    rd = StubReader("oracle", {Q4: {"expected_answer": GOLD4, "expected_citations": ["support/bug_database.csv"]}})
    import verify
    out = rd.read(pkg.text, Q4, [], Deadline.after(5))
    v = verify.verify(q, out, pkg, sample_idx)
    assert v.ok
    rids = warrant.evidence_rids(sample_idx, v)
    assert v.support_rid in rids and rids == sorted(rids)
    assert any(sample_idx.record(r).meta for r in rids)                  # name record of the support file
    assert warrant.evidence_text(sample_idx, v).startswith("[R")


# ---------------------------------------------------------------- stub oracle

def test_stub_yes_no_oracle_and_other_modes():
    gold = {Q4: {"expected_answer": GOLD4, "expected_citations": ["f.csv"]},
            "Canary?": {"expected_answer": "", "expected_citations": []}}
    o = StubReader("oracle", gold)
    assert o.yes_no("ev", Q4, "4.3.2") == 5.0 and o.yes_no("ev", Q4, "4.3.4") == -5.0
    assert o.yes_no("ev", Q4, "v4.3.2") == 5.0 and o.yes_no("ev", "Canary?", "99") == -5.0
    assert o.yes_no("ev", "unknown?", "1") == -5.0 and o.calls.count("yes_no") == 5
    for m in ("lexical", "adversarial", "refuse"):
        assert StubReader(m, gold).yes_no("ev", Q4, "4.3.4") == 0.0


# ---------------------------------------------------------------- Reader.yes_no

class FakeTok:
    VOCAB = {"YES": [11], " YES": [12], "Yes": [13], " Yes": [14], "yes": [15], " yes": [16],
             "NO": [21], " NO": [22], "No": [23], " No": [24], "no": [25], " no": [26]}

    def encode(self, text, add_special_tokens=False):
        return self.VOCAB.get(text, [1, 2])


def fake_reader(lp_yes=-0.1, lp_no=-3.0, with_keep=True):
    torch = pytest.importorskip("torch")
    r = reader.Reader("/nonexistent")
    r.processor = SimpleNamespace(tokenizer=FakeTok())
    seen = {}

    def fwd_keep(self, input_ids=None, use_cache=True, logits_to_keep=0):
        raise AssertionError

    def logits_for():
        v = torch.full((1, 1, 40), -50.0)
        v[0, 0, 12] = lp_yes            # " YES" is the best YES variant
        v[0, 0, 13] = lp_yes - 5
        v[0, 0, 23] = lp_no
        return v

    class Model:
        device = "cpu"

        def forward(self, input_ids=None, use_cache=True, logits_to_keep=0):
            return None

        def __call__(self, **kw):
            seen["kw"] = kw
            lg = logits_for()
            if "logits_to_keep" in kw:
                assert kw["logits_to_keep"] == 1
            return SimpleNamespace(logits=lg)

    if not with_keep:
        class Model(Model):                    # noqa: F811
            def forward(self, input_ids=None, use_cache=True):
                return None
    r.model = Model()
    r._encode = lambda msgs, images, prefilled=True: (seen.update(msgs=msgs, images=images, prefilled=prefilled)
                                                      or {"input_ids": torch.zeros(1, 5, dtype=torch.long)})
    return r, seen


def test_yes_no_scores_log_odds_and_builds_prompt():
    r, seen = fake_reader(-0.1, -3.0)
    s = r.yes_no("[R1] f.csv row1: fixed_in=4.3.2", Q4, "4.3.2", None, Deadline.after(30))
    assert s is not None
    # log_softmax shifts both sides equally: the margin is the raw logit gap
    assert s == pytest.approx(-0.1 - (-3.0), abs=1e-4)
    assert seen["prefilled"] is False and seen["kw"]["logits_to_keep"] == 1
    sysm, user = seen["msgs"]
    assert sysm["role"] == "system" and user["role"] == "user" and len(seen["msgs"]) == 2
    text = user["content"][-1]["text"]
    assert text.startswith("EVIDENCE\n[R1] f.csv row1: fixed_in=4.3.2")
    assert f"QUESTION: {Q4}\nPROPOSED ANSWER: 4.3.2" in text and text.rstrip().endswith("Reply YES or NO.")


def test_yes_no_negative_and_without_logits_to_keep():
    r, seen = fake_reader(-4.0, -0.2, with_keep=False)
    assert r.yes_no("ev", "q", "a") == pytest.approx(-3.8, abs=1e-4)
    assert "logits_to_keep" not in seen["kw"]


def test_yes_no_none_on_error_low_time_and_missing_tokens():
    r, _ = fake_reader()
    assert r.yes_no("ev", "q", "a", None, Deadline.after(1.0)) is None            # < 3 s left
    r._encode = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    assert r.yes_no("ev", "q", "a", None, Deadline.after(30)) is None
    r2, _ = fake_reader()
    r2.processor.tokenizer.VOCAB = {}                                              # every variant multi-token, same ids
    assert r2.yes_no("ev", "q", "a") is None


def test_build_warrant_messages_images_first_and_token_ids():
    msgs = reader.build_warrant_messages("EV", "Q?", "A", [("scan.png", "PIL")])
    content = msgs[1]["content"]
    assert content[0] == {"type": "image", "image": "PIL"} and "<IMAGE 1 = scan.png>" in content[1]["text"]
    yes, no = reader.warrant_token_ids(FakeTok())
    assert yes == [11, 12, 13, 14, 15, 16] and no == [21, 22, 23, 24, 25, 26]
    assert reader.warrant_margin([-1.0, -0.5], [-2.0, -3.0]) == 1.5
    assert reader.warrant_margin([], [-1.0]) is None


# ---------------------------------------------------------------- calibration

def _row(label, w, i="q"):
    exp = "" if label == "false" else "x"
    pts = 20 if label == "correct" else 0
    return {"id": i, "pts": pts, "got": {"answer": "v", "citations": []}, "expected": {"answer": exp},
            "notes": {} if w is None else {"warrant": w}}


def test_calibration_grid_on_synthetic_rows(tmp_path):
    tune = {"rows": [_row("correct", 4.0), _row("correct", 1.5), _row("correct", -0.5), _row("correct", None),
                     _row("false", -3.0), _row("false", -1.0), _row("false", 2.0),
                     _row("wrong", -2.0), _row("wrong", 3.0),
                     {"id": "r", "pts": 20, "got": {"answer": "", "citations": []}, "expected": {"answer": ""}}]}
    hold = {"rows": [_row("correct", 0.0), _row("false", -4.0), _row("false", 5.0), _row("wrong", -0.9)]}
    p1, p2 = tmp_path / "t.json", tmp_path / "h.json"
    p1.write_text(json.dumps(tune))
    p2.write_text(json.dumps(hold))
    rows = calib_warrant.collect([str(tmp_path / "t*.json")])
    assert len(rows) == 9                                                 # the refusal row is skipped
    best = calib_warrant.search(rows)
    assert best["theta"] == -0.75 and best["correct_refused"] == 0        # smallest theta with max refusals = widest margin
    assert best["false_refused"] == 2 and best["wrong_refused"] == 1
    assert calib_warrant.passes({"warrant": None}, 9.0) and not calib_warrant.passes({"warrant": -1}, 0)
    hc = calib_warrant.counts(calib_warrant.collect([str(p2)]), best["theta"])
    assert hc["correct_refused"] == 0 and hc["false_refused"] == 1 and hc["wrong_refused"] == 1
    assert calib_warrant.main(["--tune", str(p1), "--holdout", str(p2)]) == 0


def test_calibration_infeasible_returns_1(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"rows": [_row("correct", -99.0), _row("false", 1.0)]}))
    assert calib_warrant.search(calib_warrant.collect([str(p)])) is None
    assert calib_warrant.main(["--tune", str(p)]) == 1


def test_evidence_is_clipped_and_bounded():
    import warrant
    from schema import Verdict

    class R:
        def __init__(self, rid, text, file="logs/x.log", loc="L1"):
            self.rid, self.text, self.file, self.loc, self.meta = rid, text, file, loc, False

    class Idx:
        def __init__(self, recs):
            self.records = recs

        def record(self, r):
            return self.records[r]

        def context_rids(self, r):
            return [1, 2]

        def _is_heading(self, r):
            return False

    big = "x" * 500_000 + " code=E7731 " + "y" * 500_000
    recs = [R(0, big), R(1, "z" * 50_000, loc="L2"), R(2, "w" * 50_000, loc="L3")]
    v = Verdict(True, "OK", answer_span="E7731", support_rid=0, bridge_rids=[])
    txt = warrant.evidence_text(Idx(recs), v)
    assert len(txt) <= warrant.MAX_EVIDENCE_CHARS + 200
    assert "E7731" in txt
