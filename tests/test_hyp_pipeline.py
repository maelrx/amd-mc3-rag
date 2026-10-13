"""H2 (chain guard) and H3 (gate B log mode, fail-open fix, call-time thresholds) behind their flags."""

import json

import pytest

import config
import gates
import pipeline
from budget import Deadline
from eval.stub_reader import StubReader, parse_package
from schema import ReaderOut

LOG = "logs/prod_inference_2026-09-02.log"
CSV = "support/bug_database.csv"
LOG_TEXT = "\n".join([
    "2026-09-02T03:11:04Z inference-node-07 meridian[2211]: INFO  scheduler: batch 88213 accepted, 64 requests",
    "2026-09-02T03:14:22Z inference-node-07 meridian[2211]: WARN  thermal: die 0 at 91C, approaching limit",
    "2026-09-02T03:14:41Z inference-node-07 meridian[2211]: ERROR E7731: thermal throttle engaged on die 0",
    "2026-09-02T03:19:03Z inference-node-07 meridian[2211]: INFO  incident logged against ORR-1847",
]) + "\n"
CSV_TEXT = """ticket,component,severity,summary,status,fixed_in
ORR-1847,power,S2,Die temperature sensor reports high under sustained small-batch load,closed,4.3.2
ORR-1802,pcie,S3,Doorbell hang under sustained small-batch load,closed,4.3.1
ORR-1866,thermal,S3,Thermal throttle engages early on die 1 when inlet is above 30 C,closed,4.3.4
ORR-1755,thermal,S3,Fan curve too aggressive at idle,closed,4.2.0
"""
Q9 = "The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?"
TRUE, DECOY = "4.3.2", "4.3.4"


@pytest.fixture(scope="module")
def chain_idx(tmp_path_factory):
    corpus = tmp_path_factory.mktemp("chain_corpus")
    for rel, text in ((LOG, LOG_TEXT), (CSV, CSV_TEXT)):
        (corpus / rel).parent.mkdir(parents=True, exist_ok=True)
        (corpus / rel).write_text(text, encoding="utf-8")
    return pipeline.index_corpus(corpus, tmp_path_factory.mktemp("chain_idx")), corpus


class DecoyThenTrue(StubReader):
    """Answers the decoy row while the package holds it (and any other csv rows); the true value only on a package
    restricted to the chain. Without the decoy row (v5's own coverage second look) it finds nothing."""

    def __init__(self):
        super().__init__("refuse")
        self.packages: list[str] = []

    def read(self, package_text, question, images, deadline, max_new_tokens=96):
        self.calls.append("read")
        self.packages.append(package_text)
        recs = parse_package(package_text)
        has = lambda needle: any(needle in t for _, _, _, t in recs)  # noqa: E731
        want = TRUE if not has("ORR-1802") else (DECOY if has("ORR-1866") else "")
        rids = [rid for rid, _, _, t in recs if want and f"fixed_in={want}" in t]
        return self._out(want, rids[:1], [])


def run_q9(chain_idx, reader):
    idx, corpus = chain_idx
    return pipeline.answer(idx, reader, Q9, Deadline.after(20), corpus)


def test_chain_guard_off_is_v5_decoy(chain_idx, monkeypatch):
    monkeypatch.setattr(config, "CHAIN_GUARD", False)
    rd = DecoyThenTrue()
    res = run_q9(chain_idx, rd)
    assert res.answer == DECOY and res.citations == [CSV]
    assert "chain_guard" not in res.trace["notes"] and len(rd.packages) == 2   # v5: first read + coverage second look


def test_chain_guard_on_replaces_decoy_and_cites_log(chain_idx, monkeypatch):
    monkeypatch.setattr(config, "CHAIN_GUARD", True)
    rd = DecoyThenTrue()
    res = run_q9(chain_idx, rd)
    assert res.answer == TRUE and res.citations == sorted([LOG, CSV])
    notes = res.trace["notes"]
    assert notes["chain_guard"] == "replaced" and notes["chain_guard_reread"] == "OK"
    assert len(rd.packages) == 3                                             # read, v5 coverage look, guard re-read
    assert "ORR-1866" not in rd.packages[2] and "ORR-1847" in rd.packages[2] and "ORR-1802" not in rd.packages[2]
    assert res.trace["reason"] == "OK"


def test_chain_guard_keeps_original_when_reread_fails(chain_idx, monkeypatch):
    monkeypatch.setattr(config, "CHAIN_GUARD", True)

    class StubbornDecoy(DecoyThenTrue):
        def read(self, package_text, question, images, deadline, max_new_tokens=96):
            out = super().read(package_text, question, images, deadline, max_new_tokens)
            return out if "ORR-1802" in package_text else self._out("", [], [])

    res = run_q9(chain_idx, StubbornDecoy())
    assert res.answer == DECOY and res.citations == [CSV]                    # never a refusal
    assert res.trace["notes"]["chain_guard"] == "kept"
    assert res.trace["notes"]["chain_guard_reason"] == "reread_not_verified"


def test_chain_guard_needs_time(chain_idx, monkeypatch):
    monkeypatch.setattr(config, "CHAIN_GUARD", True)
    idx, corpus = chain_idx
    monkeypatch.setattr(pipeline.chainguard, "REREAD_MIN_REMAINING_S", 1e9)
    rd = DecoyThenTrue()
    res = pipeline.answer(idx, rd, Q9, Deadline.after(20), corpus)
    assert res.answer == DECOY and res.trace["notes"]["chain_guard_reason"] == "no_time" and len(rd.packages) == 2


def test_chain_guard_error_keeps_answer(chain_idx, monkeypatch):
    monkeypatch.setattr(config, "CHAIN_GUARD", True)
    monkeypatch.setattr(pipeline.chainguard, "restricted_package", lambda *a, **kw: 1 / 0)
    res = run_q9(chain_idx, DecoyThenTrue())
    assert res.answer == DECOY and res.trace["notes"]["chain_guard_reason"] == "error:ZeroDivisionError"


def test_chain_guard_inactive_without_named_kind(chain_idx, monkeypatch):
    monkeypatch.setattr(config, "CHAIN_GUARD", True)
    idx, corpus = chain_idx
    rd = DecoyThenTrue()
    res = pipeline.answer(idx, rd, "Which firmware release fixed the thermal throttle defect?", Deadline.after(20), corpus)
    assert "chain_guard" not in res.trace["notes"] and len(rd.packages) == 1


# ---------------------------------------------------------------- gate B

SAMPLE_Q4 = "Which firmware version fixed ticket ORR-1847?"
GOLD = {SAMPLE_Q4: {"expected_answer": "4.3.2", "expected_citations": [CSV]}}


class Scored(StubReader):
    """Oracle with a fixed mean_logprob and a counted, fixed logprob."""

    def __init__(self, s1=-5.0, lp=-1.0):
        super().__init__("oracle", GOLD)
        self.s1, self.lp = s1, lp

    def read(self, *a, **kw):
        o = super().read(*a, **kw)
        return ReaderOut(o.answer, o.support, o.bridge, o.raw, o.n_new_tokens, mean_logprob=self.s1)

    def logprob(self, *a, **kw):
        self.calls.append("logprob")
        return self.lp


def ask(chain_idx, reader, deadline=20):
    idx, corpus = chain_idx
    return pipeline.answer(idx, reader, SAMPLE_Q4, Deadline.after(deadline), corpus)


def test_gate_b_off_makes_no_logprob_calls(chain_idx, monkeypatch):
    monkeypatch.setattr(config, "GATE_B_MODE", "0")
    monkeypatch.setattr(config, "GATE_B", False)
    rd = Scored()
    res = ask(chain_idx, rd)
    assert res.answer == "4.3.2" and "logprob" not in rd.calls
    assert "gate_b_s1" not in res.trace["notes"] and "context_gain" not in res.trace["notes"]


def test_gate_b_log_records_and_never_refuses(chain_idx, monkeypatch):
    monkeypatch.setattr(config, "GATE_B_MODE", "log")
    monkeypatch.setattr(config, "GATE_B", False)
    rd = Scored(s1=-5.0)                                                   # below theta1 and gain 0 < theta2
    res = ask(chain_idx, rd)
    notes = res.trace["notes"]
    assert res.answer == "4.3.2" and res.trace["reason"] == "OK"
    assert notes["gate_b_s1"] == -5.0 and notes["context_gain"] == 0.0 and notes["gate_b_would_refuse"] is True
    assert rd.calls.count("logprob") == 2
    res = ask(chain_idx, Scored(s1=-0.1))
    assert res.trace["notes"]["gate_b_would_refuse"] is False and res.answer == "4.3.2"


def test_gate_b_log_with_missing_mean_logprob(chain_idx, monkeypatch):
    monkeypatch.setattr(config, "GATE_B_MODE", "log")
    res = ask(chain_idx, Scored(s1=None))                                     # mean_logprob None (reader gives none)
    notes = res.trace["notes"]
    assert res.answer == "4.3.2" and notes["gate_b_s1"] is None and notes["context_gain"] == 0.0
    assert notes["gate_b_would_refuse"] is True                                 # gain 0 < theta2, s1 unavailable


def test_gate_b_log_without_time_records_none(chain_idx, monkeypatch):
    monkeypatch.setattr(config, "GATE_B_MODE", "log")
    monkeypatch.setattr(config, "LOO_MIN_REMAINING_S", 1e9)
    rd = Scored(s1=None)
    res = ask(chain_idx, rd)
    notes = res.trace["notes"]
    assert res.answer == "4.3.2" and notes["context_gain"] is None and notes["gate_b_would_refuse"] is False
    assert "logprob" not in rd.calls


def test_gate_b_enforce_refuses_below_thresholds(chain_idx, monkeypatch):
    monkeypatch.setattr(config, "GATE_B_MODE", "1")
    res = ask(chain_idx, Scored(s1=-5.0))
    assert res.answer == "" and res.trace["reason"] == "GATE_B"
    assert ask(chain_idx, Scored(s1=-0.1)).answer == "4.3.2"                  # s1 >= theta1 answers


def test_gate_b_thresholds_read_at_call_time(chain_idx, monkeypatch):
    out = ReaderOut("x", mean_logprob=-1.0)
    assert gates.gate_b(out, None) is False                                    # -1.0 < -0.7
    monkeypatch.setattr(config, "GATE_B_THETA1", -2.0)
    assert gates.gate_b(out, None) is True
    monkeypatch.setattr(config, "GATE_B_THETA2", -3.0)
    assert gates.gate_b(ReaderOut("x", mean_logprob=-9.0), -2.0) is True       # gain -2 >= theta2 -3
    assert gates.gate_b(out, None, theta1=0.0) is False                        # explicit thresholds win
    monkeypatch.setattr(config, "GATE_B_MODE", "1")
    assert ask(chain_idx, Scored(s1=-1.5)).answer == "4.3.2"                   # pipeline sees the patched theta1


def test_gate_b_exception_keeps_answer_and_notes(chain_idx, monkeypatch):
    monkeypatch.setattr(config, "GATE_B_MODE", "1")

    def boom(*a, **kw):
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(gates, "gate_b", boom)
    res = ask(chain_idx, Scored(s1=-5.0))
    assert res.answer == "4.3.2" and res.trace["reason"] == "OK"
    assert "gate exploded" in res.trace["notes"]["gate_b_error"]
    assert res.citations == [CSV]
