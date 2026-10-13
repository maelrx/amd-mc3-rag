import pytest

import index as ix
import parse
import textnorm
import gates
import verify
from conftest import SAMPLE_CORPUS
from schema import Package, Q, ReaderOut, Verdict

CSV = "support/bug_database.csv"
LOG = "logs/prod_inference_2026-09-02.log"
R1 = "specs/tq40_datasheet_r1_WITHDRAWN.pdf"


@pytest.fixture(scope="module")
def idx():
    return ix.build(parse.walk(SAMPLE_CORPUS))


def rid_of(idx, file, needle):
    for r in idx.records:
        if r.file == file and not r.meta and textnorm.find_span(r.text, needle):
            return r.rid
    raise LookupError((file, needle))


Q9 = Q("Which firmware version fixed the ticket logged in the production log incident?",
       attr_terms=["version"], named_kinds={"log"})


def test_gate_a_pre_false_premise(idx):
    q = Q("Which version fixed ticket ORR-9999?", ids=["ORR-9999"], rare_ids=["ORR-9999"])
    v = gates.gate_a_pre(q, idx)
    assert v.reason == "FALSE_PREMISE" and v.hard and not v.ok


def test_gate_a_pre_ok(idx):
    q = Q("Which version fixed ticket ORR-1847?", ids=["ORR-1847"], rare_ids=["ORR-1847"])
    assert gates.gate_a_pre(q, idx) is None
    assert gates.gate_a_pre(Q("no ids"), idx) is None


def test_gate_a_pre_only_in_ineligible():
    pf = {"path": "old.txt", "kind": "txt", "status": "ok", "records": [{"loc": "L1", "text": "part ZX-4411"}],
          "text": "part ZX-4411", "images": [], "flags": {"withdrawn": True, "supersedes": [], "revision": None}}
    mi = ix.build([pf])
    q = Q("price of ZX-4411", ids=["ZX-4411"], rare_ids=["ZX-4411"])
    v = gates.gate_a_pre(q, mi)
    assert v.reason == "ONLY_IN_INELIGIBLE" and v.hard


def test_gate_a_post_chain_ok(idx):
    csv, log = rid_of(idx, CSV, "ORR-1847"), rid_of(idx, LOG, "incident logged against ORR-1847")
    p = Package(rids=[csv, log], text="", via={csv: log})
    v = verify.verify(Q9, ReaderOut("4.3.2", support=[csv]), p, idx)
    assert gates.gate_a_post(Q9, v, idx) == (True, "OK")


def test_gate_a_post_chain_open(idx):
    csv = rid_of(idx, CSV, "ORR-1847")
    stray = rid_of(idx, LOG, "ERROR E7731")
    v = Verdict(True, "OK", answer_span="4.3.2", support_rid=csv, bridge_rids=[stray])
    assert gates.gate_a_post(Q9, v, idx) == (False, "CHAIN_OPEN")


def test_gate_a_post_two_link_chain(idx):
    csv, log = rid_of(idx, CSV, "ORR-1847"), rid_of(idx, LOG, "incident logged against ORR-1847")
    v = Verdict(True, "OK", support_rid=csv, bridge_rids=[log])
    assert gates.gate_a_post(Q9, v, idx) == (True, "OK")


def test_gate_a_post_ineligible_support(idx):
    r1 = rid_of(idx, R1, "105")
    v = Verdict(True, "OK", support_rid=r1)
    assert gates.gate_a_post(Q9, v, idx) == (False, "INELIGIBLE_ONLY")
    q = Q("old", wants_old=True)
    assert gates.gate_a_post(q, v, idx) == (True, "OK")


def test_gate_a_post_not_ok(idx):
    assert gates.gate_a_post(Q9, Verdict(False, "NO_SPAN"), idx) == (False, "NO_SPAN")


@pytest.mark.parametrize("lp,gain,expected", [
    (-0.5, None, True),    # >= theta1
    (-0.7, None, True),    # boundary
    (-1.0, None, False),
    (None, 2.5, True),     # >= theta2
    (None, 2.0, True),     # boundary
    (None, 1.0, False),
    (-1.0, 3.0, True),     # either signal suffices
    (-0.2, 0.0, True),
    (-2.0, 0.5, False),
    (None, None, True),    # no signal -> answer
])
def test_gate_b(lp, gain, expected):
    assert gates.gate_b(ReaderOut("x", mean_logprob=lp), gain) is expected


def test_gate_b_custom_thresholds():
    assert gates.gate_b(ReaderOut("x", mean_logprob=-1.0), None, theta1=-1.5, theta2=9) is True
    assert gates.gate_b(ReaderOut("x", mean_logprob=-1.0), 3.0, theta1=-0.1, theta2=4.0) is False
