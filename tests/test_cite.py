import pytest

import index as ix
import parse
import textnorm
import cite
import verify
from conftest import SAMPLE_CORPUS
from schema import Package, Q, ReaderOut, Verdict

R1 = "specs/tq40_datasheet_r1_WITHDRAWN.pdf"
R2 = "specs/tq40_datasheet_r2.pdf"
XLSX = "support/rma_parts.xlsx"
CSV = "support/bug_database.csv"
LOG = "logs/prod_inference_2026-09-02.log"
PY = "engineering/ingest_service.py"


@pytest.fixture(scope="module")
def idx():
    return ix.build(parse.walk(SAMPLE_CORPUS))


def rid_of(idx, file, needle):
    for r in idx.records:
        if r.file == file and not r.meta and textnorm.find_span(r.text, needle):
            return r.rid
    raise LookupError((file, needle))


def pkg(rids, via=None):
    return Package(rids=sorted(rids), text="", via=via or {})


Q1 = Q("What is the maximum junction temperature of the TQ-40?", entities=["TQ-40"], attr_terms=["temperature"])
Q3 = Q("What is the part number of the fan assembly for the TQ-40?", entities=["TQ-40"],
       attr_terms=["part_number"])
Q4 = Q("Which firmware version fixed ticket ORR-1847?", ids=["ORR-1847"], rare_ids=["ORR-1847"],
       attr_terms=["version"])
Q6 = Q("What is the default batch timeout in seconds?", unit="s", attr_terms=["timeout"])
Q9 = Q("Which firmware version fixed the ticket logged in the production log incident?",
       attr_terms=["version"], named_kinds={"log"})


def run(idx, q, out, p):
    v = verify.verify(q, out, p, idx)
    assert v.ok, v.reason
    return v, cite.cite(q, v, idx, pkg=p)


def test_q1(idx):
    r2 = rid_of(idx, R2, "Maximum junction temperature")
    r1 = rid_of(idx, R1, "105")
    _, c = run(idx, Q1, ReaderOut("94", support=[r2]), pkg([r1, r2]))
    assert c == ["specs/tq40_datasheet_r2.pdf"]


def test_q3(idx):
    r = rid_of(idx, XLSX, "ORR-FAN-2214-B")
    _, c = run(idx, Q3, ReaderOut("ORR-FAN-2214-B", support=[r]), pkg([r]))
    assert c == [XLSX]


def test_q4_bridge_dropped_when_id_in_question(idx):
    csv, log = rid_of(idx, CSV, "ORR-1847"), rid_of(idx, LOG, "incident logged against ORR-1847")
    for bridge in ([], [log]):
        v, c = run(idx, Q4, ReaderOut("4.3.2", support=[csv], bridge=bridge), pkg([csv, log]))
        assert c == [CSV]
    v, c = run(idx, Q4, ReaderOut("4.3.2", support=[csv], bridge=[log]), pkg([csv, log], via={csv: log}))
    assert c == [CSV]


def test_q6(idx):
    const = rid_of(idx, PY, "code constant DEFAULT_BATCH_TIMEOUT_S")
    _, c = run(idx, Q6, ReaderOut("180", support=[const]), pkg([const]))
    assert c == [PY]


def test_q9_both_files(idx):
    csv, log = rid_of(idx, CSV, "ORR-1847"), rid_of(idx, LOG, "incident logged against ORR-1847")
    p = pkg([csv, log], via={csv: log})
    for bridge in ([log], []):
        _, c = run(idx, Q9, ReaderOut("4.3.2", support=[csv], bridge=bridge), p)
        assert c == sorted([CSV, LOG])


def test_wants_old_cites_r1(idx):
    q = Q("What did the withdrawn revision 1 say the maximum junction temperature of the TQ-40 was?",
          entities=["TQ-40"], attr_terms=["temperature"], wants_old=True)
    r1 = rid_of(idx, R1, "105")
    _, c = run(idx, q, ReaderOut("105", support=[r1]), pkg([r1]))
    assert c == [R1]


def test_not_ok_verdict_cites_nothing(idx):
    assert cite.cite(Q1, Verdict(False, "NO_SPAN"), idx) == []


def test_defensive_prefix_strip():
    assert cite._rel("/app/corpus/a/b.pdf") == "a/b.pdf"
    assert cite._rel("./a/b.pdf") == "a/b.pdf"
    assert cite._rel("a/b.pdf") == "a/b.pdf"


def test_premise_named_bridge_without_shared_id(idx):
    """A vetted bridge of a named kind is cited even when no rare id links it textually."""
    csv = rid_of(idx, CSV, "ORR-1847")
    log = rid_of(idx, LOG, "ERROR E7731")
    v = Verdict(True, "OK", answer_span="4.3.2", support_rid=csv, bridge_rids=[log], premise_files=[LOG])
    assert cite.cite(Q9, v, idx, pkg=pkg([csv, log])) == sorted([CSV, LOG])
    v.premise_files = []
    assert cite.cite(Q9, v, idx, pkg=pkg([csv, log])) == [CSV]


def test_loo_noop_without_reader(idx):
    cands = [1, 2]
    assert cite.loo_select(cands, None, pkg([1, 2]), Q1, None, answer="x") is cands


def test_loo_picks_best_when_time(idx):
    class R:
        def logprob(self, pkg_text, question, images, answer):
            return -1.0 if "[R2]" in pkg_text else -5.0

    class D:
        def __init__(self, rem):
            self.rem = rem

        def require(self, s):
            return self.rem >= s

    p = Package(rids=[1, 2], text="[R1] a\n[R2] b")
    assert cite.loo_select([1, 2], R(), p, Q1, D(20), answer="x") == [2]
    assert cite.loo_select([1, 2], R(), p, Q1, D(1), answer="x") == [1, 2]
    assert cite.loo_select([1, 2], R(), p, Q1, None, answer="x") == [1, 2]
