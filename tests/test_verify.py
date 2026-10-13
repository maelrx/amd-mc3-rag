from pathlib import Path

import pytest

import index as ix
import parse
import textnorm
import verify
from conftest import SAMPLE_CORPUS
from schema import Package, Q, ReaderOut


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


R1 = "specs/tq40_datasheet_r1_WITHDRAWN.pdf"
R2 = "specs/tq40_datasheet_r2.pdf"
XLSX = "support/rma_parts.xlsx"
CSV = "support/bug_database.csv"
LOG = "logs/prod_inference_2026-09-02.log"
PY = "engineering/ingest_service.py"

Q1 = Q("What is the maximum junction temperature of the TQ-40?", entities=["TQ-40"],
       attr_terms=["temperature"], type_hint={"pdf"})
Q3 = Q("What is the part number of the field-replaceable fan assembly for the TQ-40?",
       entities=["TQ-40"], attr_terms=["part_number"], type_hint={"xlsx"})
Q4 = Q("Which firmware version fixed ticket ORR-1847?", ids=["ORR-1847"], rare_ids=["ORR-1847"],
       attr_terms=["version"])
Q6 = Q("What is the default batch timeout in seconds?", unit="s", attr_terms=["timeout"], type_hint={"py"})
Q9 = Q("Which firmware version fixed the ticket logged in the production log incident?",
       attr_terms=["version"], named_kinds={"log"})
Q10 = Q("What is the price of the TQ-40 at 10,000 units?", entities=["TQ-40"], numbers=["10,000"],
        attr_terms=["price"])


def test_empty(idx):
    v = verify.verify(Q1, ReaderOut(" "), pkg([]), idx)
    assert not v.ok and v.reason == "EMPTY"


def test_q1_r2_ok(idx):
    r1, r2 = rid_of(idx, R1, "105"), rid_of(idx, R2, "Maximum junction temperature")
    p = pkg([r1, r2, rid_of(idx, R2, "corrected to 94")])
    v = verify.verify(Q1, ReaderOut("94", support=[r2]), p, idx)
    assert v.ok and v.reason == "OK" and v.support_rid == r2
    assert v.answer_span == "94" and v.grounding_score == 1 and v.bridge_rids == []
    assert verify.answer_text(v, Q1) == "94"


def test_q1_answer_with_unit_cleans(idx):
    r2 = rid_of(idx, R2, "Maximum junction temperature")
    v = verify.verify(Q1, ReaderOut("94 C", support=[r2]), pkg([r2]), idx)
    assert v.ok and v.answer_span == "94 C" and verify.answer_text(v, Q1) == "94"


def test_q1_withdrawn_in_package_is_ineligible(idx):
    r1 = rid_of(idx, R1, "105")
    r2 = rid_of(idx, R2, "Maximum junction temperature")
    v = verify.verify(Q1, ReaderOut("105", support=[r1]), pkg([r1, r2]), idx)
    assert not v.ok and v.reason == "INELIGIBLE_ONLY" and v.violators == [r1] and not v.hard


def test_q1_withdrawn_not_in_package_is_no_span(idx):
    r1 = rid_of(idx, R1, "105")
    r2 = rid_of(idx, R2, "Maximum junction temperature")
    v = verify.verify(Q1, ReaderOut("105", support=[r1]), pkg([r2]), idx)
    assert not v.ok and v.reason == "NO_SPAN" and v.violators == [r1]


def test_wants_old_allows_withdrawn(idx):
    q = Q("What did the withdrawn revision 1 say the maximum junction temperature of the TQ-40 was?",
          entities=["TQ-40"], attr_terms=["temperature"], wants_old=True)
    r1 = rid_of(idx, R1, "105")
    v = verify.verify(q, ReaderOut("105", support=[r1]), pkg([r1]), idx)
    assert v.ok and v.support_rid == r1


def test_q3_xlsx_ok(idx):
    r = rid_of(idx, XLSX, "ORR-FAN-2214-B")
    v = verify.verify(Q3, ReaderOut("ORR-FAN-2214-B", support=[r]), pkg([r]), idx)
    assert v.ok and v.answer_span == "ORR-FAN-2214-B" and v.grounding_score == 1
    assert verify.answer_text(v, Q3) == "ORR-FAN-2214-B"


def test_q3_stale_legacy_row_is_ineligible(idx):
    r = rid_of(idx, XLSX, "ORR-FAN-1990-A")
    assert idx.record(r).stale
    v = verify.verify(Q3, ReaderOut("ORR-FAN-1990-A", support=[r]), pkg([r]), idx)
    assert not v.ok and v.reason == "INELIGIBLE_ONLY"


def _mini(rows):
    """Index with one xlsx whose rows are given (loc, text, stale)."""
    rec = [{"loc": loc, "text": t, "stale": s} for loc, t, s in rows]
    pf = {"path": "parts.xlsx", "kind": "xlsx", "status": "ok", "records": rec,
          "text": " ".join(t for _, t, _ in rows), "images": [],
          "flags": {"withdrawn": False, "supersedes": [], "revision": None}}
    return ix.build([pf])


def test_near_miss_row_entity_mismatch():
    mi = _mini([
        ("S!R1", "[sheet S] columns: part_number | description | compatible_with", False),
        ("S!R2", "[sheet S] part_number=ORR-FAN-2214-B; description=Fan dual; compatible_with=TQ-40", False),
        ("S!R3", "[sheet S] part_number=ORR-HSK-1180-A; description=Heatsink; compatible_with=TQ-40", False),
        ("S!R4", "[sheet S] part_number=ORR-FAN-1990-A; description=Fan single; compatible_with=TQ-20", False),
    ])
    assert mi.file_by_path["parts.xlsx"].n_products > 1
    r = rid_of(mi, "parts.xlsx", "ORR-FAN-1990-A")
    v = verify.verify(Q3, ReaderOut("ORR-FAN-1990-A", support=[r]), pkg([r]), mi)
    assert not v.ok and v.reason == "ENTITY_MISMATCH" and v.violators == [r]
    good = rid_of(mi, "parts.xlsx", "ORR-FAN-2214-B")
    v = verify.verify(Q3, ReaderOut("ORR-FAN-1990-A", support=[r, good]), pkg([r, good]), mi)
    assert v.reason == "ENTITY_MISMATCH"


def test_single_product_file_entity_anywhere():
    mi = _mini([("S!R1", "TQ-40 spare parts", False),
                ("S!R2", "x1", False), ("S!R3", "y2", False), ("S!R4", "z3", False),
                ("S!R5", "part_number=ORR-CBL-3301-C; description=cable", False)])
    r = rid_of(mi, "parts.xlsx", "ORR-CBL-3301-C")
    v = verify.verify(Q3, ReaderOut("ORR-CBL-3301-C", support=[r]), pkg([r]), mi)
    assert v.ok


def test_q4_csv(idx):
    r = rid_of(idx, CSV, "ORR-1847")
    v = verify.verify(Q4, ReaderOut("4.3.2", support=[r]), pkg([r]), idx)
    assert v.ok and v.answer_span == "4.3.2" and v.bridge_rids == [] and v.premise_files == []


def test_q6_const_ok_comment_rejected(idx):
    const = rid_of(idx, PY, "code constant DEFAULT_BATCH_TIMEOUT_S")
    comment = rid_of(idx, PY, "Raised from 60")
    v = verify.verify(Q6, ReaderOut("180", support=[const]), pkg([const, comment]), idx)
    assert v.ok and v.support_rid in (const, rid_of(idx, PY, "code default accept"))
    v = verify.verify(Q6, ReaderOut("60", support=[comment]), pkg([const, comment]), idx)
    assert not v.ok and v.reason == "INELIGIBLE_ONLY" and v.violators == [comment]


def test_unit_mismatch(idx):
    const = rid_of(idx, PY, "code constant DEFAULT_BATCH_TIMEOUT_S")
    q = Q("What is the batch timeout in milliseconds?", unit="ms", attr_terms=["timeout"])
    v = verify.verify(q, ReaderOut("180", support=[const]), pkg([const]), idx)
    assert not v.ok and v.reason == "UNIT_MISMATCH"
    q2 = Q("timeout in seconds", unit="seconds", attr_terms=["timeout"])
    assert verify.verify(q2, ReaderOut("180", support=[const]), pkg([const]), idx).ok


def test_unit_from_span_tail(idx):
    r2 = rid_of(idx, R2, "Maximum junction temperature")
    q = Q("Max junction temperature in W?", entities=["TQ-40"], unit="W")
    v = verify.verify(q, ReaderOut("94", support=[r2]), pkg([r2]), idx)
    assert not v.ok and v.reason == "UNIT_MISMATCH"


def test_cond_mismatch_numbers(idx):
    r = rid_of(idx, XLSX, "ORR-FAN-2214-B")
    q = Q("What is the part at 10,000 units?", numbers=["10,000"], attr_terms=["part_number"])
    v = verify.verify(q, ReaderOut("ORR-FAN-2214-B", support=[r]), pkg([r]), idx)
    assert not v.ok and v.reason == "COND_MISMATCH"


def test_q10_no_span(idx):
    r = rid_of(idx, XLSX, "ORR-FAN-2214-B")
    v = verify.verify(Q10, ReaderOut("84.50", support=[r]), pkg([r]), idx)
    # the fan row holds a numerically equal cost, so the span may be found; the 10,000-unit
    # condition then rejects it. Either way Q10 must be refused.
    assert not v.ok and v.reason in ("NO_SPAN", "COND_MISMATCH")


def test_q9_bridge_via_chain(idx):
    csv, log = rid_of(idx, CSV, "ORR-1847"), rid_of(idx, LOG, "incident logged against ORR-1847")
    p = pkg([csv, log], via={csv: log})
    for bridge in ([log], []):
        v = verify.verify(Q9, ReaderOut("4.3.2", support=[csv], bridge=bridge), p, idx)
        assert v.ok and v.bridge_rids == [log] and v.premise_files == [LOG]


def test_bridge_without_link_dropped(idx):
    csv = rid_of(idx, CSV, "ORR-1847")
    other = rid_of(idx, LOG, "ERROR E7731")
    v = verify.verify(Q9, ReaderOut("4.3.2", support=[csv], bridge=[other]), pkg([csv, other]), idx)
    assert v.ok and v.bridge_rids == []


def test_bridge_model_only_shared_id(idx):
    csv, log = rid_of(idx, CSV, "ORR-1847"), rid_of(idx, LOG, "incident logged against ORR-1847")
    v = verify.verify(Q4, ReaderOut("4.3.2", support=[csv], bridge=[log]), pkg([csv, log]), idx)
    assert v.ok and v.bridge_rids == [log] and v.premise_files == []


def test_false_premise_hard(idx):
    q = Q("Which firmware version fixed ticket ORR-9999?", ids=["ORR-9999"], rare_ids=["ORR-9999"])
    v = verify.premise_check(q, idx)
    assert v is not None and v.reason == "FALSE_PREMISE" and v.hard and not v.ok
    csv = rid_of(idx, CSV, "ORR-1847")
    v = verify.verify(q, ReaderOut("4.3.2", support=[csv]), pkg([csv]), idx)
    assert v.reason == "FALSE_PREMISE" and v.hard


def test_premise_ok_when_present_and_only_in_ineligible(idx):
    assert verify.premise_check(Q4, idx) is None
    mi = ix.build([
        {"path": "a.txt", "kind": "txt", "status": "ok", "records": [{"loc": "L1", "text": "see ZX-4411 here"}],
         "text": "see ZX-4411 here", "images": [], "flags": {"withdrawn": True, "supersedes": [], "revision": None}},
    ])
    q = Q("what about ZX-4411", ids=["ZX-4411"], rare_ids=["ZX-4411"])
    v = verify.premise_check(q, mi)
    assert v.reason == "ONLY_IN_INELIGIBLE" and v.hard
    q.wants_old = True
    assert verify.premise_check(q, mi) is None

