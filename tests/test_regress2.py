"""Regressions from the second adversarial review (F1-F7)."""

from pathlib import Path

import pytest

import cite
import index as ix
import parse
import retrieve
import snap
import textnorm
import verify
from conftest import ROOT
from schema import Package, ReaderOut

S1 = ROOT / "eval" / "suites" / "s1" / "corpus"


def mk(files, withdrawn=()):
    parsed = []
    for path, kind, recs in files:
        parsed.append({"path": path, "kind": kind, "status": "ok",
                       "records": [{"loc": f"L{i}", "text": t} for i, t in enumerate(recs, 1)],
                       "flags": {"withdrawn": True} if path in withdrawn else {},
                       "text": " ".join(recs), "images": []})
    return ix.build(parsed)


def rid(idx, file, needle):
    return next(r.rid for r in idx.records if r.file == file and not r.meta and needle in r.text)


def run(idx, qtext, answer, only_file=None, sub=None):
    q = retrieve.analyze(qtext, idx)
    pkg = retrieve.assemble(idx, q, 1)
    sup = [r for r in pkg.rids if (only_file is None or idx.record(r).file == only_file)
           and (sub is None or sub in idx.record(r).text)]
    v = verify.verify(q, ReaderOut(answer, sup[:1], []), pkg, idx)
    return q, pkg, v


@pytest.fixture(scope="module")
def s1():
    if not S1.exists():
        pytest.skip("eval suite s1 not present")
    return ix.build(parse.walk(S1))


# ---------------------------------------------------------------- F1

def test_f1_relevance_beats_authority():
    idx = mk([("runbooks/fan_runbook.txt", "txt", ["HN-450 fan runbook", "Fan duty floor: 25% at idle"]),
              ("specs/hn450_pinout.pdf", "pdf", ["HN-450 pinout", "Pin 25: GND", "Rail 25 V"])])
    q, pkg, v = run(idx, "What is the fan duty floor of the HN-450, in percent: 25?", "25", "runbooks/fan_runbook.txt")
    assert v.ok and idx.record(v.support_rid).file == "runbooks/fan_runbook.txt"


def test_f1_strict_span_before_loose():
    idx = mk([("a/label.ocr", "txt", ["BOARD REVISIONREV-F7"]), ("b/notes.txt", "txt", ["HN-450 board revision REV-F7"])])
    for r in idx.records:    # mark the first as OCR
        if r.file == "a/label.ocr":
            r.src = "ocr"
    q = retrieve.analyze("What is the board revision of the HN-450?", idx)
    pkg = retrieve.assemble(idx, q, 1)
    v = verify.verify(q, ReaderOut("REV-F7", list(pkg.rids), []), pkg, idx)
    assert v.ok and idx.record(v.support_rid).file == "b/notes.txt"


def test_f1_authority_uses_basename():
    class FE:
        def __init__(self, kind, file):
            self.kind, self.file = kind, file
    assert verify._authority(FE("txt", "specs/notes.txt")) == verify._authority(FE("txt", "misc/notes.txt"))
    assert verify._authority(FE("txt", "x/inspect_log.txt")) == verify._authority(FE("txt", "x/aspect.txt"))
    assert verify._authority(FE("txt", "x/hn_datasheet_r2.txt")) > verify._authority(FE("txt", "x/notes.txt"))
    assert verify._authority(FE("txt", "x/special.txt")) == verify._authority(FE("txt", "x/notes.txt"))
    assert verify._authority(FE("txt", "x/hn_marketing.txt")) < verify._authority(FE("txt", "x/notes.txt"))


def test_f1_t8_datasheet_still_wins(s1):
    q = retrieve.analyze("What is the board power of the HN-450?", s1)
    pkg = retrieve.assemble(s1, q, 1)
    rids = [r for r in pkg.rids if textnorm.find_span(s1.record(r).text, "770")]
    files = {s1.record(r).file for r in rids}
    assert len(files) >= 2
    v = verify.verify(q, ReaderOut("770", [], []), pkg, s1)
    assert v.ok and "datasheet" in s1.record(v.support_rid).file


# ---------------------------------------------------------------- F2

def test_f2_hop_skips_digit_first_ids():
    idx = mk([("a/spec.txt", "txt", ["HN-450 uses connector 12V-2x6 and mounting 400mm"]),
              ("b/other.txt", "txt", ["HN-550 uses connector 12V-2x6 at 400mm"])])
    q = retrieve.analyze("What connector does the HN-450 use?", idx)
    hits = retrieve.hop(idx, q, retrieve.retrieve(idx, q, 1))
    assert all(not h.via for h in hits)


def _chain():
    idx = mk([("logs/prod.log", "log", ["2026-09-02 incident flap TKT-9001 link down"]),
              ("support/bugs.csv", "csv", ["ticket=TKT-9001; fixed_in=4.3.2"])])
    lg, row = rid(idx, "logs/prod.log", "TKT-9001"), rid(idx, "support/bugs.csv", "4.3.2")
    return idx, lg, row, Package(rids=[lg, row], text="", via={row: lg})


def test_f2_bridge_dropped_without_reason():
    idx, lg, row, pkg = _chain()
    q = retrieve.analyze("What is the version for the flap incident?", idx)   # no chain verb, no kind named
    assert verify._bridges(q, ReaderOut("4.3.2", [row], []), pkg, idx, row)[0] == []


def test_f2_bridge_kept_when_listed_or_chain_verb_or_kind():
    idx, lg, row, pkg = _chain()
    q = retrieve.analyze("What is the version for the flap incident?", idx)
    assert verify._bridges(q, ReaderOut("4.3.2", [row], [lg]), pkg, idx, row)[0] == [lg]
    q = retrieve.analyze("Which version fixed the flap incident?", idx)
    assert verify._bridges(q, ReaderOut("4.3.2", [row], []), pkg, idx, row)[0] == [lg]
    q = retrieve.analyze("According to the log, what is the version for the flap incident?", idx)
    assert verify._bridges(q, ReaderOut("4.3.2", [row], []), pkg, idx, row)[0] == [lg]


def test_f2_cite_keeps_bridge_file():
    idx, lg, row, pkg = _chain()
    q = retrieve.analyze("Which version fixed the flap incident?", idx)
    v = verify.verify(q, ReaderOut("4.3.2", [row], []), pkg, idx)
    assert v.ok and cite.cite(q, v, idx, pkg=pkg) == ["logs/prod.log", "support/bugs.csv"]


# ---------------------------------------------------------------- F3

def test_f3_operating_junction_ok(s1):
    for qt in ("What is the maximum operating junction temperature of the HN-450?",
               "What is the design maximum junction temperature of the HN-450?"):
        q, pkg, v = run(s1, qt, "93", "specs/hn450_datasheet_r2.pdf")
        assert v.ok, (qt, v.reason)
    q, pkg, v = run(s1, "What is the board power of the HN-450 (not the idle figure)?", "770",
                    "specs/hn450_datasheet_r2.pdf")
    assert v.ok


def test_f3_canary_attribute_removed_still_refused():
    idx = mk([("specs/nb23_datasheet_r1_WITHDRAWN.pdf", "pdf",
               ["NB-23 datasheet r1", "Maximum storage temperature: 85 C", "Maximum junction temperature: 95 C"]),
              ("specs/nb23_datasheet_r2.pdf", "pdf", ["NB-23 datasheet r2", "Maximum junction temperature: 93 C"])],
             withdrawn=("specs/nb23_datasheet_r1_WITHDRAWN.pdf",))
    q = retrieve.analyze("What is the maximum storage temperature of the NB-23?", idx)
    cur = next(r for r in idx.records if r.file.endswith("r2.pdf") and "junction" in r.text)
    assert verify.removed_in_current(q, idx, cur.rid)
    q2 = retrieve.analyze("What is the maximum junction temperature of the NB-23?", idx)
    assert not verify.removed_in_current(q2, idx, cur.rid)


# ---------------------------------------------------------------- F4

def test_f4_stemmer():
    assert textnorm.stem("values") == "value" == textnorm.stem("value")
    assert textnorm.stem("temperatures") == "temperature"
    assert textnorm.stem("cooling") == "cool"
    assert textnorm.stem("limited") == "limit"
    assert textnorm.stem("switches") == "switch"
    assert textnorm.stem("process") == "process" == textnorm.stem("processes")
    assert textnorm.stem("fans") == "fans"       # < 5 chars after... too short to touch
    assert textnorm.stem("hn450s") == "hn450s"   # not alphabetic
    assert "cool" in textnorm.tokens("cooling") and "hn-450" in textnorm.tokens("HN-450")
    assert textnorm.norm("Cooling-1") == "COOLING1" and textnorm.ids("ticket ORR-1847s")[0] == "ORR-1847s"


def test_f4_bm25_top_and_type_hint():
    import config
    assert config.BM25_TOP_RECORDS == 150
    assert "py" not in retrieve.analyze("What is the error code for the fabric flap?").type_hint
    assert "py" not in retrieve.analyze("Which code appears on the label?").type_hint
    assert "py" in retrieve.analyze("What is the default timeout in the service source code?").type_hint
    assert "py" in retrieve.analyze("What constant is set in ingest.py?").type_hint


def test_f4_query_aliases():
    q = retrieve.analyze("What is the production batch size?")
    toks = retrieve._query_tokens(q)
    assert "prod" in toks and "production" in toks
    assert "specification" in retrieve._query_tokens(retrieve.analyze("What does the spec say?"))


# ---------------------------------------------------------------- F5

def test_f5_glued_units_and_loose_prefix():
    assert snap.find("96", "die 0 at 96C, clocks 60%") == "96"
    assert snap.find("770", "TBP 770W max") == "770"
    assert snap.find("12", "rail 12V") == "12"
    assert snap.find("40", "latency 40ms") == "40"
    assert snap.find("2.4", "bandwidth 2.4TB/s") == "2.4"
    assert snap.find("12", "connector 12V-2x6") is None
    assert snap.find("5", "severity=S5") is None
    assert snap.find("REV-F7", "BOARD REVISIONREV-F7", loose=True) == "REV-F7"
    assert snap.find("REV-F7", "BOARD REVISIONREV-F7") is None
    assert snap.find("25", "J25 J26", loose=True) is None


# ---------------------------------------------------------------- F6

@pytest.mark.parametrize("raw,expected", [
    ("TQ-40", True), ("ORR-1847", True), ("HLX-2041", True), ("KX-72", True), ("E7731", True),
    ("ORR-FAN-2214-B", True),
    ("x16", False), ("RFC2119", False), ("JESD51-2", False), ("IEEE802.3", False), ("21-Sep", False),
    ("PCIe", False), ("HBM3e", False), ("96C", False), ("roadmap_fy27.docx", False), ("REV-F7", False),
])
def test_f6_premise_worthy(raw, expected):
    # the index-presence part is exercised elsewhere; only the shape rule is asserted here
    assert retrieve._premise_worthy(raw) is expected


def test_f6_non_entity_ids_do_not_refuse():
    idx = mk([("a/x.txt", "txt", ["HN-450 supports PCIe x16 per RFC2119"])])
    q = retrieve.analyze("Does the HN-450 use JESD51-2 and x16?", idx)
    assert q.rare_ids == [] or all(i == "HN-450" for i in q.rare_ids)


# ---------------------------------------------------------------- F7

def test_f7a_word_numbers_need_aggregation():
    assert retrieve.analyze("What is the combined power of four boards?").numbers == ["4"]
    assert retrieve.analyze("What is the idle power of the four-way HN board in four slots?").numbers == []
    assert retrieve.analyze("What is the budget in thousand USD?").numbers == []
    assert retrieve.analyze("What is the power per board of the twelve fans?").numbers == ["12"]


def test_f7b_query_tokens_keep_filename_parts_and_standalone():
    q = retrieve.analyze("What does roadmap_fy27.docx say about launch?")
    toks = retrieve._query_tokens(q)
    assert "roadmap" in toks and "fy27" in toks
    q = retrieve.analyze("Which HN boards are newer than the HN-450?")
    assert "hn" in retrieve._query_tokens(q)
    q = retrieve.analyze("What is the power of the HN-450?")
    assert "hn" not in retrieve._query_tokens(q)


def test_f7c_unicode_dashes():
    for d in ("‐", "‑", "–", "—", "−"):
        assert textnorm.ids(f"board HN{d}450 here") == ["HN-450"]
        assert textnorm.tokens(f"HN{d}450") == textnorm.tokens("HN-450")
        assert textnorm.find_span(f"board HN{d}450 here", "HN-450") == (6, 12)
        assert textnorm.find_span("board HN-450 here", f"HN{d}450") == (6, 12)
        assert textnorm.find_span_loose(f"lbl SNOS4{d}118823", "OS4-118823") is not None
    assert textnorm.norm("HN‑450") != textnorm.norm("HN-450")   # norm stays the official rule


def test_f7d_entity_cache_lives_on_index():
    idx = mk([("a/x.txt", "txt", ["HN-450 board"])])
    assert not hasattr(verify, "_FILE_ENTITY_CACHE")
    assert verify._entity_in_file(idx, "a/x.txt", "HN-450")
    assert ("a/x.txt", "HN-450") in idx._ent_cache
    idx2 = mk([("a/x.txt", "txt", ["HN-550 board"])])
    assert not verify._entity_in_file(idx2, "a/x.txt", "HN-450")


def test_f7e_second_look_threshold():
    import inspect
    import pipeline
    assert "cov2 >= cov + 0.15" in inspect.getsource(pipeline)
