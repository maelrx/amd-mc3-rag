from pathlib import Path

import pytest

import config
import index as ix
import parse
import textnorm
from conftest import SAMPLE_CORPUS
from schema import Record


@pytest.fixture(scope="module")
def idx():
    return ix.build(parse.walk(SAMPLE_CORPUS))


def _pf(path, text="", revision=None, withdrawn=False, status="ok"):
    """Hand-made parse.walk entry."""
    return {"path": path, "kind": Path(path).suffix.lstrip("."), "status": status,
            "records": [{"loc": "p1", "text": text, "stale": False}] if text else [], "text": text,
            "images": [], "flags": {"withdrawn": withdrawn, "supersedes": [], "revision": revision}}


def test_sample_families(idx):
    r1 = idx.file_by_path["specs/tq40_datasheet_r1_WITHDRAWN.pdf"]
    r2 = idx.file_by_path["specs/tq40_datasheet_r2.pdf"]
    assert r1.family == r2.family == "specs/tq40_datasheet"
    assert r1.withdrawn and not r1.eligible
    assert r2.eligible and r2.superseded_by is None
    assert not idx.file_by_path["vendor/supplier_agreement_ENCRYPTED.pdf"].eligible
    assert idx.file_by_path["support/bug_database.csv"].eligible


def test_sample_record_layout(idx):
    assert [r.rid for r in idx.records] == list(range(len(idx.records)))
    for f in idx.files:
        first = idx.records[f.rid_range[0]]
        assert first.meta and first.loc == "name" and first.src == "name" and first.file == f.file
        assert idx.by_file[f.file][0] == f.rid_range[0]
    assert idx.records[idx.file_by_path["support/asset_label.jpg"].rid_range[0]].text == "support asset label jpg"
    assert idx.file_by_path["specs/backplane_pinout.png"].images == [["file", "specs/backplane_pinout.png"]]


def test_sample_id_index(idx):
    e = idx.ids["ORR1847"]
    assert e["df"] == 2
    assert set(e["files"]) == {"logs/prod_inference_2026-09-02.log", "support/bug_database.csv"}
    assert "ORR-1847" in e["raw"]
    assert idx.has_id("orr-1847") and not idx.has_id("ZZZ-9999")
    assert idx.files_with_id("ORR-1847") == e["files"]
    assert "432" in idx.ids  # 4.3.2 as dotted version
    assert "Q3" not in idx.ids and "2026" not in idx.ids


def test_rare_ids_in(idx):
    assert idx.rare_ids_in("Which firmware version fixed ticket ORR-1847?") == ["ORR-1847"]
    assert idx.rare_ids_in("Which firmware version fixed ticket ZZZ-9999?") == []


def test_bm25_ranks_csv_row_first(idx):
    hits = idx.bm25.search(textnorm.tokens("ORR-1847 fixed_in"), 5)
    top = idx.record(hits[0][0])
    assert top.file == "support/bug_database.csv" and top.loc == "row2"
    assert hits == sorted(hits, key=lambda h: -h[1])


def test_n_products(idx):
    assert idx.file_by_path["support/bug_database.csv"].n_products == 8  # ORR-#### tickets match the product regex
    assert idx.file_by_path["specs/tq40_datasheet_r2.pdf"].n_products >= 1
    assert idx.file_by_path["specs/backplane_pinout.png"].n_products == 0
    assert ix.build([_pf("a.txt", "TQ-40 and tq40 and TQ-20")]).files[0].n_products == 2


def test_eligible_and_context(idx):
    csv = idx.file_by_path["support/bug_database.csv"]
    row = csv.rid_range[0] + 3  # name, header, row2, row3
    assert idx.eligible(row)
    assert not idx.eligible(idx.file_by_path["specs/tq40_datasheet_r1_WITHDRAWN.pdf"].rid_range[0] + 1)
    ctx = idx.context_rids(csv.rid_range[0] + 5)
    assert ctx[0] == csv.rid_range[0]
    assert csv.rid_range[0] + 1 in ctx  # csv header record
    assert idx.file_of(row) is csv


def test_synthetic_family():
    parsed = [_pf("report_v1.docx", "alpha", "1"), _pf("report_v2.docx", "beta", "2"),
              _pf("notes.txt", "gamma")]
    i = ix.build(parsed)
    v1, v2, notes = (i.file_by_path[p] for p in ("report_v1.docx", "report_v2.docx", "notes.txt"))
    assert v1.family == v2.family == "/report" and notes.family == "/notes"
    assert v1.superseded_by == "report_v2.docx" and not v1.eligible
    assert v2.superseded_by is None and v2.eligible and notes.eligible


def test_family_revision_ordering_and_withdrawn():
    parsed = [_pf("d/spec_r2.pdf", "a", "2"), _pf("d/spec_r10.pdf", "b", "10"),
              _pf("d/spec_r11_WITHDRAWN.pdf", "c", "11", withdrawn=True), _pf("d/spec_r2b.pdf", "d", "2b")]
    i = ix.build(parsed)
    assert i.file_by_path["d/spec_r10.pdf"].eligible
    assert i.file_by_path["d/spec_r2.pdf"].superseded_by == "d/spec_r10.pdf"
    assert i.file_by_path["d/spec_r2b.pdf"].superseded_by == "d/spec_r10.pdf"
    assert i.file_by_path["d/spec_r11_WITHDRAWN.pdf"].superseded_by is None
    assert not i.file_by_path["d/spec_r11_WITHDRAWN.pdf"].eligible


def test_unrelated_stems_never_form_a_family():
    parsed = [_pf("a/alpha_r1.pdf", "x", "1"), _pf("a/beta_r2.pdf", "y", "2"),
              _pf("a/gamma.pdf", "z"), _pf("b/alpha_r2.pdf", "w", "2")]
    i = ix.build(parsed)
    assert len({f.family for f in i.files}) == 4
    assert all(f.superseded_by is None and f.eligible for f in i.files)


def test_no_revision_token_never_superseded():
    i = ix.build([_pf("x/plan.docx", "a"), _pf("x/plan_final.docx", "b"), _pf("x/plan_v3.docx", "c", "3")])
    assert i.file_by_path["x/plan.docx"].eligible and i.file_by_path["x/plan_final.docx"].eligible


def test_ocr_records():
    parsed = [_pf("img/label.jpg")]
    i = ix.build(parsed, {"img/label.jpg": {"lines": ["SN OS4-118823-77", "", "REV-C2"], "pairs": ["A1: CLK", "B14: RST#"]}})
    recs = i.records
    assert [(r.loc, r.src) for r in recs] == [("name", "name"), ("ocr.L1", "ocr"), ("ocr.L3", "ocr"), ("ocr.pairs", "ocr")]
    assert recs[3].text == "A1: CLK; B14: RST#"
    assert i.has_id("OS4-118823-77") and i.has_id("RST#")
    assert i.file_by_path["img/label.jpg"].rid_range == [0, 4]


def test_save_load_roundtrip(idx, tmp_path):
    ix.save(tmp_path, idx)
    assert not list(tmp_path.glob("*.tmp"))
    again = ix.load(tmp_path)
    assert again.records == idx.records and again.files == idx.files and again.ids == idx.ids
    assert again.bm25.search(["orr-1847"], 3) == idx.bm25.search(["orr-1847"], 3)
    (tmp_path / "ids.json").unlink()
    assert ix.load(tmp_path).ids == idx.ids


def test_append_records(idx, tmp_path):
    ix.save(tmp_path, idx)
    n = len(idx.records)
    new = [Record(rid=999, file="specs/backplane_pinout.png", loc="vlm.L1", text="pin B14 = THERM_ALERT#", src="vlm"),
           Record(rid=999, file="specs/backplane_pinout.png", loc="vlm.L2", text="ticket ORR-1847", src="vlm")]
    assert ix.append_records(tmp_path, new) == [n, n + 1]
    assert ix.append_records(tmp_path, []) == []
    again = ix.load(tmp_path)
    assert len(again.records) == n + 2 and again.records[n].rid == n and again.records[n + 1].loc == "vlm.L2"
    assert again.ids["ORR1847"]["df"] == 3
    assert again.has_id("THERM_ALERT#")
    assert again.by_file["specs/backplane_pinout.png"][-2:] == [n, n + 1]
    assert again.bm25.search(textnorm.tokens("therm_alert#"), 1)[0][0] == n


def test_bm25_basics():
    b = ix.BM25([["a", "b"], ["b", "b", "c"], ["c"]])
    assert b.search(["zzz"], 3) == []
    top = b.search(["b", "b"], 3)
    assert [d for d, _ in top] == [1, 0]
    assert config.RARE_DF >= 1
