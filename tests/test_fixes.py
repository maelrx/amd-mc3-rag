"""Regression tests for fixes found on the first GPU run."""

import json
from pathlib import Path

import index as index_mod
import parse
import retrieve
import verify

ROOT = Path(__file__).resolve().parents[1]


def _idx():
    return index_mod.build(parse.walk(ROOT / "sample-corpus"))


def test_unit_with_degrees_word():
    q = retrieve.analyze("What is the maximum junction temperature of the NB-23 in degrees Fahrenheit?")
    assert q.unit == "F"
    q = retrieve.analyze("What is the default batch timeout, in seconds, in the ingest service?")
    assert q.unit == "s"
    q = retrieve.analyze("What is the storage temperature in degrees Celsius?")
    assert q.unit == "C"


def test_spelled_out_count_is_a_condition():
    q = retrieve.analyze("What is the combined board power of four HN-450 boards?")
    assert "4" in q.numbers
    q = retrieve.analyze("What is the part number of the four-pin fan connector?")
    assert "4" not in q.numbers


def test_authority_prefers_specs_over_prose():
    from schema import FileEntry
    pdf = FileEntry(file="specs/hn450_datasheet_r2.pdf", kind="pdf", status="ok")
    txt = FileEntry(file="marketing/hn450_product_brief.txt", kind="txt", status="ok")
    assert verify._authority(pdf) > verify._authority(txt)


def test_coverage_sample_questions_high_for_gold_records():
    idx = _idx()
    gold = json.loads((ROOT / "sample-questions.json").read_text())["queries"]
    low = []
    for g in gold:
        if not g["expected_answer"]:
            continue
        q = retrieve.analyze(g["query"], idx)
        f = g["expected_citations"][-1]
        rids = [r.rid for r in idx.records if r.file == f and not r.meta
                and verify.snap.find(g["expected_answer"], r.text) is not None]
        if not rids:
            continue  # image answers need OCR records
        bridges = [r.rid for r in idx.records if r.file in g["expected_citations"][:-1] and "ORR-1847" in r.text]
        cov = max(verify.coverage(q, idx, rid, bridges) for rid in rids)
        if cov < 0.4:
            low.append((g["n"], round(cov, 2)))
    assert low == [], low
