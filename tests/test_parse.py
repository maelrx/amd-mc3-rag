"""parse.py against the sample corpus, the edge fixtures and synthetic inputs."""

import zipfile
from pathlib import Path

import openpyxl
import pytest

import parse
from conftest import ROOT, SAMPLE_CORPUS

EDGE = ROOT / "eval" / "edge"


@pytest.fixture(scope="module")
def sample() -> dict:
    return {r["path"]: r for r in parse.walk(SAMPLE_CORPUS)}


@pytest.fixture(scope="module")
def edge() -> dict:
    return {r["path"]: r for r in parse.walk(EDGE)}


def _texts(rec: dict) -> list[str]:
    return [r["text"] for r in rec["records"]]


def test_walk_lists_13_files_and_ignores_empty_dir(sample):
    assert len(sample) == 13
    assert not any(p.startswith("archive/") for p in sample)
    assert list(sample) == sorted(sample, key=lambda p: (Path(p).parent.as_posix(), Path(p).name))


def test_statuses(sample):
    enc = sample["vendor/supplier_agreement_ENCRYPTED.pdf"]
    assert enc["status"] == "encrypted" and enc["records"] == []
    assert sample["vendor/telemetry_capture.dat"]["status"] == "binary"
    assert sample["vendor/internal_audit.txt"]["status"] in ("unreadable", "ok")


def test_revision_flags(sample):
    r1 = sample["specs/tq40_datasheet_r1_WITHDRAWN.pdf"]["flags"]
    r2 = sample["specs/tq40_datasheet_r2.pdf"]["flags"]
    assert r1["withdrawn"] is True
    assert r2["withdrawn"] is False and r2["supersedes"] == ["1"]


def test_withdrawn_re_underscore_and_word_boundaries():
    assert parse.WITHDRAWN_RE.search("tq40_datasheet_r1_WITHDRAWN")
    assert parse.WITHDRAWN_RE.search("Do not use")
    assert not parse.WITHDRAWN_RE.search("avoid")
    assert not parse.WITHDRAWN_RE.search("devoid")


def test_withdrawn_by_filename_even_with_blank_title():
    flags = parse._flags("specs/tq40_datasheet_r1_WITHDRAWN.pdf", "\n\n")
    assert flags["withdrawn"] is True and flags["revision"] == "1"


def test_python_const_default_and_comment_records(sample):
    recs = sample["engineering/ingest_service.py"]["records"]
    assert any(r["loc"].endswith((".const", ".default")) and "180" in r["text"] for r in recs)
    assert any(r["loc"].endswith(".comment") and "60" in r["text"] for r in recs)
    assert not any(r["loc"].endswith(".comment") and not r["text"].startswith("#") for r in recs)


def test_csv_xlsx_docx_pdf_content(sample):
    csv_rows = {r["loc"]: r["text"] for r in sample["support/bug_database.csv"]["records"]}
    assert "fixed_in=4.3.2" in csv_rows["row2"]
    assert any("ORR-FAN-2214-B" in t for t in _texts(sample["support/rma_parts.xlsx"]))
    # the sample docx has no table: the fact sits in a paragraph (table rows: see test_edge_fixtures)
    assert any("Q3 FY27" in t for t in _texts(sample["planning/roadmap_fy27.docx"]))
    assert any("94" in t for t in _texts(sample["specs/tq40_datasheet_r2.pdf"]))


def test_xlsx_legacy_row_is_stale(sample):
    recs = sample["support/rma_parts.xlsx"]["records"]
    legacy = [r for r in recs if "ORR-FAN-1990-A" in r["text"]]
    assert legacy and all(r["stale"] for r in legacy)
    assert not any(r["stale"] for r in recs if "ORR-FAN-2214-B" in r["text"])


def test_images_lists(sample):
    assert sample["specs/backplane_pinout.png"]["images"] == [("file", "specs/backplane_pinout.png")]
    assert sample["support/asset_label.jpg"]["images"] == [("file", "support/asset_label.jpg")]
    for rec in sample.values():
        for kind, ref in rec["images"]:
            if kind == "pdfpage":
                assert isinstance(ref, int) and ref >= 1


def test_edge_fixtures(edge):
    xl = {r["loc"]: r for r in edge["edge.xlsx"]["records"]}
    assert "columns:" in xl["Costs!R3"]["text"]                 # merged title is not the header
    assert "total_usd=845" in xl["Costs!R4"]["text"]            # formula evaluated
    assert "unit_usd=1,234.50" in xl["Costs!R5"]["text"]        # number_format honoured
    assert xl["Archive!R2"]["stale"] is True                    # hidden sheet
    assert any("Max fan speed is 4200 RPM" in t for t in _texts(edge["misc/notes_utf16.txt"]))
    assert "fixed_in=5.0.1" in edge["misc/tickets_semicolon.csv"]["records"][1]["text"]
    assert edge["misc/report.bin"]["kind"] == "docx"
    assert any("table" in r["loc"] and "Q3 FY27" in r["text"] for r in edge["misc/report.bin"]["records"])
    assert any("table" in r["loc"] and "Q3 FY27" in r["text"] for r in edge["edge.docx"]["records"])
    assert any("ORION-7" in t for t in _texts(edge["misc/UPPER.TXT"]))


def test_stale_mark_re_on_synthetic_records(tmp_path):
    (tmp_path / "parts.csv").write_text(
        "part,description\nORR-FAN-1990-A,Fan assembly (legacy)\nORR-FAN-2214-B,Fan assembly\n")
    (tmp_path / "notes.txt").write_text("Board X status: obsolete\nBoard Y status = active\n")
    wb = openpyxl.Workbook()
    wb.active.append(["part", "description"])
    wb.active.append(["ORR-FAN-1990-A (legacy)", "Single rotor"])
    wb.save(tmp_path / "p.xlsx")
    found = {r["path"]: r for r in parse.walk(tmp_path)}
    csv_recs = {r["text"]: r["stale"] for r in found["parts.csv"]["records"]}
    assert csv_recs["part=ORR-FAN-1990-A; description=Fan assembly (legacy)"] is True
    assert csv_recs["part=ORR-FAN-2214-B; description=Fan assembly"] is False
    txt = {r["text"]: r["stale"] for r in found["notes.txt"]["records"]}
    assert txt == {"Board X status: obsolete": True, "Board Y status = active": False}
    assert any(r["stale"] and "ORR-FAN-1990-A (legacy)" in r["text"] for r in found["p.xlsx"]["records"])


def test_supersedes_does_not_eat_the_r_of_a_word():
    assert parse.SUPERSEDES_RE.findall("supersedes reference X") == ["reference"]
    assert parse.SUPERSEDES_RE.findall("supersedes r1") == ["1"]
    assert parse.SUPERSEDES_RE.findall("supersedes revision 3") == ["3"]
    assert parse.SUPERSEDES_RE.findall("Doc rev C - supersedes rev B") == ["B"]


def _blank_pdf(path: Path, pages: int) -> None:
    from pypdf import PdfWriter
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(200, 200)
    w.write(path)


def test_pdf_blank_pages_go_to_vision_one_based(tmp_path):
    _blank_pdf(tmp_path / "scan.pdf", 3)
    rec = parse.extract(tmp_path, tmp_path / "scan.pdf")
    assert rec["status"] == "ok" and rec["records"] == []
    assert rec["images"] == [("pdfpage", 1), ("pdfpage", 2), ("pdfpage", 3)]


def test_pdf_page_cap(tmp_path):
    _blank_pdf(tmp_path / "long.pdf", parse.PDF_VISION_PAGES + 5)
    rec = parse.extract(tmp_path, tmp_path / "long.pdf")
    assert rec["images"][-1] == ("pdfpage", parse.PDF_VISION_PAGES)   # vision candidates: the first 60 pages only


def test_corrupt_pdf_never_raises(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4\ngarbage")
    assert parse.extract(tmp_path, tmp_path / "bad.pdf")["status"] in ("error", "encrypted", "ok")


def test_docx_media_listed_as_zip_members(tmp_path):
    p = tmp_path / "d.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml",
                   '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                   "<w:body><w:p><w:r><w:t>hi</w:t></w:r></w:p></w:body></w:document>")
        z.writestr("word/media/image1.png", b"x")
    assert parse.extract(tmp_path, p)["images"] == [("zip", "word/media/image1.png")]
