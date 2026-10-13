"""vision.py: RapidOCR on the sample images, caching, deadline handling and geometry helpers."""

import io
import json
import re
import time
import zipfile

import pytest

import parse
import vision
from conftest import SAMPLE_CORPUS

pytest.importorskip("rapidocr")
pytest.importorskip("onnxruntime")
from PIL import Image  # noqa: E402

PINOUT = "specs/backplane_pinout.png"
LABEL = "support/asset_label.jpg"


def n(s: str) -> str:
    """Compare form: upper-case, no spaces/underscores/dashes."""
    return re.sub(r"[\s_\-]", "", s).upper()


@pytest.fixture(scope="module")
def pinout(tmp_path_factory) -> dict:
    t = time.time()
    res = vision.ocr_file(SAMPLE_CORPUS, PINOUT, tmp_path_factory.mktemp("c"))
    print(f"\nOCR {PINOUT}: {time.time() - t:.2f}s (incl. engine load)")
    return res


def test_pinout_tokens_and_pair(pinout):
    texts = [n(ln["text"]) for ln in pinout["lines"]]
    assert "B14" in texts and "THERMALERT#" in texts
    assert any(n(p) == "B14:THERMALERT#" for p in pinout["pairs"]), pinout["pairs"]
    assert pinout["rot"] == 0
    assert any(n("B14") in n(row) and n("THERM_ALERT#") not in n(row) for row in pinout["layout"])


def test_label_tokens(tmp_path):
    t = time.time()
    res = vision.ocr_file(SAMPLE_CORPUS, LABEL, tmp_path)
    print(f"\nOCR {LABEL}: {time.time() - t:.2f}s")
    assert any("REVC2" in n(ln["text"]) for ln in res["lines"])
    assert any("REVC2" in n(row) for row in res["layout"])


def test_rotation_retry_keeps_best_mean_conf(tmp_path, monkeypatch):
    """RapidOCR's angle classifier already handles flipped pages, so drive the retry with a fake pass."""
    def fake_run(_engine, img):
        if img.width > img.height:           # upright landscape input: poor read
            return [_line("junk", 0, 0, 40, 20) | {"conf": 0.3}]
        return [_line("REV-C2", 0, 0, 40, 20), _line("MODEL", 0, 30, 40, 50)]
    monkeypatch.setattr(vision, "_run", fake_run)
    monkeypatch.setattr(vision, "get_ocr", lambda: object())
    res = vision.ocr_image(Image.new("RGB", (300, 100), "white"), "r", tmp_path)
    assert res["rot"] == 90 and [ln["text"] for ln in res["lines"]] == ["REV-C2", "MODEL"]


def test_upright_good_read_skips_rotations(monkeypatch):
    calls = []

    def fake_run(_engine, img):
        calls.append(img.size)
        return [_line("A1", 0, 0, 40, 20), _line("B2", 0, 30, 40, 50)]
    monkeypatch.setattr(vision, "_run", fake_run)
    monkeypatch.setattr(vision, "get_ocr", lambda: object())
    assert vision.ocr_image(Image.new("RGB", (300, 100), "white"), "u", None)["rot"] == 0
    assert len(calls) == 1


def test_large_image_is_downscaled(monkeypatch):
    seen = []
    monkeypatch.setattr(vision, "_run", lambda _e, img: seen.append(img.size) or [])
    monkeypatch.setattr(vision, "get_ocr", lambda: object())
    vision.ocr_image(Image.new("RGB", (4000, 1000), "white"), "big", None)
    assert seen[0] == (2000, 500)


def test_cache_roundtrip_skips_engine(tmp_path, monkeypatch):
    img = Image.open(SAMPLE_CORPUS / LABEL)
    first = vision.ocr_image(img, "k1", tmp_path)
    assert json.loads((tmp_path / "k1.ocr.json").read_text()) == first
    monkeypatch.setattr(vision, "get_ocr", lambda: pytest.fail("engine used despite cache"))
    assert vision.ocr_image(img, "k1", tmp_path) == first


def test_ocr_file_cache_key_is_sha1(tmp_path, monkeypatch):
    first = vision.ocr_file(SAMPLE_CORPUS, LABEL, tmp_path)
    names = [p.name for p in tmp_path.iterdir()]
    assert len(names) == 1 and re.fullmatch(r"[0-9a-f]{40}\.ocr\.json", names[0])
    monkeypatch.setattr(vision, "get_ocr", lambda: pytest.fail("engine used despite cache"))
    assert vision.ocr_file(SAMPLE_CORPUS, LABEL, tmp_path) == first


def test_ocr_all_sample_corpus(tmp_path):
    parsed = parse.walk(SAMPLE_CORPUS)
    t = time.time()
    out = vision.ocr_all(parsed, SAMPLE_CORPUS, tmp_path, time.time() + 120)
    print(f"\nocr_all sample corpus (cold, 4 threads): {time.time() - t:.2f}s")
    assert set(out) == {PINOUT, LABEL}
    assert out[PINOUT]["images"] == 1 and out[PINOUT]["missing"] == 0
    assert any(n(p) == "B14:THERMALERT#" for p in out[PINOUT]["pairs"])
    assert any("REVC2" in n(r) for r in out[LABEL]["lines"])
    assert not any(r.startswith("[img") for r in out[LABEL]["lines"])  # single image: no prefix


def test_ocr_all_expired_deadline_returns_fast(tmp_path):
    parsed = parse.walk(SAMPLE_CORPUS)
    t = time.time()
    out = vision.ocr_all(parsed, SAMPLE_CORPUS, tmp_path, time.time())
    assert time.time() - t < 2
    assert all(e["images"] == 0 and e["missing"] == 1 and e["lines"] == [] for e in out.values())


def test_ocr_all_docx_media_prefix_and_tiny_skip(tmp_path):
    from PIL import ImageDraw
    big = Image.new("RGB", (400, 120), "white")
    ImageDraw.Draw(big).text((20, 40), "SERIAL ABC-123", fill="black")
    docx = tmp_path / "d.docx"
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml", "<x/>")
        for name, im in (("image1.png", big), ("image2.png", Image.new("RGB", (16, 16), "white"))):
            buf = io.BytesIO()
            im.save(buf, "PNG")
            z.writestr(f"word/media/{name}", buf.getvalue())
    parsed = [{"path": "d.docx", "images": [("zip", "word/media/image1.png"), ("zip", "word/media/image2.png")]}]
    out = vision.ocr_all(parsed, tmp_path, tmp_path / "cache", time.time() + 60)["d.docx"]
    assert out["images"] == 1 and out["missing"] == 0     # tiny image skipped, not "missing"
    assert all(r.startswith("[img word/media/image1.png] ") for r in out["lines"])


def test_ocr_all_unreadable_source_counts_missing(tmp_path):
    parsed = [{"path": "gone.png", "images": [("file", "gone.png")]}]
    out = vision.ocr_all(parsed, tmp_path, tmp_path / "c", time.time() + 10)
    assert out["gone.png"]["missing"] == 1 and out["gone.png"]["images"] == 0


def test_render_pdf_page_without_pypdfium2(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "pypdfium2", None)
    assert vision.render_pdf_page(SAMPLE_CORPUS / "specs/tq40_datasheet_r2.pdf", 1) is None


def test_render_and_ocr_pdf_page(tmp_path):
    pytest.importorskip("pypdfium2")
    img = vision.render_pdf_page(SAMPLE_CORPUS / "specs/tq40_datasheet_r2.pdf", 1)
    assert img is not None and min(img.size) > 64
    res = vision.ocr_pdf_page(SAMPLE_CORPUS, "specs/tq40_datasheet_r2.pdf", 1, tmp_path)
    assert any("94" in ln["text"] for ln in res["lines"])
    assert any(p.name.endswith("-p1.ocr.json") for p in tmp_path.iterdir())


def _line(text, x0, y0, x1, y1):
    return {"box": [x0, y0, x1, y1], "text": text, "conf": 1.0}


def test_pairs_geometry_pure():
    lines = [_line("B14", 100, 100, 140, 124), _line("THERM_ALERT#", 60, 160, 180, 184),
             _line("B15", 300, 100, 340, 124), _line("GND", 305, 160, 335, 184),
             _line("REV:", 10, 300, 50, 324), _line("C2", 60, 300, 90, 324),
             _line("SN", 10, 350, 40, 374), _line("OS4-1", 60, 350, 120, 374),
             _line("a long sentence of prose that is not a label at all", 10, 400, 400, 424),
             _line("and another sentence of prose directly below it, too", 10, 430, 400, 454)]
    got = vision._pairs(lines, vision._rows(lines))
    assert "B14: THERM_ALERT#" in got and "B15: GND" in got
    assert "REV: C2" in got and "SN: OS4-1" in got
    assert not any("prose" in p for p in got)


def test_rows_cluster_and_order():
    lines = [_line("right", 200, 102, 260, 126), _line("left", 10, 100, 60, 124),
             _line("below", 10, 200, 60, 224)]
    assert ["  ".join(x["text"] for x in r) for r in vision._rows(lines)] == ["left  right", "below"]


def test_get_ocr_is_singleton_and_none_when_unavailable(monkeypatch):
    assert vision.get_ocr() is vision.get_ocr()
    monkeypatch.setattr(vision, "_engine", None)
    monkeypatch.setattr(vision, "_engine_tried", True)
    assert vision.get_ocr() is None
    res = vision.ocr_image(Image.new("RGB", (100, 100), "white"), "x", None)
    assert res["lines"] == [] and res["pairs"] == []
