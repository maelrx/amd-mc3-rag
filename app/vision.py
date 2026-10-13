"""OCR side of the image pipeline (RapidOCR on CPU). The VLM transcription lives in reader.py.

Every image (standalone file, docx/pptx media, weak PDF page) goes through ``ocr_image``,
which returns reading-order rows (``layout``) plus ``PIN: SIGNAL`` style ``pairs`` built from
box geometry, and caches the result by content hash. All third-party imports are optional.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import statistics
import sys
import threading
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import config

MAX_SIDE = 2000          # longest side fed to the detector
MIN_SIDE = 64            # smaller images carry no text worth reading
ROW_TOL = 0.6            # row clustering tolerance, in median line heights
PAIR_DX = 0.5            # max |dx centre| for a vertical pair, in max line widths
PAIR_DY = 3.0            # max dy centre for a vertical pair, in median line heights
FREE_PAIR_MAX_CHARS = 40  # code-less vertical pairs only between short texts (no prose)
CODE_RE = re.compile(r"^[A-Z]{1,3}\d{1,3}$|^\d{1,3}$")
LABEL_RE = re.compile(r"^[A-Z]{2,8}$")

# OCR glues a printed label to the ID after it ("BOARD REVISIONREV-A4", "SERIALCS-836046-T"). All-caps runs have no
# case boundary to cut at, so they are split after a known label word; mixed case splits at the case change.
_LABEL_WORDS = ("REVISION", "FIRMWARE", "HARDWARE", "ASSEMBLY", "VERSION", "SERIES", "NUMBER", "SERIAL", "MODEL",
                "BOARD", "BATCH", "LABEL", "PART", "DATE", "CODE", "TYPE", "ASSY")
_GLUE_LABEL_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(_LABEL_WORDS) + r")(?=[A-Z]{2,5}-(?:\d|[A-Z]\d))")
_GLUE_ID_RE = re.compile(r"(?<=[a-z])(?=[A-Z]{1,5}-\d)")      # SerialCS-836046
_GLUE_CASE_RE = re.compile(r"(?<=[a-z]{2})(?=[A-Z]{2,})")       # FirmwareFW2 (not dBA, eMMC)


def split_glue(text: str) -> str:
    """Insert the space OCR dropped between a label and the ID glued to it. Idempotent."""
    text = _GLUE_LABEL_RE.sub(r"\1 ", str(text))
    return _GLUE_CASE_RE.sub(" ", _GLUE_ID_RE.sub(" ", text))


_engine = None
_engine_tried = False
_engine_lock = threading.Lock()


def get_ocr():
    """Lazily built RapidOCR engine, or None (logged once) if it cannot be imported."""
    global _engine, _engine_tried
    with _engine_lock:
        if _engine_tried:
            return _engine
        _engine_tried = True
        try:
            from rapidocr import RapidOCR
            try:
                _engine = RapidOCR(params={"EngineConfig.onnxruntime.intra_op_num_threads": 4,
                                           "Global.log_level": "warning"})
            except Exception:  # noqa: BLE001 - params layout differs across rapidocr versions
                _engine = RapidOCR()
        except Exception as e:  # noqa: BLE001 - ImportError, missing onnxruntime, missing models
            print(f"vision: RapidOCR unavailable ({type(e).__name__}: {e})", file=sys.stderr)
            _engine = None
        return _engine


# ---------------------------------------------------------------- geometry helpers

def _rows(lines: list[dict]) -> list[list[dict]]:
    """Cluster lines into rows by y centre; rows top-to-bottom, lines left-to-right."""
    if not lines:
        return []
    mh = statistics.median(max(1.0, ln["box"][3] - ln["box"][1]) for ln in lines)
    rows: list[list[dict]] = []
    centres: list[float] = []
    for ln in sorted(lines, key=lambda x: x["box"][1] + x["box"][3]):
        cy = (ln["box"][1] + ln["box"][3]) / 2
        if rows and abs(cy - centres[-1]) <= ROW_TOL * mh:
            rows[-1].append(ln)
            centres[-1] = sum((r["box"][1] + r["box"][3]) / 2 for r in rows[-1]) / len(rows[-1])
        else:
            rows.append([ln])
            centres.append(cy)
    return [sorted(r, key=lambda x: x["box"][0]) for r in rows]


def _pairs(lines: list[dict], rows: list[list[dict]]) -> list[str]:
    """``KEY: value`` strings from vertical neighbours (pin over signal) and row labels."""
    out: list[str] = []
    if len(lines) < 2:
        return out
    mh = statistics.median(max(1.0, ln["box"][3] - ln["box"][1]) for ln in lines)
    info = [(ln, (ln["box"][0] + ln["box"][2]) / 2, (ln["box"][1] + ln["box"][3]) / 2,
             ln["box"][2] - ln["box"][0]) for ln in lines]
    for a, ax, ay, aw in info:
        best = None
        for b, bx, by, bw in info:
            dy = by - ay
            if 0 < dy <= PAIR_DY * mh and abs(bx - ax) <= PAIR_DX * max(aw, bw):
                if best is None or (dy, abs(bx - ax)) < best[0]:
                    best = ((dy, abs(bx - ax)), b)
        if best is None:
            continue
        ta, tb = a["text"].strip(), best[1]["text"].strip()
        a_code, b_code = bool(CODE_RE.match(ta)), bool(CODE_RE.match(tb))
        if a_code != b_code:
            key, val = (ta, tb) if a_code else (tb, ta)
        elif a_code or max(len(ta), len(tb)) <= FREE_PAIR_MAX_CHARS:
            key, val = ta, tb
        else:
            continue
        out.append(f"{key}: {val}")
    for row in rows:
        for left, right in zip(row, row[1:]):
            lt = left["text"].strip()
            if lt.endswith(":") or LABEL_RE.match(lt):
                out.append(f"{lt.rstrip(':').strip()}: {right['text'].strip()}")
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- OCR core

def _prepare(img):
    """EXIF-upright, opaque RGB, longest side <= MAX_SIDE."""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    img = img.convert("RGB")
    scale = MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                         Image.LANCZOS)
    return img


def _run(engine, img) -> list[dict]:
    """One engine pass; lines as ``{"box": [x0,y0,x1,y1], "text", "conf"}``."""
    import numpy as np
    out = engine(np.asarray(img))
    if out is None or out.boxes is None or out.txts is None:
        return []
    lines = []
    for box, text, conf in zip(out.boxes, out.txts, out.scores):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        if str(text).strip():
            lines.append({"box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                          "text": str(text), "conf": round(float(conf), 4)})
    return lines


def _mean_conf(lines: list[dict]) -> float:
    return sum(ln["conf"] for ln in lines) / len(lines) if lines else 0.0


def _cache_path(cache_dir: Path | None, key: str) -> Path | None:
    return Path(cache_dir) / f"{key}.ocr.json" if cache_dir else None


def _load_cache(cache_dir: Path | None, key: str) -> dict | None:
    p = _cache_path(cache_dir, key)
    if p is None or not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ocr_image(img, key: str, cache_dir: Path | None) -> dict:
    """OCR a PIL image. Result (cached at ``cache_dir/<key>.ocr.json``):
    ``{"lines": [{"box","text","conf"}], "rot": 0|90|180|270, "layout": [row, ...], "pairs": [...]}``.
    Boxes are in the (upright, downscaled) coordinates of the image that was read."""
    hit = _load_cache(cache_dir, key)
    if hit is not None:
        return hit
    engine = get_ocr()
    if engine is None:
        return {"lines": [], "rot": 0, "layout": [], "pairs": []}
    base = _prepare(img)
    lines, rot = _run(engine, base), 0
    if len(lines) < 2 or _mean_conf(lines) < config.OCR_MIN_CONF:
        best = _mean_conf(lines)
        for angle in (90, 180, 270):
            cand = _run(engine, base.rotate(angle, expand=True))
            if _mean_conf(cand) > best:
                lines, rot, best = cand, angle, _mean_conf(cand)
    rows = _rows(lines)
    raw_layout = ["  ".join(ln["text"].strip() for ln in row) for row in rows]
    raw_pairs = _pairs(lines, rows)
    # "layout"/"pairs" are what gets indexed (glue split); the raw OCR text stays in the cache next to them
    res = {"lines": lines, "rot": rot,
           "layout": [split_glue(r) for r in raw_layout], "pairs": [split_glue(p) for p in raw_pairs],
           "raw_layout": raw_layout, "raw_pairs": raw_pairs}
    p = _cache_path(cache_dir, key)
    if p is not None:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, p)
        except OSError:
            pass
    return res


# ---------------------------------------------------------------- sources

def _open_bytes(data: bytes):
    """PIL image from bytes, or None if undecodable / PIL missing."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:  # noqa: BLE001 - corrupt, EMF/WMF, truncated
        return None


@lru_cache(maxsize=256)
def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_pdf_page(path: Path, n1: int, dpi: int = 150):
    """PIL image of 1-based page ``n1`` via pypdfium2, or None if unavailable."""
    try:
        import pypdfium2
        pdf = pypdfium2.PdfDocument(str(path))
        try:
            return pdf[n1 - 1].render(scale=dpi / 72).to_pil()
        finally:
            pdf.close()
    except Exception:  # noqa: BLE001 - missing module, bad page, corrupt file
        return None


def _finish(img, key: str, cache_dir: Path | None) -> dict | None:
    """OCR a decoded image; tiny ones come back empty with ``"tiny": True`` (not cached)."""
    if img is None:
        return None
    if min(img.size) < MIN_SIDE:
        return {"lines": [], "rot": 0, "layout": [], "pairs": [], "tiny": True}
    return ocr_image(img, key, cache_dir)


def ocr_file(corpus: Path, rel: str, cache_dir: Path | None) -> dict | None:
    """OCR a standalone image file (key = sha1 of its bytes); None if unreadable."""
    try:
        data = (Path(corpus) / rel).read_bytes()
    except OSError:
        return None
    key = hashlib.sha1(data).hexdigest()
    hit = _load_cache(cache_dir, key)
    return hit if hit is not None else _finish(_open_bytes(data), key, cache_dir)


def ocr_zip_member(path: Path, member: str, cache_dir: Path | None) -> dict | None:
    """OCR an image inside a docx/pptx (key = sha1 of the member bytes); None if unreadable."""
    try:
        with zipfile.ZipFile(path) as z:
            data = z.read(member)
    except (OSError, KeyError, zipfile.BadZipFile):
        return None
    key = hashlib.sha1(data).hexdigest()
    hit = _load_cache(cache_dir, key)
    return hit if hit is not None else _finish(_open_bytes(data), key, cache_dir)


def ocr_pdf_page(corpus: Path, rel: str, n1: int, cache_dir: Path | None) -> dict | None:
    """OCR a rendered 1-based PDF page (key = sha1(file)-p{n1}); None if it cannot be rendered."""
    path = Path(corpus) / rel
    try:
        key = f"{_sha1_file(str(path))}-p{n1}"
    except OSError:
        return None
    hit = _load_cache(cache_dir, key)
    return hit if hit is not None else _finish(render_pdf_page(path, n1), key, cache_dir)


def _ocr_source(corpus: Path, rel: str, src: tuple) -> Callable[[Path | None], dict | None]:
    """Binds one ``images`` entry to the matching OCR function (cache_dir applied later)."""
    kind, ref = src[0], src[1]
    if kind == "file":
        return lambda cd: ocr_file(corpus, ref, cd)
    if kind == "zip":
        return lambda cd: ocr_zip_member(Path(corpus) / rel, ref, cd)
    return lambda cd: ocr_pdf_page(corpus, rel, int(ref), cd)


def _label(src: tuple) -> str:
    return f"p{src[1]}" if src[0] == "pdfpage" else str(src[1])


def ocr_all(parsed: list[dict], corpus: Path, cache_dir: Path, deadline_ts: float,
            threads: int = config.OCR_THREADS) -> dict[str, dict]:
    """OCR every image listed by ``parse.walk`` results.

    Returns ``{file_rel: {"lines": [row, ...], "pairs": [...], "images": n_ok, "missing": n_failed}}``.
    Work not started before ``deadline_ts`` counts as missing. Files with several images get
    ``[img <source>]`` prefixed to each row."""
    jobs = [(p["path"], tuple(src), len(p["images"])) for p in parsed for src in p["images"]]
    out: dict[str, dict] = {p["path"]: {"lines": [], "pairs": [], "images": 0, "missing": 0}
                            for p in parsed if p["images"]}
    if not jobs:
        return out
    engine_ok = get_ocr() is not None

    def work(job):
        rel, src, _ = job
        if not engine_ok or time.time() > deadline_ts:
            return None
        try:
            return _ocr_source(corpus, rel, src)(cache_dir)
        except Exception as e:  # noqa: BLE001 - one bad image must not stop the batch
            print(f"vision: OCR failed for {rel} {src}: {type(e).__name__}: {e}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        results = list(pool.map(work, jobs))
    for (rel, src, n_imgs), res in zip(jobs, results):
        entry = out[rel]
        if res is None:
            entry["missing"] += 1
        elif not res.get("tiny"):
            entry["images"] += 1
            prefix = f"[img {_label(src)}] " if n_imgs > 1 else ""
            entry["lines"] += [prefix + split_glue(row) for row in res["layout"]]   # old caches hold raw rows
            entry["pairs"] += [split_glue(p) for p in res["pairs"]]
    return out
