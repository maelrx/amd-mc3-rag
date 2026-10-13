"""Question -> Result: retrieve, read, verify, (retry), image check, cite, gates (spec 5.12), plus corpus indexing.

``answer`` never raises: every failure path returns the best verified result so far or ``schema.refusal(reason)``.
The reader is any object with the protocol of ``reader.Reader`` (``eval/stub_reader.StubReader`` on a laptop).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import chainguard
import cite
import config
import gates
import index as index_mod
import parse
import retrieve
import snap
import textnorm
import verify
import vision
import warrant
from budget import Deadline, Trace
from schema import Record, Result, Verdict, refusal

VQA_EDGE = int(config.IMAGE_LONGEST_EDGE_READ * 1.5)   # != the READ budget, so the second look is a different view
TRANSCRIBE_PER_IMAGE_S = 30.0
OCR_SKIP_LINES = 20                                     # dense scan already well covered by OCR
MIN_IMAGE_SIDE = vision.MIN_SIDE
_IMAGE_SRC = ("ocr", "vlm")


def _same_value(a: str, b: str) -> bool:
    """Equal after unit/currency stripping ("900 ms" and "900" are the same number)."""
    try:
        return textnorm.norm(snap.clean(str(a))) == textnorm.norm(snap.clean(str(b))) != ""
    except Exception:  # noqa: BLE001
        return textnorm.norm(str(a)) == textnorm.norm(str(b))


def _log(msg: str) -> None:
    print(f"pipeline: {msg}", file=sys.stderr, flush=True)


def has_vision(reader) -> bool:
    """Real VLM reader (``reader.Reader``) or anything that opts in with ``has_vision = True``."""
    return bool(getattr(reader, "has_vision", False)) or type(reader).__name__ == "Reader"


# ---------------------------------------------------------------- pixels

def _to_rgb(img):
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


def load_image(corpus: Path, file: str, desc):
    """PIL RGB pixels for one descriptor of ``file`` (("file", rel) | ("zip", member) | ("pdfpage", n1)), or None."""
    try:
        from PIL import Image
        kind, ref = desc[0], desc[1]
        if kind == "file":
            img = Image.open(Path(corpus) / ref)
            img.load()
        elif kind == "zip":
            with zipfile.ZipFile(Path(corpus) / file) as z:
                data = z.read(ref)
            img = Image.open(io.BytesIO(data))
            img.load()
        elif kind == "pdfpage":
            img = vision.render_pdf_page(Path(corpus) / file, int(ref))
            if img is None:
                return None
        else:
            return None
        return _to_rgb(img)
    except Exception:  # noqa: BLE001 - corrupt, missing, EMF/WMF, PIL absent: just no pixels
        return None


def load_images(pkg, corpus: Path | None, limit: int = config.MAX_IMAGES) -> list:
    """[(file, PIL)] for the image descriptors of the package (failures skipped, at most ``limit``)."""
    if corpus is None:
        return []
    out = []
    for file, desc in pkg.images:
        if len(out) >= limit:
            break
        img = load_image(corpus, file, desc)
        if img is not None:
            out.append((file, img))
    return out


def _with_image_notes(text: str, images: list) -> str:
    notes = [f"<IMAGE {k} = {f}; OCR text is in the records above>" for k, (f, _) in enumerate(images, 1)]
    return text + ("\n" + "\n".join(notes) if notes else "")


# ---------------------------------------------------------------- answer

class _Run:
    """Mutable state of one ``answer`` call (so ``finish`` and the outer guard see the same trace)."""

    def __init__(self, trace: Trace):
        self.trace = trace
        self.attempts = 0
        self.support_rid: int | None = None
        self.best: Result | None = None      # verified result, kept while optional checks still run

    def finish(self, res: Result, reason: str) -> Result:
        res.trace = {**self.trace.to_dict(), **res.trace, "reason": reason,
                     "support_rid": self.support_rid, "attempts": self.attempts}
        return res

    def refuse(self, reason: str) -> Result:
        return self.finish(refusal(reason), reason)


def _drop_lines(text: str, rids) -> str:
    drop = tuple(f"[R{r}] " for r in rids)
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith(drop))


def _support_image(index, rec, images: list, corpus: Path | None):
    """(file, PIL) for the image behind an OCR/VLM support record, else None."""
    fe = index.file_by_path[rec.file]
    if corpus is not None and len(fe.images) > 1:
        # several images in one file: OCR rows carry an "[img <label>]" prefix naming the source
        for d in fe.images:
            label = f"p{d[1]}" if d[0] == "pdfpage" else str(d[1])
            if f"[img {label}]" in rec.text:
                img = load_image(corpus, rec.file, tuple(d))
                if img is not None:
                    return (rec.file, img)
    mine = [(f, im) for f, im in images if f == rec.file]
    if mine:
        return mine[0]
    if corpus is not None and fe.images:
        img = load_image(corpus, rec.file, tuple(fe.images[0]))
        if img is not None:
            return (rec.file, img)
    return None


def _image_disagrees(run: _Run, index, reader, query: str, v, images: list, corpus: Path | None,
                     deadline: Deadline) -> bool:
    """True when a second visual reading of the support image contradicts the answer (spec 5.12 step 7)."""
    rec = index.record(v.support_rid)
    if rec.src not in _IMAGE_SRC or not deadline.require(config.VQA_MIN_REMAINING_S):
        return False
    span = v.answer_span or ""
    # RapidOCR already read this exact token from the pixels: nothing to cross-check
    if any(r.src == "ocr" and snap.find(span, r.text) for r in (index.record(x) for x in index.by_file[rec.file])):
        return False
    image = _support_image(index, rec, images, corpus)
    if image is None:
        return False
    alt = reader.vqa(query, image, deadline, longest_edge=VQA_EDGE)
    run.trace.note("vqa_alt", alt)
    if not str(alt).strip():
        return False
    na, nv = textnorm.norm(alt), textnorm.norm(span)
    if na == nv or (nv and nv in na):       # equal, or the model answered with a sentence around the value
        return False
    run.trace.note("image_disagree", alt)
    return True


def _context_gain(reader, pkg_text: str, query: str, images: list, ans: str, drop_rids, drop_image: bool):
    lp_with = reader.logprob(pkg_text, query, images, ans)
    lp_without = reader.logprob(_drop_lines(pkg_text, drop_rids), query, [] if drop_image else images, ans)
    return lp_with - lp_without


def _answer(run: _Run, index, reader, query: str, deadline: Deadline, corpus: Path | None) -> Result:
    trace = run.trace
    if reader is None:
        return run.refuse("NO_READER")
    if deadline.expired():
        return run.refuse("TIMEOUT")

    with trace.stage("analyze"):
        q = retrieve.analyze(query, index)
    with trace.stage("gate_a_pre"):
        pre = gates.gate_a_pre(q, index)
    if pre is not None:
        return run.refuse(pre.reason)

    exclude: set[int] = set()
    first_reason, first_answer = None, None
    v = out = pkg = None
    images: list = []
    for attempt in (1, 2):
        if attempt == 2 and not (not v.ok and not v.hard and deadline.require(config.RETRY_MIN_REMAINING_S)):
            break
        run.attempts = attempt
        with trace.stage(f"assemble{attempt}"):
            pkg = retrieve.assemble(index, q, attempt, exclude_rids=exclude)
        with trace.stage(f"images{attempt}"):
            images = load_images(pkg, corpus)
        text = _with_image_notes(pkg.text, images)
        rd = deadline.sub(1.0)
        if rd.expired():
            return run.refuse("TIMEOUT")
        with trace.stage(f"read{attempt}"):
            out = reader.read(text, query, images, rd)
        with trace.stage(f"verify{attempt}"):
            v = verify.verify(q, out, pkg, index)
        trace.note(f"reason_{attempt}", v.reason)
        if attempt == 2 and v.ok and first_reason == "UNIT_MISMATCH" and first_answer \
                and _same_value(out.answer, first_answer):
            # the same number re-found in a record without a unit cannot change its unit: refuse
            v = Verdict(False, "UNIT_MISMATCH", violators=v.violators, hard=True)
            trace.note("reason_2", "UNIT_MISMATCH_SAME_VALUE")
        if attempt == 2 and v.ok and first_reason == "EMPTY" \
                and verify.coverage(q, index, v.support_rid, v.bridge_rids) < config.COVERAGE_MIN:
            # the model found nothing in the focused package; a low-overlap line from the wide one is a guess
            v = Verdict(False, "EMPTY", violators=v.violators)
            trace.note("reason_2", "EMPTY_THEN_LOW_COVERAGE")
        if v.ok or v.hard:
            break
        first_reason, first_answer = v.reason, out.answer
        exclude = set(v.violators)
    if not v.ok:
        return run.refuse(v.reason)

    # Second look when the support line barely overlaps the question (prose that mentions the
    # product instead of the attribute line). Keep whichever verified answer covers more.
    if run.attempts == 1 and deadline.require(config.RETRY_MIN_REMAINING_S):
        cov = verify.coverage(q, index, v.support_rid, v.bridge_rids)
        trace.note("coverage_1", round(cov, 2))
        if cov < config.COVERAGE_MIN:
            with trace.stage("assemble2"):
                pkg2 = retrieve.assemble(index, q, 2, exclude_rids={v.support_rid})
            with trace.stage("images2"):
                images2 = load_images(pkg2, corpus)
            rd = deadline.sub(1.0)
            if not rd.expired():
                with trace.stage("read2"):
                    out2 = reader.read(_with_image_notes(pkg2.text, images2), query, images2, rd)
                with trace.stage("verify2"):
                    v2 = verify.verify(q, out2, pkg2, index)
                trace.note("reason_2", v2.reason)
                if v2.reason == "UNIT_MISMATCH" and _same_value(out2.answer, v.answer_span or out.answer):
                    # the wider look found the same number next to a conflicting unit: the first hit was a bare copy
                    return run.refuse("UNIT_MISMATCH")
                if v2.ok and textnorm.norm(v2.answer_span or "") != textnorm.norm(v.answer_span or ""):
                    cov2 = verify.coverage(q, index, v2.support_rid, v2.bridge_rids)
                    trace.note("coverage_2", round(cov2, 2))
                    if cov2 >= cov + 0.15:
                        v, out, pkg, images = v2, out2, pkg2, images2
                        run.attempts = 2
    if config.COVERAGE_REFUSE > 0 and not v.bridge_rids and index.record(v.support_rid).src not in _IMAGE_SRC:
        cov_final = verify.coverage(q, index, v.support_rid, v.bridge_rids)
        trace.note("coverage_final", round(cov_final, 2))
        if cov_final < config.COVERAGE_REFUSE:
            return run.refuse("LOW_COVERAGE")
    if config.CHAIN_GUARD:
        with trace.stage("chain_guard"):
            swapped = chainguard.apply(q, v, out, pkg, index, reader, query, deadline, trace)
        if swapped is not None:
            v, out, pkg, images = swapped
            run.attempts = 2
    run.support_rid = v.support_rid

    # optional second look at the pixels; an error or an empty VQA keeps the answer
    with trace.stage("image_check"):
        try:
            if _image_disagrees(run, index, reader, query, v, images, corpus, deadline):
                return run.refuse("IMAGE_DISAGREE")
        except Exception as e:  # noqa: BLE001
            trace.note("image_check_error", repr(e))

    with trace.stage("gate_a_post"):
        ok_post, reason = gates.gate_a_post(q, v, index)
    if not ok_post:
        return run.refuse(reason)

    with trace.stage("cite"):
        cites = cite.cite(q, v, index, reader=None, pkg=pkg, deadline=deadline)
        ans = verify.answer_text(v, q)
    if not ans or not cites:
        return run.refuse("EMPTY")
    sup_src = index.record(v.support_rid).src
    conf = config.CONF_CHAIN if v.bridge_rids else (config.CONF_IMAGE if sup_src in _IMAGE_SRC else config.CONF_SINGLE)
    run.best = Result(ans, cites, conf, {})

    mode = gates.gate_b_mode()
    if mode != "0":
        # "log": same signals, recorded in the trace, never refuses. "1": refuses below the thresholds.
        # Any error inside the gate keeps the verified answer (explicit fail-open, with a note).
        with trace.stage("gate_b"):
            try:
                gain = None
                if deadline.require(config.LOO_MIN_REMAINING_S):
                    try:
                        drop = [v.support_rid, *v.bridge_rids]
                        gain = _context_gain(reader, _with_image_notes(pkg.text, images), query, images, ans, drop,
                                             drop_image=any(index.record(r).src in _IMAGE_SRC for r in drop))
                    except Exception as e:  # noqa: BLE001
                        trace.note("gate_b_error", repr(e))
                trace.note("context_gain", gain)
                trace.note("gate_b_s1", out.mean_logprob)
                keep = gates.gate_b(out, gain)
                if mode == "log":
                    trace.note("gate_b_would_refuse", not keep)
                elif not keep:
                    return run.refuse("GATE_B")
            except Exception as e:  # noqa: BLE001
                trace.note("gate_b_error", repr(e))

    wmode = warrant.mode()
    if wmode != "0":
        # H6: "log" records the YES/NO score, never refuses. "1" refuses below WARRANT_THETA. An error keeps the answer.
        with trace.stage("warrant"):
            try:
                score = warrant.check(reader, index, q, v, pkg, out, deadline, trace, images, ans)
                if wmode == "1" and score is not None and score < config.WARRANT_THETA:
                    return run.refuse("WARRANT")
            except Exception as e:  # noqa: BLE001
                trace.note("warrant_error", repr(e))
    return run.finish(run.best, "OK")


def answer(index, reader, query: str, deadline: Deadline, corpus: Path | None = None,
           trace: Trace | None = None) -> Result:
    """Never raises. Exception or expired deadline -> best verified result so far, else a refusal."""
    run = _Run(trace if trace is not None else Trace())
    try:
        return _answer(run, index, reader, query, deadline, Path(corpus) if corpus is not None else None)
    except Exception as e:  # noqa: BLE001
        _log(f"answer failed: {type(e).__name__}: {e}")
        run.trace.note("error", f"{type(e).__name__}: {e}")
        if run.best is not None:
            return run.finish(run.best, "OK")
        return run.refuse("ERROR")


# ---------------------------------------------------------------- indexing

def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _update_status(index_dir: Path, **fields) -> None:
    p = Path(index_dir) / "status.json"
    try:
        cur = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cur = {}
    cur.update(fields)
    _write_atomic(p, json.dumps(cur))


def _build(corpus: Path, index_dir: Path, t0: float):
    parsed = parse.walk(corpus)
    t_parse = time.time() - t0
    t1 = time.time()
    try:
        ocr = vision.ocr_all(parsed, corpus, index_dir / "images", t0 + config.OCR_BUDGET_S)
    except Exception as e:  # noqa: BLE001 - OCR is an enhancement
        _log(f"OCR failed: {type(e).__name__}: {e}")
        ocr = {}
    t_ocr = time.time() - t1
    idx = index_mod.build(parsed, ocr)
    index_mod.save(index_dir, idx)
    _update_status(index_dir, files=len(idx.files), records=len(idx.records),
                   images=sum(int(v.get("images", 0)) for v in ocr.values()),
                   t_parse=round(t_parse, 3), t_ocr=round(t_ocr, 3))
    _write_atomic(index_dir / "INDEXED", f"{time.time():.0f}\n")
    return idx


def _vlm_prefix(desc, n_zip: int, j_zip: int) -> str:
    if desc[0] == "pdfpage":
        return f"p{desc[1]}.vlm"
    if desc[0] == "zip":
        return "zip.vlm" if n_zip <= 1 else f"zip{j_zip}.vlm"
    return "vlm"


def _image_key(corpus: Path, file: str, desc) -> str | None:
    """Content key matching vision's cache keys (sha1 of bytes; pdf: sha1(file)-p{n})."""
    try:
        if desc[0] == "file":
            return hashlib.sha1((corpus / desc[1]).read_bytes()).hexdigest()
        if desc[0] == "zip":
            with zipfile.ZipFile(corpus / file) as z:
                return hashlib.sha1(z.read(desc[1])).hexdigest()
        if desc[0] == "pdfpage":
            return f"{hashlib.sha1((corpus / file).read_bytes()).hexdigest()}-p{desc[1]}"
    except (OSError, KeyError, zipfile.BadZipFile):
        pass
    return None


def _pending_images(idx, corpus: Path) -> list[tuple]:
    """(priority, file, desc, prefix) for every image not yet transcribed, standalone images first."""
    done = {(r.file, r.loc.rsplit(".L", 1)[0]) for r in idx.records if r.src == "vlm"}
    items = []
    for fe in idx.files:
        if fe.status != "ok" or not fe.images:
            continue
        recs = [idx.record(r) for r in idx.by_file.get(fe.file, [])]
        n_lines = sum(1 for r in recs if r.src == "ocr" and r.loc.startswith("ocr.L"))
        if n_lines >= OCR_SKIP_LINES and not any(r.loc == "ocr.pairs" for r in recs):
            continue
        n_zip = sum(1 for d in fe.images if d[0] == "zip")
        j_zip = 0
        for d in fe.images:
            d = tuple(d)
            prefix = _vlm_prefix(d, n_zip, j_zip)
            if d[0] == "zip":
                j_zip += 1
            if (fe.file, prefix) in done:
                continue
            prio = {"file": 0, "zip": 1, "pdfpage": 2}.get(d[0], 3)
            items.append((prio, fe.file, d, prefix))
    items.sort(key=lambda t: t[0])      # stable: walk order inside a priority
    return items


def transcribe_pending(idx, corpus: Path, index_dir: Path, reader, deadline: Deadline):
    """VLM-transcribe images by priority until the deadline margin; appends ``vlm`` records. -> (Index, stats)."""
    corpus, index_dir = Path(corpus), Path(index_dir)
    t0 = time.time()
    cache = index_dir / "images"
    items = _pending_images(idx, corpus)
    done = skipped = appended = 0
    todo = len(items)
    new_records: list[Record] = []
    for n, (_, file, desc, prefix) in enumerate(items):
        if deadline.remaining() < config.VLM_INDEX_MARGIN_S:
            break
        key = _image_key(corpus, file, desc)
        cpath = cache / f"{key}.vlm.txt" if key else None
        text = ""
        if cpath is not None and cpath.exists():
            try:
                text = cpath.read_text(encoding="utf-8")
            except OSError:
                text = ""
        if not text:
            img = load_image(corpus, file, desc)
            if img is None or min(img.size) < MIN_IMAGE_SIDE:
                skipped += 1
                continue
            text = reader.transcribe((file, img), Deadline(min(deadline.ts - config.VLM_INDEX_MARGIN_S,
                                                                time.time() + TRANSCRIBE_PER_IMAGE_S))) or ""
            if text.strip() and cpath is not None:
                try:
                    cache.mkdir(parents=True, exist_ok=True)
                    _write_atomic(cpath, text)
                except OSError:
                    pass
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        if not lines:
            skipped += 1
            continue
        new_records.extend(Record(rid=0, file=file, loc=f"{prefix}.L{i}", text=ln, src="vlm")
                           for i, ln in enumerate(lines, 1))
        appended += 1
        done += 1
    pending = max(0, todo - done - skipped)
    if new_records:  # one append + one reload for the whole batch, not one per image
        index_mod.append_records(index_dir, new_records)
        idx = index_mod.load(index_dir)
    t_vlm = round(time.time() - t0, 3)
    try:
        _update_status(index_dir, vlm_done=done, vlm_pending=pending, t_vlm=t_vlm)
    except OSError:
        pass
    return idx, {"vlm_done": done, "vlm_pending": pending, "t_vlm": t_vlm}


def index_corpus(corpus: Path, index_dir: Path, reader=None, deadline: Deadline | None = None,
                 rebuild: bool = False):
    """Index from disk when ``app.py --index`` already persisted it, else build it; then VLM-transcribe images.

    ``rebuild`` ignores whatever is on disk (the client-side build failed and may have left a partial dir)."""
    corpus, index_dir = Path(corpus), Path(index_dir)
    t0 = time.time()
    idx = None
    if not rebuild and (index_dir / "INDEXED").exists():
        try:
            idx = index_mod.load(index_dir)
        except Exception as e:  # noqa: BLE001
            _log(f"persisted index unreadable ({type(e).__name__}: {e}); rebuilding")
    if idx is None:
        idx = _build(corpus, index_dir, t0)
    if reader is not None and deadline is not None and has_vision(reader):
        try:
            idx, _ = transcribe_pending(idx, corpus, index_dir, reader, deadline)
        except Exception as e:  # noqa: BLE001 - transcription is best effort
            _log(f"transcribe_pending failed: {type(e).__name__}: {e}")
    return idx
