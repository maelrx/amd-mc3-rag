"""Model "warrant" check (H6, behind MC3_WARRANT). Stdlib only.

After the deterministic verifier accepted an answer, ask the reader a YES/NO question over the evidence it was
read from: does it state exactly this value for exactly this question? The score is
log P(YES) - log P(NO) (``Reader.yes_no``). It targets what the verifier cannot see: a neighbouring event's code,
a low-coverage row of another product, a unit or column mix-up, the wrong one of two similar files.

``check`` only computes and records; the refuse/keep decision lives in the pipeline hook.
"""

from __future__ import annotations

import config

_IMAGE_SRC = ("ocr", "vlm")


def mode() -> str:
    """"0" off | "log" compute and record, never refuse | "1" enforce. Read at call time (A/B runners flip config)."""
    m = str(config.WARRANT).strip().lower()
    return m if m in ("1", "log") else "0"


MAX_LINE_CHARS = 600       # same cut the reader's package uses (retrieve._line)
MAX_EVIDENCE_CHARS = 6000  # the yes_no prefill is not interruptible: keep it to a few thousand tokens at most


def _clip(text: str, span: str | None = None) -> str:
    """At most MAX_LINE_CHARS; for a long record, the window around the verified span."""
    if len(text) <= MAX_LINE_CHARS:
        return text
    pos = text.find(span) if span else -1
    if pos < 0:
        return text[:MAX_LINE_CHARS] + " …"
    start = max(0, pos - MAX_LINE_CHARS // 2)
    return ("… " if start else "") + text[start:start + MAX_LINE_CHARS] + " …"


def _line(rec, span: str | None = None) -> str:
    return f"[R{rec.rid}] {rec.file} {rec.loc}: {_clip(rec.text, span)}"


def evidence_rids(index, verdict) -> list[int]:
    """Support + bridge records (document order) with the support file's name record and nearest heading."""
    sup = verdict.support_rid
    rids = [sup, *verdict.bridge_rids]
    try:
        for x in index.context_rids(sup):
            r = index.record(x)
            if r.meta or index._is_heading(r) or _is_table_header(r.text):
                rids.append(x)
        hdr = _table_header_rid(index, sup)
        if hdr is not None:
            rids.append(hdr)
    except Exception:  # noqa: BLE001 - context is a nicety
        pass
    return sorted(dict.fromkeys(rids))


def _is_table_row(text: str) -> bool:
    t = text.strip()
    return t.startswith("|") and t.count("|") >= 3


def _is_table_header(text: str) -> bool:
    t = text.strip()
    return _is_table_row(t) and not any(ch.isdigit() for ch in t)


def _table_header_rid(index, rid: int, max_back: int = 60) -> int | None:
    """For a markdown/pipe table row, the header row of its table (first row of the contiguous '|' block):
    without it the warrant judge sees '| VDDQ_HBM | 0.80 | 292 |' and cannot tell which column 292 is."""
    rec = index.record(rid)
    if not _is_table_row(rec.text):
        return None
    best = None
    for x in range(rid - 1, max(-1, rid - 1 - max_back), -1):
        r = index.record(x)
        if r.file != rec.file or not _is_table_row(r.text):
            break
        if not set(r.text.replace("|", "").strip()) <= set("-: "):   # skip the |---|---| separator
            best = x
    return best


def evidence_text(index, verdict) -> str:
    """Evidence lines in document order, each clipped; the support line always stays, others are dropped once the
    total would pass MAX_EVIDENCE_CHARS (bridges first, then context)."""
    sup = verdict.support_rid
    rids = evidence_rids(index, verdict)
    lines = {r: _line(index.record(r), verdict.answer_span if r == sup else None) for r in rids}
    keep = {sup}
    budget = MAX_EVIDENCE_CHARS - len(lines.get(sup, ""))
    for r in [*verdict.bridge_rids, *[x for x in rids if x != sup and x not in verdict.bridge_rids]]:
        if r in lines and r not in keep and len(lines[r]) + 1 <= budget:
            keep.add(r)
            budget -= len(lines[r]) + 1
    return "\n".join(lines[r] for r in rids if r in keep)


def check(reader, index, q, verdict, pkg, out, deadline, trace, images=None, answer=None) -> float | None:
    """Warrant score of the verified answer, or None (reader without ``yes_no``, no time, model error).

    ``images`` are the loaded (file, PIL) pairs of the package; only those of OCR/VLM-sourced evidence files are
    attached. ``answer`` is the final answer text (default: the verified span). The score is noted as ``warrant`` in the trace."""
    fn = getattr(reader, "yes_no", None)
    if fn is None:
        trace.note("warrant", None)
        return None
    recs = [index.record(r) for r in evidence_rids(index, verdict)]
    files = {r.file for r in recs if r.src in _IMAGE_SRC}
    imgs = [(f, im) for f, im in (images or []) if f in files]
    answer = answer or verdict.answer_span or getattr(out, "answer", "")
    score = fn(evidence_text(index, verdict), q.text, answer, imgs or None, deadline)
    score = None if score is None else float(score)
    trace.note("warrant", None if score is None else round(score, 3))
    return score
