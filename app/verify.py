"""Deterministic verifier: reader output -> Verdict (span, support record, bridges). Stdlib only."""

from __future__ import annotations

import os
import re
from collections import Counter

import config
import snap
import textnorm
from retrieve import _CHAIN_VERB_RE
from schema import Package, Q, ReaderOut, Verdict

# kinds named in a question -> file kinds that satisfy them (txt also covers md)
_NAMED_KIND_FILES = {"txt": {"txt", "md"}}

_ATTR_LEXICON = {
    "temperature": r"temperature|junction|thermal|\btj\b",
    "version": r"fixed_in|version|release|firmware",
    "part_number": r"\bpart\b|p/n|part_number|assembly",
    "price": r"price|cost|usd|\$",
    "cost": r"price|cost|usd|\$",
    "timeout": r"timeout|time_out",
    "pin": r"\bpin\b|signal",
    "revision": r"\brev\b|revision",
    "error_code": r"error|\bcode\b|\bE\d{3,5}\b",
    "schedule": r"quarter|sampling|\bq[1-4]\b|\bfy\b",
    "lead_time": r"\blead\b",
    "serial": r"serial|\bsn\b",
    "power": r"power|\bw\b",
    "speed": r"rpm|mhz|speed",
}
_ATTR_RES = {k: re.compile(v, re.I) for k, v in _ATTR_LEXICON.items()}

_KV_RE = re.compile(r"\b[A-Za-z_]\w*=")
_KEY_BEFORE_RE = re.compile(r"([A-Za-z_]\w*)\)?\s*=\s*[\"']?$")
_RIGHT_UNIT_RE = re.compile(r"^\s*(°\s*[CF]\b|[A-Za-z%/]+)")
_KEY_SUFFIXES = (("_ms", "ms"), ("_sec", "s"), ("_min", "min"), ("_s", "s"), ("_h", "h"))
_UNIT_CANON = {
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "ms": "ms", "millisecond": "ms", "milliseconds": "ms",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "c": "c", "celsius": "c", "deg": "c", "degree": "c", "degrees": "c",
    "gigabytes": "gb", "gibibytes": "gib", "megabytes": "mb", "mebibytes": "mib", "terabytes": "tb",
    "tebibytes": "tib", "hertz": "hz", "kilohertz": "khz", "megahertz": "mhz", "gigahertz": "ghz",
    "day": "d", "days": "d", "week": "w", "weeks": "w", "month": "mo", "months": "mo",
}


# ---------------------------------------------------------------- helpers

def kind_matches(file_kind: str, named: set[str]) -> bool:
    return any(file_kind == n or file_kind in _NAMED_KIND_FILES.get(n, ()) for n in named)


def _ids_of(text: str) -> set[str]:
    return {textnorm.nid(x) for x in textnorm.ids(text)}


def shared_rare_ids(index, rid_a: int, rid_b: int) -> list[str]:
    """nids present in both records with corpus df <= RARE_DF (and indexed)."""
    common = _ids_of(index.record(rid_a).text) & _ids_of(index.record(rid_b).text)
    return sorted(n for n in common if n in index.ids and index.ids[n]["df"] <= config.RARE_DF)


def _canon_unit(u: str) -> str:
    t = re.sub(r"[°\s]", "", str(u)).lower()
    return _UNIT_CANON.get(t, t)


_QFILE_RE = re.compile(r"(?<![\w/.])([\w\-]+(?:\.[\w\-]+)*\.(?:pdf|docx|xlsx|xlsm|csv|txt|log|md|py|json|ya?ml|png|jpe?g|pptx))\b", re.I)


def _named_file_missing(q: Q, index) -> bool:
    """The question names a file (roadmap_fy27.docx) that no corpus file matches by basename."""
    names = {os.path.basename(f.file).lower() for f in index.files}
    for m in _QFILE_RE.finditer(q.text):
        if m.group(1).lower() not in names:
            return True
    return False


def premise_check(q: Q, index) -> Verdict | None:
    """FALSE_PREMISE / ONLY_IN_INELIGIBLE (hard) when a rare ID of the question has no usable evidence."""
    if _named_file_missing(q, index):
        return Verdict(False, "FALSE_PREMISE", hard=True)
    for raw in q.rare_ids:
        e = index.ids.get(textnorm.nid(raw))
        if not e:
            return Verdict(False, "FALSE_PREMISE", hard=True)
        if q.wants_old:
            continue
        # name/OCR/VLM records count as present; stale rows and ineligible files do not
        if not any(index.eligible(rid) for rid in e["rids"]):
            return Verdict(False, "ONLY_IN_INELIGIBLE", hard=True)
    return None


def _is_prose_line(text: str) -> bool:
    t = text.lstrip()
    return t.startswith(("#", '"""', "'''")) or ("=" not in t and "(" not in t)


def _code_comment_loses(index, rid: int) -> bool:
    """A .py comment (or prose line) loses to a code constant/default literal in the same file."""
    r = index.record(rid)
    if not r.file.endswith(".py"):
        return False
    if not (r.loc.endswith(".comment") or (re.fullmatch(r"L\d+", r.loc) and _is_prose_line(r.text))):
        return False
    for x in index.by_file[r.file]:
        o = index.record(x)
        if (o.loc.endswith(".const") or o.loc.endswith(".default")) and textnorm.numbers(o.text):
            return True
    return False


def _context_texts(index, rid: int) -> list[str]:
    """Context records of rid, minus data rows (k=v; k=v) so a neighbour row cannot vouch for this one."""
    out = []
    for x in index.context_rids(rid):
        t = index.record(x).text
        if x != rid and len(_KV_RE.findall(t)) >= 2:
            continue
        out.append(t)
    return out


def _num_forms(n: str) -> set[str]:
    """Comparable forms of a number: unsigned ("-60" in a range is 60) and integer-collapsed ("02" is 2)."""
    k = textnorm.num_key(n).lstrip("+-")
    forms = {k}
    try:
        f = float(k)
        if f.is_integer():
            forms.add(str(int(f)))
    except ValueError:
        pass
    return forms


def _entity_in_file(index, file: str, entity: str) -> bool:
    cache = getattr(index, "_ent_cache", None)
    if cache is None:
        cache = index._ent_cache = {}
    key = (file, entity)
    if key not in cache:
        cache[key] = any(textnorm.find_span(index.record(x).text, entity) for x in index.by_file.get(file, []))
    return cache[key]


def _entity_ok(index, rid: int, entity: str) -> bool:
    r = index.record(rid)
    if textnorm.find_span(r.text, entity):
        return True
    if any(textnorm.find_span(t, entity) for t in _context_texts(index, rid)):
        return True
    if index.file_of(rid).n_products <= 1:
        return _entity_in_file(index, r.file, entity)
    return False


def _span_unit(text: str, span: str) -> str | None:
    """Canonical unit attached to the span: its own tail, the word after it, or the key suffix."""
    m = textnorm.UNIT_RE.search(span)
    if m and textnorm.numbers(span[:m.start()]) and not span[:m.start()].strip().isalpha():
        return _canon_unit(m.group().strip())
    hit = textnorm.find_span(text, span)   # the token-bounded occurrence, not the first substring
    pos = hit[0] if hit else text.find(span)
    if pos < 0:
        return None
    if text[pos + len(span):pos + len(span) + 2].lstrip().startswith("%"):
        return "%"  # "reduced to 60%" is a percentage, never a seconds value
    mr = _RIGHT_UNIT_RE.match(text[pos + len(span):])
    if mr and textnorm.UNIT_RE.fullmatch(mr.group(1)):
        return _canon_unit(mr.group(1))
    mk = _KEY_BEFORE_RE.search(text[:pos])
    if mk:
        key = mk.group(1).lower()
        for suf, unit in _KEY_SUFFIXES:
            if key.endswith(suf):
                return unit
    return None


def _attr_score(q: Q, text: str) -> int:
    n = 0
    for term in q.attr_terms:
        rx = _ATTR_RES.get(term) or re.compile(re.escape(term), re.I)
        if rx.search(text):
            n += 1
    return n


_KIND_AUTHORITY = {"pdf": 5, "xlsx": 4, "csv": 4, "py": 4, "docx": 3, "pptx": 3, "image": 3, "log": 3,
                   "txt": 1, "md": 1}
# letters-only boundaries on the BASENAME: "specs/" the directory, "special", "inspect", "aspect" do not count
_PRIMARY_NAME_RE = re.compile(r"(?<![A-Za-z])(?:datasheet|spec|specification|errata|pinout|schematic|bom)s?(?![A-Za-z])", re.I)
_SECONDARY_NAME_RE = re.compile(
    r"(?<![A-Za-z])(?:marketing|brief|brochure|faq|blog|whitepaper|white_paper|overview|newsletter)s?(?![A-Za-z])", re.I)


def _authority(fe) -> float:
    """Late tie-break when the same value sits in several equally relevant files: specs beat prose."""
    a = _KIND_AUTHORITY.get(fe.kind, 2) / 2
    name = os.path.basename(fe.file)
    if _PRIMARY_NAME_RE.search(name):
        a += 2
    if _SECONDARY_NAME_RE.search(name):
        a -= 1
    return a


_FRAME_WORDS = {"what", "which", "who", "whom", "when", "where", "how", "does", "do", "did", "is", "are", "was",
                "were", "value", "many", "much", "according", "listed", "given", "specified", "shown", "show",
                "shows", "please", "underlying", "exactly", "currently"}


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s", "ly"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _covers(qstem: str, have: set[str]) -> bool:
    if qstem in have:
        return True
    if len(qstem) >= 3:
        return any(h.startswith(qstem) or (len(h) >= 3 and qstem.startswith(h)) for h in have)
    return False


def coverage(q: Q, index, rid: int, extra_rids=()) -> float:
    """Share of the question's content words present in the support record, its bridges or context.

    A low value means the model answered from a line that merely mentions the product
    ("commissioned with the inlet at 26 C") instead of the attribute line; the pipeline then
    takes a second look without that record and keeps whichever verified answer covers more.
    Bridge records count because a multi-hop question describes the first link, not the value row.
    """
    skip = {textnorm.nid(i) for i in q.ids} | {textnorm.nid(e) for e in q.entities}
    qset = set()
    for t in textnorm.tokens(q.text):
        if t in _FRAME_WORDS or not any(c.isalpha() for c in t) or textnorm.nid(t) in skip:
            continue
        qset.add(_stem(t))
    if not qset:
        return 1.0
    rids = [rid, *extra_rids]
    # only the lines themselves plus the file names: headers and first lines of a long document
    # would make any line of a cooling white paper look like it covers "coolant temperature"
    texts = [index.record(r).text for r in rids]
    for r in rids:
        texts.extend(index.record(c).text for c in index.context_rids(r) if index.record(c).meta)
    have = {_stem(t) for t in textnorm.tokens(" ".join(texts))}
    return sum(1 for s in qset if _covers(s, have)) / len(qset)


def _question_words(q: Q) -> set[str]:
    skip = {textnorm.nid(i) for i in q.ids} | {textnorm.nid(e) for e in q.entities}
    for raw in (*q.ids, *q.entities):   # "NB-23" must not leave "nb" behind as a content word
        skip.update(textnorm.stem(p) for p in re.split(r"[-._/]", textnorm.fold_dashes(raw).lower()) if p)
    return {_stem(t) for t in textnorm.tokens(q.text)
            if t not in _FRAME_WORDS and any(c.isalpha() for c in t) and textnorm.nid(t) not in skip
            and t not in skip}


_REMOVED_COVER = 0.8


def _record_cover(qwords: set[str], index, rid: int) -> float:
    have = {_stem(t) for t in textnorm.tokens(index.record(rid).text)}
    return sum(1 for w in qwords if _covers(w, have)) / len(qwords)


def removed_in_current(q: Q, index, rid: int) -> bool:
    """True when ONE record of the withdrawn/superseded revision of the same family covers >= 80% of the
    question's content words and no single record of the current file does: the attribute was removed,
    so the value the model found in the current file is a different attribute (T18 canary)."""
    fe = index.file_of(rid)
    sibs = [f for f in index.files if f.family == fe.family and f.file != fe.file
            and (f.withdrawn or f.superseded_by == fe.file) and f.status == "ok"]
    if not sibs:
        return False
    qwords = _question_words(q)
    if not qwords:
        return False
    old_best = max((_record_cover(qwords, index, r) for f in sibs for r in index.by_file.get(f.file, [])),
                   default=0.0)
    if old_best < _REMOVED_COVER:
        return False
    cur_best = max((_record_cover(qwords, index, r) for r in index.by_file.get(fe.file, [])), default=0.0)
    return cur_best < _REMOVED_COVER


_CODE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{1,4})[-‐‑–]?(\d{1,4}[A-Za-z]?)(?![A-Za-z0-9])")


def _names_sibling_only(text: str, entity: str) -> bool:
    """The record names another product of the same family (NB-80) and never the asked one (NB-60)."""
    m = re.fullmatch(r"([A-Za-z]{1,4})-?(\d{1,4}[A-Za-z]?)", textnorm.fold_dashes(entity))
    if not m:
        return False
    prefix, code = m.group(1).upper(), m.group(2).upper()
    seen_self = seen_sibling = False
    for p, c in _CODE_TOKEN_RE.findall(textnorm.fold_dashes(text)):
        if p.upper() != prefix:
            continue
        if c.upper() == code:
            seen_self = True
        else:
            seen_sibling = True
    return seen_sibling and not seen_self


def _check(q: Q, index, rid: int, span: str) -> str | None:
    """First hard-failure code of a candidate, else None."""
    r = index.record(rid)
    have: set[str] = set()
    if q.numbers:
        for n in textnorm.numbers(r.text):
            have |= _num_forms(n)
    ctx: set[str] | None = None
    for n in q.numbers:
        if _num_forms(n) & have:
            continue
        if ctx is None:   # "2.0" is in the path v2.0/release_manifest.txt, not in the row: that still conditions it
            ctx = set()
            for t in (index.file_of(rid).file, *_context_texts(index, rid)):
                for x in textnorm.numbers(t):
                    ctx |= _num_forms(x)
        if not (_num_forms(n) & ctx):
            return "COND_MISMATCH"
    for ent in q.entities:
        if not _entity_ok(index, rid, ent):
            return "ENTITY_MISMATCH"
        if _names_sibling_only(r.text, ent):
            return "ENTITY_MISMATCH"   # "Required driver for the NB-80 kit ..." is not about the NB-60
    if q.unit:
        u = _span_unit(r.text, span)
        if u is not None and u != _canon_unit(q.unit):
            return "UNIT_MISMATCH"
    return None


def _chain_parents(pkg: Package, rid: int) -> list[int]:
    out, seen = [], {rid}
    cur = pkg.via.get(rid)
    while cur is not None and cur not in seen:
        out.append(cur)
        seen.add(cur)
        cur = pkg.via.get(cur)
    return out


def _bridges(q: Q, out: ReaderOut, pkg: Package, index, rid: int) -> tuple[list[int], list[str]]:
    r = index.record(rid)
    in_pkg = set(pkg.rids)
    chain = [b for b in _chain_parents(pkg, rid) if 0 <= b < len(index.records)]
    # a hop-derived bridge needs a reason: the model listed it (or its file), the question reads like a
    # chain ("fixed", "logged", ...), or it names the bridge file's kind; parents beyond a dropped link go too
    listed_files = {index.record(b).file for b in out.bridge if 0 <= b < len(index.records)}
    chain_q = bool(_CHAIN_VERB_RE.search(q.text))
    justified = []
    for b in chain:
        brec = index.record(b)
        if not (chain_q or b in out.bridge or brec.file in listed_files
                or kind_matches(index.file_of(b).kind, q.named_kinds)):
            break
        justified.append(b)
    chain = justified
    cand = chain + [b for b in out.bridge if b in in_pkg and b not in chain and b != rid]
    kept: list[int] = []
    changed = True
    while changed:  # fixpoint: a bridge may attach to r or to an already kept bridge
        changed = False
        for b in cand:
            if b in kept or index.record(b).file == r.file:
                continue
            if b in chain or any(shared_rare_ids(index, b, x) for x in [rid, *kept]):
                kept.append(b)
                changed = True
    order = {b: i for i, b in enumerate(cand)}
    kept.sort(key=order.__getitem__)
    premise = sorted({index.record(b).file for b in kept
                      if kind_matches(index.file_of(b).kind, q.named_kinds)})
    return kept, premise


# ---------------------------------------------------------------- main

_ANSWER_RANGE_RE = re.compile(r"\bbetween\b|\d\s*(?:–|—|\s-\s|\s+to\s+)\s*\d", re.I)
_BOUND_Q_RE = re.compile(r"(?<![/\w])(?:min(?:imum)?|max(?:imum)?|lower|upper|floor|ceiling|lowest|highest|at least|at most)\b", re.I)  # not L/min


def _in_range_expr(text: str, span: str) -> bool:
    """The span is one end of a range in the record ("between 8 and 14 L/min", "2400–6000 RPM", "8 to 14")."""
    s = re.escape(span)
    num = r"\d[\d.,]*"
    pats = (rf"\bbetween\s+{s}\s+and\s+{num}", rf"\bbetween\s+{num}\s+and\s+{s}(?![\d.])",
            rf"(?<![\d.]){s}\s*(?:–|—|\s-\s|\s+to\s+)\s*{num}", rf"{num}\s*(?:–|—|\s-\s|\s+to\s+)\s*{s}(?![\d.])")
    return any(re.search(p, text, re.I) for p in pats)


def verify(q: Q, out: ReaderOut, pkg: Package, index) -> Verdict:
    if not str(out.answer).strip():
        return Verdict(False, "EMPTY")
    if _ANSWER_RANGE_RE.search(str(out.answer)) and not _BOUND_Q_RE.search(q.text):
        return Verdict(False, "RANGE")   # "between 2400 and 6000 RPM" is not a value
    pv = premise_check(q, index)
    if pv:
        return pv

    support = [x for x in out.support if x in set(pkg.rids) and 0 <= x < len(index.records)]
    ordered = list(dict.fromkeys(support + [x for x in pkg.rids if 0 <= x < len(index.records)]))
    spans: dict[int, str] = {}
    for rid in ordered:
        span = snap.find(out.answer, index.record(rid).text)
        if span:
            spans[rid] = span
    if not spans:   # OCR/VLM joins ("BOARD REVISIONREV-F7") only when no strict span exists anywhere
        for rid in ordered:
            rec = index.record(rid)
            if rec.src in ("ocr", "vlm"):
                span = snap.find(out.answer, rec.text, loose=True)
                if span:
                    spans[rid] = span
    if not spans:
        return Verdict(False, "NO_SPAN", violators=list(out.support))

    kept = []
    for rid in spans:
        rec, fe = index.record(rid), index.file_of(rid)
        if rec.meta or fe.status != "ok":
            continue
        if not q.wants_old and (not fe.eligible or rec.stale):
            continue
        if _code_comment_loses(index, rid):
            continue
        kept.append(rid)
    if not kept:
        return Verdict(False, "INELIGIBLE_ONLY", violators=list(spans))

    fails: dict[int, str] = {}
    scored = []
    for pos, rid in enumerate(kept):
        code = _check(q, index, rid, spans[rid])
        if code:
            fails[rid] = code
            continue
        fe = index.file_of(rid)
        # a record reached by a rare-ID hop is the chain's end: it beats a direct decoy that merely shares words
        scored.append((_attr_score(q, index.record(rid).text),
                       1 if rid in pkg.via else 0,
                       1 if (q.type_hint and kind_matches(fe.kind, q.type_hint)) else 0,
                       round(coverage(q, index, rid), 2), _authority(fe),
                       1 if rid in out.support else 0, -pos, rid))
    if not scored:
        code = Counter(fails.values()).most_common(1)[0][0]
        return Verdict(False, code, violators=list(kept))
    score, _hop, _kind, _cov, _auth, _hint, _pos, best = max(scored)
    if not textnorm.is_identifier(spans[best]) and not _BOUND_Q_RE.search(q.text) \
            and _in_range_expr(index.record(best).text, spans[best]):
        return Verdict(False, "RANGE", violators=[best])   # one end of "between 8 and 14" is not the value
    if not q.wants_old and removed_in_current(q, index, best):
        return Verdict(False, "ONLY_IN_WITHDRAWN", violators=[best], hard=True)

    bridges, premise = _bridges(q, out, pkg, index, best)
    return Verdict(True, "OK", answer_span=spans[best], support_rid=best, bridge_rids=bridges,
                   grounding_score=score, violators=[], premise_files=premise)


def answer_text(verdict: Verdict, q: Q | None = None) -> str:
    """Final answer string: bare number for quantities, identifiers verbatim."""
    span = verdict.answer_span
    if not span:
        return ""
    return snap.clean(span, is_identifier=True if textnorm.is_identifier(span) else None)
