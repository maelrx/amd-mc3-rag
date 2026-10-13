"""Text normalisation shared by index, retrieval, verification and snapping. Stdlib only."""

from __future__ import annotations

import re

STOPWORDS = frozenset("""
a an the of in on at by to for from with without into onto about as is are was were be been being
am do does default did has have had what which who whom whose when where why how this that these those it
its and or but if then than there their they them you your we our can will would should could may
might also any all per so such i me my
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-._/#][a-z0-9#]+)*#?")
_PART_SPLIT_RE = re.compile(r"[-._/]")
_DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212"})


def fold_dashes(s: str) -> str:
    """Unicode hyphens/dashes/minus -> "-" (one char for one, so offsets are preserved)."""
    return str(s).translate(_DASHES)


def norm(s: str) -> str:
    """Official answer normalisation (identical to eval/score.py:norm)."""
    return re.sub(r"[\s\-.·_]", "", str(s)).upper()


def nid(s: str) -> str:
    """Dict key of the ID index."""
    return norm(s)


def _keep(tok: str) -> bool:
    if tok in STOPWORDS:
        return False
    return len(tok) > 1 or tok.isdigit()


def stem(tok: str) -> str:
    """Light stemmer for alphabetic tokens of >= 5 chars: one of ing / ed / es / s, remainder >= 4 chars.

    "es" is stripped only after s/x/z/ch/sh ("passes", "switches") so that "values" and "value" meet;
    "s" is not stripped after another "s" ("process")."""
    if len(tok) < 5 or not tok.isalpha():
        return tok
    for suf in ("ing", "ed", "es", "s"):
        if not tok.endswith(suf) or len(tok) - len(suf) < 4:
            continue
        rest = tok[: -len(suf)]
        if suf == "es" and not rest.endswith(("ss", "x", "z", "ch", "sh")):
            continue
        if suf == "s" and rest.endswith("s"):
            continue
        return rest
    return tok


def tokens(s: str) -> list[str]:
    """BM25 tokens: whole ID, its parts and its normalised form; duplicates are kept (tf)."""
    out: list[str] = []
    for whole in _TOKEN_RE.findall(fold_dashes(s).lower()):
        base = whole.rstrip("#")
        forms = [whole, base, *_PART_SPLIT_RE.split(base), norm(base).lower()]
        seen: set[str] = set()
        for f in forms:
            f = stem(f)
            if f and f not in seen and _keep(f):
                seen.add(f)
                out.append(f)
    return out


# ---------------------------------------------------------------- identifiers

_ALNUM_ID_RE = re.compile(
    r"(?<![A-Za-z0-9#_.\-])(?=[A-Za-z0-9#_.\-]*\d)(?=[A-Za-z0-9#_.\-]*[A-Za-z])"
    r"[A-Za-z0-9][A-Za-z0-9#_.\-]{1,30}(?![A-Za-z0-9#_.\-])")
_VERSION_RE = re.compile(r"(?<![A-Za-z0-9#_.\-])\d+(?:\.\d+){1,3}(?![A-Za-z0-9#_])(?!\.\d)")
_QUARTER_RE = re.compile(r"(?<![A-Za-z0-9])Q[1-4]\s*FY\s*\d{2,4}(?!\d)")
_SIGNAL_RE = re.compile(r"(?<![A-Za-z0-9#_.\-])[A-Z][A-Z0-9_]{2,}#")
_ID_PARTS = (_QUARTER_RE, _SIGNAL_RE, _ALNUM_ID_RE, _VERSION_RE)
ID_RE = re.compile("|".join(f"(?:{r.pattern})" for r in _ID_PARTS))

_TIME_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{4}-\d{2}-\d{2}[T ])?\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?")
_ISO_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_ID_TRAIL = ".,;:)"


def ids(s: str) -> list[str]:
    """Raw identifiers in order of appearance, deduplicated."""
    s = _TIME_RE.sub(lambda m: " " * len(m.group()), fold_dashes(s))  # times/datetimes are not IDs
    found = sorted(((m.start(), m.end()) for r in _ID_PARTS for m in r.finditer(s)),
                   key=lambda se: (se[0], se[0] - se[1]))
    out: list[str] = []
    seen: set[str] = set()
    last_end = -1
    for start, end in found:
        if start < last_end:  # overlapped by a longer ID already accepted
            continue
        last_end = end
        raw = s[start:end].rstrip(_ID_TRAIL)
        if not raw or _ISO_DATE_PREFIX_RE.match(raw) or raw.isdigit() or raw in seen:
            continue
        seen.add(raw)
        out.append(raw)
    return out


_QUARTER_FULL_RE = re.compile(r"Q[1-4]\s*FY\s*\d{2,4}", re.I)
_DOTTED_RE = re.compile(r"\d+(?:\.\d+){2,3}")
_CURRENCY_RE = re.compile(r"^(?:R\$|USD|EUR|[$€£])\s*", re.I)
_QTY_UNITS = (
    r"°\s*[CF]|[CF]|(?i:degrees?|deg|secs?|seconds?)|s|ms|W|V|A|mA|mm|cm|m|kg|g|GiB|GB|MiB|MB|TiB|TB/s|TB|GB/s|Gbps"
    r"|kHz|MHz|GHz|Hz|RPM|L/min|(?i:rpm|units?|pcs|days?|hours?)|h|(?i:hrs|weeks?|months?)")
UNIT_RE = re.compile(rf"\s*(?:{_QTY_UNITS})$")
QTY_RE = re.compile(rf"^([+-]?\d[\d,]*(?:\.\d+)?)(?:\s*(?:{_QTY_UNITS}))?$")


# a pure number followed by a short unit-like word ("100 kHz", "85 µs", "2.2 kΩ", "3 years", "95 °C max"):
# blank before the unit (or none before µ/°/Ω); never an identifier (a letter before the digits, "B14", "4.3.2")
_UNIT_TOKEN = r"[A-Za-zµμΩ°%]{1,5}"
_QUALIFIER = r"(?i:max|min|typ|typical|nominal|nom)\.?"
_NUM_TAIL_RE = re.compile(
    rf"^([+-]?\d[\d,]*(?:\.\d+)?)(?:\s+|(?=[µμ°Ω]))({_UNIT_TOKEN})(?:\s+{_QUALIFIER})?$")


def qty_value(s: str) -> str | None:
    """The bare number of a quantity ("94 C", "100 kHz", "3 years", "95 °C max", "42"), else None."""
    s = fold_dashes(str(s)).strip()
    m = QTY_RE.match(s)
    if m:
        return m.group(1).rstrip(",")
    m = _NUM_TAIL_RE.match(s)
    if m and any(c.isalpha() for c in m.group(2)):
        return m.group(1).rstrip(",")
    return None


def is_identifier(s: str) -> bool:
    """True for codes, versions and quarters; False for plain numbers, with or without unit/currency."""
    t = fold_dashes(str(s)).strip().strip("\"'").rstrip(".,;:)")
    if _QUARTER_FULL_RE.fullmatch(t):
        return True
    if qty_value(_CURRENCY_RE.sub("", t).rstrip(".,;:")) is not None:
        return False
    return bool(re.search(r"[A-Za-z]", t)) or bool(_DOTTED_RE.fullmatch(t))


_NUM_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")


def numbers(s: str) -> list[str]:
    """Numeric literals as written (commas kept); compare them through num_key."""
    return _NUM_RE.findall(str(s))


def num_key(s: str) -> str:
    """Comparison form of a number: no thousands commas."""
    return str(s).replace(",", "")


# ---------------------------------------------------------------- span search

_SEP_RE = re.compile(r"[^\s;|=]+")
_OPEN = "\"'“‘([{<"
_CLOSE = "\"'”’)]}>.,;:!?"
_JOIN = ".-/_"
_MAX_WINDOW = 6


def _trim(hay: str, start: int, end: int) -> tuple[int, int]:
    """Offsets of hay[start:end] without surrounding quotes, brackets and sentence punctuation."""
    while start < end and hay[start] in _OPEN:
        start += 1
    while end > start and hay[end - 1] in _CLOSE:
        end -= 1
    return start, end


def _bounded(hay: str, start: int, end: int) -> bool:
    """No alphanumeric (or identifier joiner followed by one) glued to either side of the span."""
    if start > 0:
        c = hay[start - 1]
        if c.isalnum() or c in "#_" or (c in _JOIN and start > 1 and hay[start - 2].isalnum()):
            return False
    if end < len(hay):
        c = hay[end]
        if c.isalnum() or c in "#_" or (c in _JOIN and end + 1 < len(hay) and hay[end + 1].isalnum()):
            return False
    return True


def _windows(hay: str):
    """Yield (start, end) of every window of 1..6 consecutive tokens."""
    toks = [(m.start(), m.end()) for m in _SEP_RE.finditer(hay)]
    for i, (s, e0) in enumerate(toks):
        if not norm(hay[s:e0]):  # pure punctuation ("...") cannot open a window
            continue
        for j in range(i, min(i + _MAX_WINDOW, len(toks))):
            if norm(hay[toks[j][0]:toks[j][1]]):
                yield s, toks[j][1]


def _needles(needle: str) -> set[str]:
    return {norm(needle), norm(needle.replace(" ", ""))} - {""}


_BARE_NUM_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
# a unit glued to digits ("96C", "770W", "2.4TB/s"); regex backtracking tries the longer unit alternatives
_GLUED_UNIT_RE = re.compile(rf"(?:{_QTY_UNITS})(?![A-Za-z0-9#_]|[.\-/_][A-Za-z0-9])")


def _left_bounded(hay: str, start: int) -> bool:
    if start == 0:
        return True
    c = hay[start - 1]
    return not (c.isalnum() or c in "#_" or (c in _JOIN and start > 1 and hay[start - 2].isalnum()))


def _glued_number(hay: str, needle: str) -> tuple[int, int] | None:
    """A bare number directly followed by a unit token: the span is the number only."""
    for m in re.finditer(re.escape(needle), hay):
        a, b = m.start(), m.end()
        if b < len(hay) and _left_bounded(hay, a) and _GLUED_UNIT_RE.match(hay, b):
            return a, b
    return None


_VERSION_NEEDLE_RE = re.compile(r"\d+(?:\.\d+){1,3}")


def _right_bounded(hay: str, end: int) -> bool:
    if end >= len(hay):
        return True
    c = hay[end]
    return not (c.isalnum() or c in "#_" or (c in _JOIN and end + 1 < len(hay) and hay[end + 1].isalnum()))


def _v_version(hay: str, needle: str) -> tuple[int, int] | None:
    """A dotted version written with a glued leading v ("v6.9.1"): the span is the digits only."""
    for m in re.finditer(r"[vV](" + re.escape(needle) + ")", hay):
        if _left_bounded(hay, m.start()) and _right_bounded(hay, m.end()):
            return m.start(1), m.end(1)
    return None


def find_span(hay: str, needle: str) -> tuple[int, int] | None:
    """Char offsets of the first token-aligned occurrence of needle in hay, else None."""
    hay = fold_dashes(hay)
    needle = fold_dashes(needle).strip()
    keys = _needles(needle)
    if not keys:
        return None
    for s, e in _windows(hay):
        for a, b in (_trim(hay, s, e), (s, e)):
            if a < b and norm(hay[a:b]) in keys and _bounded(hay, a, b):
                return a, b
    for m in re.finditer(re.escape(needle), hay, re.I):
        if _bounded(hay, m.start(), m.end()):
            return m.start(), m.end()
    if _VERSION_NEEDLE_RE.fullmatch(needle):
        hit = _v_version(hay, needle)
        if hit:
            return hit
    if _BARE_NUM_RE.fullmatch(needle):
        return _glued_number(hay, needle)
    return None


def _id_like(needle: str) -> bool:
    return "-" in needle and any(c.isalpha() for c in needle) and any(c.isdigit() for c in needle)


def find_span_loose(hay: str, needle: str) -> tuple[int, int] | None:
    """find_span, plus OCR joins: a single token that ends with the needle after a prefix.

    The prefix is <= 3 chars; for an ID-shaped needle (letter, digit and "-") any run of letters
    ("REVISIONREV-F7"). A digit-only needle never gets a prefix ("J25" is not 25)."""
    hit = find_span(hay, needle)
    if hit:
        return hit
    hay = fold_dashes(hay)
    needle = fold_dashes(needle).strip()
    if _BARE_NUM_RE.fullmatch(needle):
        return None
    keys = _needles(needle)
    if not keys:
        return None
    idlike = _id_like(needle)
    for m in _SEP_RE.finditer(hay):
        s, e = _trim(hay, m.start(), m.end())
        tok = hay[s:e]
        for k in range(1, len(tok)):
            if norm(tok[k:]) not in keys:
                continue
            pre = tok[:k]
            if 0 < len(norm(pre)) <= 3 or (idlike and pre.isalpha()):
                return s + k, e
    return None


def word_boundary_contains(hay: str, needle: str) -> bool:
    """True when needle occurs in hay with non-alphanumeric boundaries."""
    return find_span(hay, needle) is not None
