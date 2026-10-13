"""Question analysis, retrieval (BM25 + ID hits + RRF), ID hops and evidence packaging. Stdlib only."""

from __future__ import annotations

import re
from dataclasses import replace

import config
import index as index_mod
import textnorm
from schema import FileHit, Package, Q

_NUM_UNIT_RE = re.compile(r"^[+-]?\d[\d.,]*[A-Za-z%°]{1,4}$")   # 35C, 5V, 12V, 90W
_FILENAME_RE = re.compile(r"\.[A-Za-z0-9]{2,5}$")              # roadmap_fy27.docx


_PREMISE_RES = (
    re.compile(r"^[A-Za-z]{1,5}-\d{2,6}[A-Za-z]?$"),            # TQ-40, ORR-1847, HLX-2041, KX-72
    re.compile(r"^[A-Za-z]{1,2}\d{3,6}$"),                      # E7731
    re.compile(r"^[A-Za-z]{2,5}(?:-[A-Za-z0-9]{1,6}){2,4}$"),    # ORR-FAN-2214-B (>= 3 digits)
)


def _premise_worthy(raw: str) -> bool:
    """Only entity-shaped codes (tickets, error codes, part numbers, products) are checked for existence
    and the index must be able to hold them; x16, RFC2119, JESD51-2, IEEE802.3, 21-Sep, HBM3e, Q3 FY29,
    file names or 12V/96C would otherwise be 'absent' and refuse an answerable question."""
    if not index_mod._indexable(raw) or _NUM_UNIT_RE.match(raw) or _FILENAME_RE.search(raw):
        return False
    if _PREMISE_RES[0].match(raw) or _PREMISE_RES[1].match(raw):
        return True
    return bool(_PREMISE_RES[2].match(raw)) and sum(c.isdigit() for c in raw) >= 3


# ---------------------------------------------------------------- analyze

_PRODUCT_RE = re.compile(r"^[A-Za-z]{1,4}-?\d{1,4}[A-Za-z]?$")
_PLAIN_DECIMAL_RE = re.compile(r"^\d+\.\d+$")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?Z?)?")
_QUARTER_RE = re.compile(r"\bQ[1-4]\s*FY\s*\d{2,4}\b", re.I)

_UNIT_RE = re.compile(
    r"\bin\s+(?:degrees?\s+)?(seconds|ms|milliseconds|minutes|hours|watts|volts|amps|celsius|fahrenheit|kelvin"
    r"|mm|cm|percent|units|days|weeks|months"
    r"|tb/s|gb/s|mb/s|gbps|gigabytes|gibibytes|megabytes|mebibytes|terabytes|tebibytes|(?:gb|gib|mb|mib|tb|tib)(?!/)|rpm"
    r"|hertz|kilohertz|megahertz|gigahertz|(?:khz|mhz|ghz|hz)(?!/)|l/min|liters per minute|litres per minute)\b"
    r"|(°\s*[CF])"
    r"|\bdegrees?\s+(celsius|fahrenheit|kelvin)\b", re.I)
_UNIT_CANON = {"seconds": "s", "milliseconds": "ms", "ms": "ms", "minutes": "min", "hours": "h",
               "watts": "W", "volts": "V", "amps": "A", "celsius": "C", "fahrenheit": "F", "kelvin": "K",
               "mm": "mm", "cm": "cm", "percent": "%", "units": "units", "days": "days", "weeks": "weeks",
               "months": "months",
               "gigabytes": "GB", "gb": "GB", "gibibytes": "GiB", "gib": "GiB", "megabytes": "MB", "mb": "MB",
               "mebibytes": "MiB", "mib": "MiB", "terabytes": "TB", "tb": "TB", "tebibytes": "TiB", "tib": "TiB",
               "rpm": "RPM", "hertz": "Hz", "hz": "Hz", "kilohertz": "kHz", "khz": "kHz", "megahertz": "MHz",
               "mhz": "MHz", "gigahertz": "GHz", "ghz": "GHz", "l/min": "L/min", "liters per minute": "L/min",
               "litres per minute": "L/min", "tb/s": "TB/s", "gb/s": "GB/s", "mb/s": "MB/s", "gbps": "Gbps"}
_AGG_RE = re.compile(r"\b(?:combined|total|sum|altogether|in total|average|per|each|aggregate)\b", re.I)
# spelled-out counts are conditions too ("combined power of four boards"); "four-pin" is excluded
_WORD_NUM_RE = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|fifty)\b(?!-)", re.I)
_WORD_NUMS = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
              "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20",
              "fifty": "50"}

_ATTR_LEXICON = (
    ("price", r"\b(?:unit price|price|cost)\b"),
    ("temperature", r"\b(?:temperature|junction|thermal)\b"),
    ("timeout", r"\btimeout\b"),
    ("version", r"\b(?:firmware|version|release|fixed)\b"),
    ("part_number", r"\b(?:part number|part no|p/n)\b"),
    ("pin", r"\b(?:pin|carries)\b"),
    ("revision", r"\b(?:revision|rev)\b"),
    ("error_code", r"\b(?:error code|logged)\b"),
    ("schedule", r"\b(?:quarter|sampling|ships|launch)\b"),
    ("lead_time", r"\blead time\b"),
    ("serial", r"\bserial\b"),
    ("power", r"\b(?:power|wattage)\b"),
    ("voltage", r"\bvoltage\b"),
    ("weight", r"\bweight\b"),
    ("speed", r"\b(?:speed|rpm|frequency)\b"),
)
_ATTR_RES = tuple((k, re.compile(p, re.I)) for k, p in _ATTR_LEXICON)

_TYPE_LEXICON = (
    ("log", r"\b(?:logs?|logged)\b"),
    ("pdf", r"\b(?:datasheets?|specs?|specifications?)\b"),
    ("image", r"\b(?:labels?|printed|photos?|pictures?|diagrams?|pinouts?|pins?|images?|schematics?)\b"),
    ("xlsx", r"\b(?:spreadsheets?|sheets?|parts list|rma|cost|lead time|inventory)\b"),
    ("py", r"\b(?:source code|python|constants?|defaults?|service|functions?)\b|\.py\b"),
    ("docx", r"\b(?:roadmaps?|memos?|plans?|documents?|docx)\b"),
    ("csv", r"\b(?:tickets?|bug database|bugs?|trackers?)\b"),
    ("txt", r"\b(?:release notes|changelog|faq|readme)\b"),
)
_TYPE_RES = tuple((k, re.compile(p, re.I)) for k, p in _TYPE_LEXICON)

_WANTS_OLD_RE = re.compile(
    r"\b(?:withdrawn|superseded|previous(?:ly)?|old(?:er|est)?|original(?:ly)?|revision\s*1|r1"
    r"|earlier|deprecated|obsolete)\b", re.I)
_CHAIN_VERB_RE = re.compile(r"\b(fixed|caused|reported|logged|replaced|ordered|resolved|closed)\b", re.I)

# type_hint -> file kinds it matches
_HINT_KINDS = {"log": {"log"}, "txt": {"txt", "md"}, "image": {"image"}, "py": {"py"},
               "docx": {"docx"}, "xlsx": {"xlsx"}, "csv": {"csv"}, "pdf": {"pdf"}}


def _blank(text: str, raw: str) -> str:
    pat = r"(?<![A-Za-z0-9#_.\-])" + re.escape(raw) + r"(?![A-Za-z0-9#_])"
    return re.sub(pat, lambda m: " " * len(m.group()), text)


def analyze(query: str, index=None) -> Q:
    """Question -> Q (ids, rare ids, entities, numbers, unit, attribute terms, type hint, wants_old)."""
    query = str(query)
    all_ids = textnorm.ids(query)
    ids = [i for i in all_ids if not _PLAIN_DECIMAL_RE.match(i)]
    if index is None:
        rare = [i for i in ids if _premise_worthy(i)]
    else:
        rare = []
        for i in ids:
            if not _premise_worthy(i):
                continue
            e = index.ids.get(textnorm.nid(i))
            if e is None or e["df"] <= config.RARE_DF:
                rare.append(i)
    entities = [i for i in ids if _PRODUCT_RE.match(i)]
    glued = [m.group(1) for i in ids for m in [re.match(r"^[+-]?(\d[\d.,]*)[A-Za-z%°]{1,4}$", i)] if m]

    stripped = _ISO_DATE_RE.sub(lambda m: " " * len(m.group()), textnorm.fold_dashes(query))
    stripped = _QUARTER_RE.sub(lambda m: " " * len(m.group()), stripped)
    for raw in ids:
        stripped = _blank(stripped, raw)
    numbers = []
    for n in textnorm.numbers(stripped):
        n = n.rstrip(",")
        if n and not _YEAR_RE.match(n):
            numbers.append(n)
    if _AGG_RE.search(query):   # "four" is a condition only in an aggregation ("combined power of four boards")
        for m in _WORD_NUM_RE.finditer(stripped):
            numbers.append(_WORD_NUMS[m.group(1).lower()])
    numbers.extend(g.rstrip(",.") for g in glued if g.rstrip(",.") and not _YEAR_RE.match(g))

    unit = None
    m = _UNIT_RE.search(query)
    if m:
        word = m.group(1) or m.group(3)
        if word:
            unit = _UNIT_CANON[word.lower()]
        else:
            unit = m.group(2).replace(" ", "")[-1].upper()

    attrs = [k for k, rx in _ATTR_RES if rx.search(query)]
    hints = {k for k, rx in _TYPE_RES if rx.search(query)}
    return Q(text=query, ids=ids, rare_ids=rare, entities=entities, numbers=numbers, unit=unit,
             attr_terms=attrs, type_hint=hints, wants_old=bool(_WANTS_OLD_RE.search(query)),
             named_kinds=set(hints))


# ---------------------------------------------------------------- retrieve

def _kind_matches(kind: str, hints: set[str]) -> bool:
    return any(kind in _HINT_KINDS.get(h, ()) for h in hints)


def _usable(index, rid: int, q: Q) -> bool:
    """Record may enter the candidate pool: readable file, and eligible/non-stale unless wants_old."""
    if index.file_of(rid).status != "ok":
        return False
    return True if q.wants_old else index.eligible(rid)


def _rrf(lists: list[list[int]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for lst in lists:
        for rank, rid in enumerate(lst, 1):
            out[rid] = out.get(rid, 0.0) + 1.0 / (config.RRF_K + rank)
    return out


_ALIASES = {"production": ("prod",), "spec": ("specification", "datasheet"),
            "specification": ("spec", "datasheet"), "datasheet": ("spec", "specification")}


def _query_tokens(q: Q) -> list[str]:
    """BM25 query tokens: an ID contributes its whole token once, not its parts.

    textnorm.tokens("HN-450") yields hn-450, hn, 450 and hn450; letting all four score makes every
    row that names the product outrank the one line that holds the attribute. The ID index already
    rewards exact ID matches separately. File names keep their parts, and a part that also stands
    alone elsewhere in the question ("HN" in "HN series ... HN-450") stays.
    """
    parts: set[str] = set()
    blanked = textnorm.fold_dashes(q.text)
    for raw in q.ids + q.entities:
        if _FILENAME_RE.search(raw) and re.search(r"[A-Za-z]", raw.rsplit(".", 1)[1]):
            continue    # roadmap_fy27.docx: its parts are words of the question, not noise
        low = textnorm.fold_dashes(raw).lower()
        parts.update(textnorm.stem(p) for p in re.split(r"[-._/]", low) if p)
        parts.add(textnorm.nid(raw).lower())
        blanked = _blank(blanked, raw)
    parts -= set(textnorm.tokens(blanked))
    whole = {textnorm.fold_dashes(raw).lower() for raw in q.ids + q.entities}
    # product codes stay out of the BM25 query altogether: every title line of every file about
    # the product would otherwise outrank the one row holding the attribute. The ID list and the
    # entity check in verify handle the product.
    ents = {textnorm.fold_dashes(raw).lower() for raw in q.entities}
    out = []
    for t in textnorm.tokens(q.text):
        if t in ents or (t in parts and t not in whole):
            continue
        out.append(t)
        for alias in _ALIASES.get(t, ()):
            out.append(alias)
    return out


def retrieve(index, q: Q, attempt: int = 1, exclude_rids: set[int] | frozenset = frozenset()) -> list[FileHit]:
    """Files ranked by BM25 + exact-ID evidence (RRF over records, best record per file + 1/4 of the next two)."""
    top_n = config.BM25_TOP_RECORDS * (2 if attempt >= 2 else 1)
    a = [rid for rid, _ in index.bm25.search(_query_tokens(q), top_n)]

    b_scores: dict[int, float] = {}
    for raw in q.ids:
        e = index.ids.get(textnorm.nid(raw))
        if not e:
            continue
        w = 1.0 / max(1, e["df"])
        for rid in e["rids"]:
            b_scores[rid] = b_scores.get(rid, 0.0) + w
    b = [rid for rid, _ in sorted(b_scores.items(), key=lambda kv: (-kv[1], kv[0]))]

    scores = {rid: s for rid, s in _rrf([a, b]).items()
              if rid not in exclude_rids and _usable(index, rid, q)}

    per_file: dict[str, list[int]] = {}
    for rid in sorted(scores, key=lambda r: (-scores[r], r)):
        per_file.setdefault(index.record(rid).file, []).append(rid)

    max_rids = 20 if attempt >= 2 else 12   # a long log with decoy ERROR lines must keep the right line in
    hits = []
    for fname, rids in per_file.items():
        fe = index.file_by_path[fname]
        # best record counts fully, the next two only a quarter: a file with one line that matches
        # the attribute must beat a CSV whose every row merely names the product
        score = scores[rids[0]] + 0.25 * sum(scores[r] for r in rids[1:3])
        if _kind_matches(fe.kind, q.type_hint):
            score *= 1.25
        if any(index.record(r).meta for r in rids):
            score *= 1.15
        hits.append(FileHit(file=fname, score=score, best=scores[rids[0]],
                            rids=[r for r in rids if not index.record(r).meta][:max_rids]))
    hits.sort(key=lambda h: (-h.score, h.file))
    return hits


# ---------------------------------------------------------------- hop

def _chain_hops(q: Q, max_hops: int) -> int:
    limit = min(max_hops, 2)
    if max_hops >= 2 and len(_CHAIN_VERB_RE.findall(q.text)) >= 2:
        limit = 3
    return limit


def hop(index, q: Q, hits: list[FileHit], max_hops: int = config.MAX_HOPS) -> list[FileHit]:
    """Follow rare IDs from the best records into other files (score 0.9 x parent file)."""
    merged = {h.file: replace(h, rids=list(h.rids), via=dict(h.via)) for h in hits}
    ordered = sorted(merged.values(), key=lambda h: (-h.score, h.file))
    q_nids = {textnorm.nid(i) for i in q.ids}

    frontier: list[int] = []
    for h in ordered:
        frontier.extend(h.rids)
        if len(frontier) >= 8:
            break
    frontier = frontier[:8]

    ancestors: dict[int, frozenset[str]] = {rid: frozenset() for rid in frontier}
    seen_parents: set[int] = set()
    for _ in range(_chain_hops(q, max_hops)):
        nxt: list[int] = []
        for parent in frontier:
            if parent in seen_parents:
                continue
            seen_parents.add(parent)
            prec = index.record(parent)
            parent_hit = merged.get(prec.file)
            if parent_hit is None:
                continue
            chain = ancestors.get(parent, frozenset()) | {prec.file}
            for raw in textnorm.ids(prec.text):
                key = textnorm.nid(raw)
                e = index.ids.get(key)
                if key in q_nids or not e or e["df"] > config.RARE_DF:
                    continue
                # Only code-like IDs (letters + digits) bridge files: a shared version
                # or decimal such as 4.3.1 is a coincidence, not a chain link.
                if not re.search(r"[A-Za-z]", raw) or raw[0].isdigit() or re.fullmatch(r"Q[1-4]\s*FY\s*\d{2,4}", raw, re.I):
                    continue
                for rid in e["rids"][:12]:
                    rec = index.record(rid)
                    if rec.file in chain or rid in seen_parents or rec.meta:
                        continue
                    if not _usable(index, rid, q):
                        continue
                    score = 0.9 * parent_hit.score
                    tgt = merged.get(rec.file)
                    if tgt is None:
                        tgt = merged[rec.file] = FileHit(file=rec.file, score=score)
                    else:
                        tgt.score = max(tgt.score, score)
                    if rid not in tgt.via:
                        tgt.via[rid] = parent
                    if rid not in tgt.rids:
                        tgt.rids.insert(0, rid)
                    ancestors[rid] = chain
                    nxt.append(rid)
        frontier = nxt
        if not frontier:
            break
    return sorted(merged.values(), key=lambda h: (-h.score, h.file))


# ---------------------------------------------------------------- package

_MAX_LINE_CHARS = 600
_EXTRA_HOP_FILES = 2


def _line(rec) -> str:
    text = rec.text if len(rec.text) <= _MAX_LINE_CHARS else rec.text[:_MAX_LINE_CHARS - 1] + "…"
    return f"[R{rec.rid}] {rec.file} {rec.loc}: {text}"


_TOP_RECORD_FILES = 8


def _choose_files(index, hits: list[FileHit], n: int) -> list[FileHit]:
    chosen = list(hits[:n])
    names = {h.file for h in chosen}
    # the files holding the best-scoring single records always travel, even when their file
    # score is low: a two-line quick reference must not lose to a CSV by volume
    for h in sorted(hits, key=lambda h: (-h.best, h.file))[:_TOP_RECORD_FILES]:
        if h.rids and h.file not in names:
            chosen.append(h)
            names.add(h.file)
    extra = 0
    for h in hits[n:]:  # chain targets must not fall off the end because their score is discounted
        if extra >= _EXTRA_HOP_FILES:
            break
        if h.via and any(index.record(p).file in names for p in h.via.values()):
            chosen.append(h)
            extra += 1
    return chosen


def package(index, q: Q, hits: list[FileHit], attempt: int = 1,
            exclude_rids: set[int] | frozenset = frozenset()) -> Package:
    """Evidence block: top files, their best records + neighbours, in document order."""
    ai = 1 if attempt >= 2 else 0
    chosen = _choose_files(index, hits, config.PKG_FILES[ai])
    neigh = config.PKG_NEIGHBORS[ai]
    budget = config.PKG_TOKENS[ai]

    def ok(rid: int) -> bool:
        rec = index.record(rid)
        if rec.meta or rid in exclude_rids:
            return False
        return True if q.wants_old else index.eligible(rid)

    # candidate records with a priority key; lower is taken first
    prio: dict[int, tuple] = {}
    forced: set[int] = set()

    def offer(rid: int, key: tuple) -> None:
        if ok(rid) and (rid not in prio or key < prio[rid]):
            prio[rid] = key

    for fi, h in enumerate(chosen):
        rids = index.by_file.get(h.file, [])
        pos = {r: i for i, r in enumerate(rids)}
        core = [r for r in h.rids if ok(r)]
        if not core:  # file matched only through its name: lead with its first records
            core = [r for r in rids if ok(r)][:1]
        if fi < 2 and core:
            forced.add(core[0])
        for rank, r in enumerate(core):
            offer(r, (0 if (fi < 2 and rank == 0) else 1, rank, fi, r))
            for d in range(1, neigh + 1):
                for p in (pos.get(r, -99) - d, pos.get(r, -99) + d):
                    if 0 <= p < len(rids):
                        offer(rids[p], (2, d, fi, rids[p]))
        if len(rids) <= config.PKG_WHOLE_FILE_MAX_RECORDS:
            for p, r in enumerate(rids):
                offer(r, (3, fi, p, r))

    file_order = {h.file: fi for fi, h in enumerate(chosen)}
    taken: list[int] = []
    files_open: set[str] = set()
    used = 0.0
    for rid in sorted(prio, key=lambda r: prio[r]):
        rec = index.record(rid)
        cost = len(_line(rec)) + 1
        if rec.file not in files_open:
            cost += len(f"## {rec.file} ({index.file_of(rid).kind})") + 1
        if rid not in forced and (used + cost) / config.CHARS_PER_TOKEN > budget:
            continue
        taken.append(rid)
        files_open.add(rec.file)
        used += cost

    taken.sort(key=lambda r: (file_order[index.record(r).file], r))
    parts: list[str] = []
    cur = None
    for rid in taken:
        rec = index.record(rid)
        if rec.file != cur:
            cur = rec.file
            parts.append(f"## {cur} ({index.file_of(rid).kind})")
        parts.append(_line(rec))
    text = "\n".join(parts)

    # image descriptors; the pipeline loads the pixels
    images: list = []
    top_files = [h.file for h in chosen]
    for fi, fname in enumerate(top_files):
        fe = index.file_by_path[fname]
        if fe.kind != "image" or not (fi < 2 or ("image" in q.type_hint and fi < 5)):
            continue
        for desc in fe.images:
            if len(images) < config.MAX_IMAGES:
                images.append((fname, tuple(desc)))

    via: dict[int, int] = {}
    tset = set(taken)
    for h in chosen:
        via.update({r: p for r, p in h.via.items() if r in tset})
    return Package(rids=taken, text=text, images=images, token_est=int(len(text) / config.CHARS_PER_TOKEN),
                   files=top_files, via=via)


def assemble(index, q: Q, attempt: int = 1, exclude_rids: set[int] | frozenset = frozenset()) -> Package:
    """retrieve -> hop -> package."""
    hits = retrieve(index, q, attempt, exclude_rids)
    hits = hop(index, q, hits)
    return package(index, q, hits, attempt, exclude_rids)


def diagnose(index, q: Q, pkg: Package) -> str:
    """Compact multi-line summary of an assembled package (for logs and tests)."""
    got = set(pkg.rids)
    lines = [f"Q: {q.text}",
             f"   ids={q.ids} rare={q.rare_ids} ent={q.entities} nums={q.numbers} unit={q.unit} "
             f"attr={q.attr_terms} hint={sorted(q.type_hint)} old={q.wants_old}",
             f"   package: {len(pkg.rids)} records, ~{pkg.token_est} tokens, images={pkg.images}"]
    for fname in pkg.files:
        fe = index.file_by_path[fname]
        rids = [r for r in pkg.rids if index.record(r).file == fname]
        lines.append(f"   - {fname} ({fe.kind}{'' if fe.eligible else ', INELIGIBLE'}): "
                     f"{len(rids)}/{len(index.by_file.get(fname, []))} records "
                     f"R{rids[0]}..R{rids[-1]}" if rids else
                     f"   - {fname} ({fe.kind}): no text records")
    if pkg.via:
        lines.append("   via: " + ", ".join(f"R{r}<-R{p}{'' if p in got else '(absent)'}"
                                            for r, p in sorted(pkg.via.items())))
    return "\n".join(lines)
