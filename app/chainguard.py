"""Named-kind chain guard (H2, behind MC3_CHAIN_GUARD). Stdlib only.

Failure it prevents: "The production log shows a thermal throttle incident. Which firmware release fixed the
underlying defect?" cites [log, csv]. The log line names the ticket, the csv row of that ticket holds the fix.
A csv decoy whose summary matches the topic ("Thermal throttle engages early ...") can win the read; the answer is
then a topical guess and the citation misses the log.

``apply`` runs after the final verified verdict. When the question names a file kind (the log), reads like a chain,
and neither the support nor any kept bridge comes from that kind, it re-reads a package restricted to the chain
(named-kind parent records + their hop targets). The new answer is accepted only if it verifies and bridges to a
named-kind file; otherwise the original is kept. It never refuses.
"""

from __future__ import annotations

import re

import config
import retrieve
import verify
from schema import Package

_CHAIN_PHRASE_RE = re.compile(r"\b(shows?|showed|incident|underlying)\b", re.I)
_IMAGE_SRC = ("ocr", "vlm")
REREAD_MIN_REMAINING_S = 6.0
_PARENT_NEIGHBORS = 3       # records of the parent's file around it that the package already holds


def _chain_like(q) -> bool:
    return bool(retrieve._CHAIN_VERB_RE.search(q.text) or _CHAIN_PHRASE_RE.search(q.text))


def _named(index, rid: int, q) -> bool:
    return verify.kind_matches(index.file_of(rid).kind, q.named_kinds)


def _targets(q, pkg: Package, index, sup: int) -> dict[int, int]:
    """Hop targets of the package (target rid -> parent rid) whose parent comes from a named-kind file."""
    held = set(pkg.rids)
    out = {}
    for tgt, parent in pkg.via.items():
        if tgt not in held or parent not in held or tgt == sup:
            continue
        trec = index.record(tgt)
        if trec.meta or trec.src in _IMAGE_SRC or index.record(parent).src in _IMAGE_SRC:
            continue
        if not _named(index, parent, q) or _named(index, tgt, q):
            continue
        if not q.wants_old and (not index.eligible(tgt) or not index.eligible(parent)):
            continue
        out[tgt] = parent
    return out


def restricted_package(index, pkg: Package, targets: dict[int, int]) -> Package:
    """Parents (with a few neighbours the package already holds) + targets, with file headers and the via map."""
    held = set(pkg.rids)
    keep = set(targets) | set(targets.values())
    for parent in targets.values():
        rids = index.by_file.get(index.record(parent).file, [])
        if parent in rids:
            pos = rids.index(parent)
            for p in range(max(0, pos - _PARENT_NEIGHBORS), min(len(rids), pos + _PARENT_NEIGHBORS + 1)):
                if rids[p] in held and not index.record(rids[p]).meta:
                    keep.add(rids[p])
    order = {f: i for i, f in enumerate(pkg.files)}
    rids = sorted(keep, key=lambda r: (order.get(index.record(r).file, len(order)), r))
    parts, cur = [], None
    for rid in rids:
        rec = index.record(rid)
        if rec.file != cur:
            cur = rec.file
            parts.append(f"## {cur} ({index.file_of(rid).kind})")
        parts.append(retrieve._line(rec))
    text = "\n".join(parts)
    files = list(dict.fromkeys(index.record(r).file for r in rids))
    via = {t: p for t, p in targets.items() if t in keep and p in keep}
    return Package(rids=rids, text=text, images=[], token_est=int(len(text) / config.CHARS_PER_TOKEN),
                   files=files, via=via)


def apply(q, v, out, pkg: Package, index, reader, query: str, deadline, trace):
    """None = keep the original answer, else (verdict, reader_out, package, images) of the accepted re-read."""
    try:
        if not (v.ok and v.support_rid is not None and q.named_kinds and pkg is not None and _chain_like(q)):
            return None
        sup = v.support_rid
        if _named(index, sup, q) or any(_named(index, b, q) for b in v.bridge_rids):
            return None
        targets = _targets(q, pkg, index, sup)
        if not targets:
            trace.note("chain_guard", "kept")
            trace.note("chain_guard_reason", "no_named_kind_hop_target")
            return None
        if not deadline.require(REREAD_MIN_REMAINING_S):
            trace.note("chain_guard", "kept")
            trace.note("chain_guard_reason", "no_time")
            return None
        pkg2 = restricted_package(index, pkg, targets)
        trace.note("chain_guard", "fired")
        rd = deadline.sub(1.0)
        if rd.expired():
            trace.note("chain_guard", "kept")
            trace.note("chain_guard_reason", "no_time")
            return None
        out2 = reader.read(pkg2.text, query, [], rd)
        v2 = verify.verify(q, out2, pkg2, index)
        trace.note("chain_guard_reread", v2.reason)
        if v2.ok and any(_named(index, b, q) for b in v2.bridge_rids):
            trace.note("chain_guard", "replaced")
            trace.note("chain_guard_reason", f"support R{sup} -> R{v2.support_rid}")
            return v2, out2, pkg2, []
        trace.note("chain_guard", "kept")
        trace.note("chain_guard_reason", "reread_not_verified" if not v2.ok else "reread_no_named_bridge")
        return None
    except Exception as e:  # noqa: BLE001 - the guard is optional: any error keeps the verified answer
        trace.note("chain_guard", "kept")
        trace.note("chain_guard_reason", f"error:{type(e).__name__}")
        return None
