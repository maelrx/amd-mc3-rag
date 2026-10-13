"""Deterministic last-resort answer when the daemon gave nothing usable (spec 5.12). Stdlib only.

One rare ID in the question (df <= 2) plus one attribute word (fixed/price/lead/part/...) is enough to read a
``key=value`` pair straight from the single eligible record that carries the ID. Anything ambiguous refuses.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import index
import snap
import textnorm

RARE_DF_FALLBACK = 2

# question word prefix -> substrings that identify the matching key (lower-case)
LEXICON: dict[str, tuple[str, ...]] = {
    "fixed": ("fixed_in", "version", "release", "firmware"),
    "firmware": ("fixed_in", "version", "release", "firmware"),
    "version": ("fixed_in", "version", "release", "firmware"),
    "release": ("fixed_in", "version", "release", "firmware"),
    "cost": ("cost", "price", "usd"),
    "price": ("cost", "price", "usd"),
    "lead": ("lead",),
    "part": ("part",),
    "assembly": ("part",),
    "status": ("status",),
    "severity": ("severity",),
    "owner": ("owner",),
}

_WORD_RE = re.compile(r"[a-z]+")


def _wanted_key_stems(query: str) -> set[str]:
    stems: set[str] = set()
    for w in _WORD_RE.findall(query.lower()):
        for prefix, keys in LEXICON.items():
            if w.startswith(prefix):
                stems.update(keys)
    return stems


def _pairs(text: str):
    for part in text.split("; "):
        key, sep, value = part.partition("=")
        if sep and key.strip() and value.strip():
            yield key.strip(), value.strip()


def _expired(deadline_ts: float | None) -> bool:
    return deadline_ts is not None and time.time() > deadline_ts


def _lookup(index_dir: Path, query: str, deadline_ts: float | None) -> tuple[str, list[str]]:
    idx = index.load(Path(index_dir))
    if _expired(deadline_ts):
        return "", []
    rare = [raw for raw in dict.fromkeys(textnorm.ids(query))
            if idx.ids.get(textnorm.nid(raw), {}).get("df", 99) <= RARE_DF_FALLBACK]
    if len({textnorm.nid(r) for r in rare}) != 1:
        return "", []
    stems = _wanted_key_stems(query)
    if not stems:
        return "", []
    hits: list[tuple[int, str, str]] = []
    for rid in idx.ids[textnorm.nid(rare[0])]["rids"]:
        if _expired(deadline_ts):
            return "", []
        rec = idx.record(rid)
        if rec.meta or not idx.eligible(rid):
            continue
        for key, value in _pairs(rec.text):
            if any(s in key.lower() for s in stems):
                hits.append((rid, key, value))
    if len(hits) != 1:
        return "", []
    rid, _, value = hits[0]
    return snap.clean(value), [idx.record(rid).file]


def lookup(index_dir: Path, query: str, deadline_ts: float | None = None) -> tuple[str, list[str]]:
    """``(answer, citations)`` from the on-disk index, or ``("", [])``. Never raises."""
    try:
        return _lookup(index_dir, query, deadline_ts)
    except Exception:  # noqa: BLE001 - the fallback must never be the thing that crashes
        return "", []
