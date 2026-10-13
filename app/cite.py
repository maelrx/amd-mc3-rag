"""Citation set from the verified support record and bridges. Stdlib only."""

from __future__ import annotations

import config
import textnorm
from schema import Package, Q, Verdict
from verify import shared_rare_ids

_PREFIXES = ("/app/corpus/", "./")


def _rel(path: str) -> str:
    p = str(path)
    for pre in _PREFIXES:
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def loo_select(candidates, reader, pkg, q, deadline, answer: str = ""):
    """v2 hook: among tied support rids keep the one whose removal hurts logP(answer) most.

    Returns candidates unchanged unless there is a reader, an answer, >1 candidates and enough time.
    """
    if reader is None or pkg is None or not answer or len(candidates) < 2:
        return candidates
    if deadline is None or not deadline.require(config.LOO_MIN_REMAINING_S):
        return candidates
    try:
        scored = []
        for keep in candidates:
            drop = {f"[R{c}]" for c in candidates if c != keep}
            variant = "\n".join(ln for ln in pkg.text.splitlines()
                                if not any(ln.startswith(d) for d in drop))
            scored.append((reader.logprob(variant, q.text, [], answer), keep))
        best = max(scored, key=lambda t: t[0])[1]
        return [best]
    except Exception:
        return candidates


def cite(q: Q, verdict: Verdict, index, reader=None, pkg: Package | None = None, deadline=None) -> list[str]:
    if not verdict.ok or verdict.support_rid is None:
        return []
    sup = verdict.support_rid
    value_file = index.record(sup).file
    q_ids = {textnorm.nid(x) for x in q.ids}
    chain = set()
    if pkg is not None:
        cur, seen = pkg.via.get(sup), {sup}
        while cur is not None and cur not in seen:
            chain.add(cur)
            seen.add(cur)
            cur = pkg.via.get(cur)
    links = [sup, *verdict.bridge_rids]
    files = {value_file}
    for b in verdict.bridge_rids:
        bfile = index.record(b).file
        if bfile == value_file:
            continue
        shared = {n for o in links if o != b for n in shared_rare_ids(index, b, o)}
        if shared:
            if shared <= q_ids:     # the question already names the linking id (Q4): no bridge needed
                continue
        elif not (b in chain or bfile in verdict.premise_files or pkg is None):
            continue
        files.add(bfile)
    return sorted({_rel(f) for f in files})
