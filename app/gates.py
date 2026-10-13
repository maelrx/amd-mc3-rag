"""Abstention gates: A (deterministic, pre/post verify) and B (log-prob, behind MC3_GATE_B). Stdlib only."""

from __future__ import annotations

import config
from schema import Q, ReaderOut, Verdict
from verify import premise_check, shared_rare_ids


def gate_a_pre(q: Q, index) -> Verdict | None:
    """Hard refusal when a rare ID of the question is absent or only in unusable files."""
    return premise_check(q, index)


def gate_a_post(q: Q, verdict: Verdict, index) -> tuple[bool, str]:
    if verdict is None or not verdict.ok or verdict.support_rid is None:
        return False, verdict.reason if verdict is not None else "EMPTY"
    sup = verdict.support_rid
    rec, fe = index.record(sup), index.file_of(sup)
    if fe.status != "ok" or (not q.wants_old and (not fe.eligible or rec.stale)):
        return False, "INELIGIBLE_ONLY"
    # every bridge must hang off the chain: reachable from the support through shared rare ids
    linked, pending = [sup], list(verdict.bridge_rids)
    progress = True
    while pending and progress:
        progress = False
        for b in list(pending):
            if any(shared_rare_ids(index, b, x) for x in linked):
                linked.append(b)
                pending.remove(b)
                progress = True
    return (False, "CHAIN_OPEN") if pending else (True, "OK")


def gate_b(out: ReaderOut, context_gain: float | None,
           theta1: float = config.GATE_B_THETA1, theta2: float = config.GATE_B_THETA2) -> bool:
    """True = answer. Both signals unavailable -> answer."""
    s1 = out.mean_logprob
    if s1 is None and context_gain is None:
        return True
    return (s1 is not None and s1 >= theta1) or (context_gain is not None and context_gain >= theta2)
