#!/usr/bin/env python3
"""Calibrate gate B (app/gates.py) thresholds from report.json files of a log-mode run (MC3_GATE_B=log).

    python3 eval/calib_gate_b.py --tune 'OUT/gateb_log/s*/report.json' ... --holdout 'OUT/gateb_log/h*/report.json'

Every answered row (got.answer != "") carries the trace notes ``gate_b_s1`` (mean log-prob of the generated answer
tokens) and ``context_gain``. Labels: correct (pts == 20), canary false answer (expected answer == ""), wrong
(answerable, answered, not correct). The gate answers iff (s1 >= t1) or (gain >= t2); a missing signal never passes
for itself; both missing -> answer. Grid: t1 in [-3.0, -0.05] step 0.05, t2 in [-2, 10] step 0.25. The pick refuses ZERO
correct answers on TUNE and maximises refused canary false answers (ties: larger margin = distance to the nearest
correct row's signals, i.e. the loosest feasible pair). Prints the counts and the thresholds as JSON.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

T1_GRID = [round(-3.0 + 0.05 * i, 2) for i in range(int(round((-0.05 + 3.0) / 0.05)) + 1)]
T2_GRID = [round(-2.0 + 0.25 * i, 2) for i in range(int(round(12 / 0.25)) + 1)]


def collect(paths: list[str]) -> list[dict]:
    """Answered rows of the given report.json files (globs ok): {"label", "s1", "gain", "id", "file"}."""
    files: list[str] = []
    for p in paths:
        hits = sorted(glob.glob(p))
        files += hits if hits else [p]
    rows = []
    for f in files:
        rep = json.loads(Path(f).read_text(encoding="utf-8"))
        for r in rep.get("rows", []):
            got = r.get("got") or {}
            if (got.get("answer") or "") == "":
                continue
            exp = (r.get("expected") or {}).get("answer", "")
            label = "correct" if r.get("pts") == 20 else ("false" if exp == "" else "wrong")
            notes = r.get("notes") or {}
            rows.append({"label": label, "s1": notes.get("gate_b_s1"), "gain": notes.get("context_gain"),
                         "id": r.get("id"), "file": str(f)})
    return rows


def passes(row: dict, t1: float, t2: float) -> bool:
    """True = the gate lets the answer through."""
    s1, gain = row.get("s1"), row.get("gain")
    if s1 is None and gain is None:
        return True
    return (s1 is not None and s1 >= t1) or (gain is not None and gain >= t2)


def counts(rows: list[dict], t1: float, t2: float) -> dict:
    out = {"rows": len(rows)}
    for lab in ("correct", "false", "wrong"):
        sel = [r for r in rows if r["label"] == lab]
        out[lab] = len(sel)
        out[f"{lab}_refused"] = sum(1 for r in sel if not passes(r, t1, t2))
    return out


def _margin(rows: list[dict], t1: float, t2: float) -> float:
    """Smallest slack of a correct row that is only barely let through (bigger = safer)."""
    m = float("inf")
    for r in rows:
        if r["label"] != "correct":
            continue
        slack = []
        if r.get("s1") is not None:
            slack.append(r["s1"] - t1)
        if r.get("gain") is not None:
            slack.append((r["gain"] - t2) / 4.0)       # gain spans ~4x the s1 range; bring both to similar scale
        if slack:
            m = min(m, max(slack))
    return m


def search(rows: list[dict]) -> dict | None:
    """Best (t1, t2) refusing no correct row of ``rows``; None when no pair is feasible."""
    best, best_key = None, None
    for t1 in T1_GRID:
        for t2 in T2_GRID:
            c = counts(rows, t1, t2)
            if c["correct_refused"]:
                continue
            key = (c["false_refused"], _margin(rows, t1, t2), c["wrong_refused"])
            if best_key is None or key > best_key:
                best, best_key = {"theta1": t1, "theta2": t2, "margin": key[1], **c}, key
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tune", nargs="+", required=True, help="report.json files/globs of the TUNE suites")
    ap.add_argument("--holdout", nargs="*", default=[], help="report.json files/globs of the HOLDOUT suites")
    args = ap.parse_args(argv)
    tune = collect(args.tune)
    hold = collect(args.holdout) if args.holdout else []
    best = search(tune)
    res = {"tune_rows": len(tune), "holdout_rows": len(hold),
           "tune_missing_signal": sum(1 for r in tune if r["s1"] is None and r["gain"] is None),
           "recommended": None, "tune": None, "holdout": None}
    if best is None:
        res["note"] = "no (theta1, theta2) refuses zero correct answers on TUNE"
    else:
        t1, t2 = best["theta1"], best["theta2"]
        res["recommended"] = {"MC3_GATE_B_T1": t1, "MC3_GATE_B_T2": t2}
        res["tune"] = counts(tune, t1, t2)
        if hold:
            res["holdout"] = counts(hold, t1, t2)
    print(json.dumps(res, indent=1))
    return 0 if best is not None else 1


if __name__ == "__main__":
    sys.exit(main())
