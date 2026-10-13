#!/usr/bin/env python3
"""Calibrate the warrant threshold (app/warrant.py, MC3_WARRANT_T) from report.json files of a log-mode run (MC3_WARRANT=log).

    python3 eval/calib_warrant.py --tune 'OUT/warrant_log/s*/report.json' ... --holdout 'OUT/warrant_log/h*/report.json'

Every answered row (got.answer != "") carries the trace note ``warrant`` = log P(YES) - log P(NO). Labels: correct
(pts == 20), canary false answer (expected answer == ""), wrong (answerable, answered, not correct). The check answers
iff warrant >= theta; a missing score always answers. Grid: theta in [-10, 10] step 0.25. The pick refuses ZERO correct
answers on TUNE and maximises refused canary false answers plus wrong answers (ties: more false answers refused, then
the larger margin = distance of theta to the lowest correct score). Prints the counts and the threshold as JSON.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

THETA_GRID = [round(-10.0 + 0.25 * i, 2) for i in range(int(round(20 / 0.25)) + 1)]


def collect(paths: list[str]) -> list[dict]:
    """Answered rows of the given report.json files (globs ok): {"label", "warrant", "id", "file"}."""
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
            rows.append({"label": label, "warrant": notes.get("warrant"), "id": r.get("id"), "file": str(f)})
    return rows


def passes(row: dict, theta: float) -> bool:
    """True = the check lets the answer through."""
    w = row.get("warrant")
    return w is None or w >= theta


def counts(rows: list[dict], theta: float) -> dict:
    out = {"rows": len(rows)}
    for lab in ("correct", "false", "wrong"):
        sel = [r for r in rows if r["label"] == lab]
        out[lab] = len(sel)
        out[f"{lab}_refused"] = sum(1 for r in sel if not passes(r, theta))
    return out


def _margin(rows: list[dict], theta: float) -> float:
    """Lowest correct score minus theta (bigger = safer); inf when no correct row has a score."""
    ws = [r["warrant"] for r in rows if r["label"] == "correct" and r.get("warrant") is not None]
    return min(ws) - theta if ws else float("inf")


def search(rows: list[dict]) -> dict | None:
    """Best theta refusing no correct row of ``rows``; None when none is feasible."""
    best, best_key = None, None
    for theta in THETA_GRID:
        c = counts(rows, theta)
        if c["correct_refused"]:
            continue
        key = (c["false_refused"] + c["wrong_refused"], c["false_refused"], _margin(rows, theta))
        if best_key is None or key > best_key:
            best, best_key = {"theta": theta, "margin": key[2], **c}, key
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
           "tune_missing_signal": sum(1 for r in tune if r["warrant"] is None),
           "recommended": None, "tune": None, "holdout": None}
    if best is None:
        res["note"] = "no theta refuses zero correct answers on TUNE"
    else:
        theta = best["theta"]
        res["recommended"] = {"MC3_WARRANT_T": theta}
        res["tune"] = counts(tune, theta)
        if hold:
            res["holdout"] = counts(hold, theta)
    print(json.dumps(res, indent=1))
    return 0 if best is not None else 1


if __name__ == "__main__":
    sys.exit(main())
