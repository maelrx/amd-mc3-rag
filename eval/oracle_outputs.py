#!/usr/bin/env python3
"""Write the gold answers as harness outputs, to prove a generated suite is self-consistent.

    python3 eval/oracle_outputs.py eval/suites/s1/questions.json <out_dir>
    python3 eval/score.py eval/suites/s1/questions.json <out_dir>     # TOTAL = 20 x N

One <out_dir>/query_NN_output.json per query: gold answer and citations with confidence 1.0;
refusals ("unanswerable") get answer "", citations [] and confidence 0.0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    gold = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    out = Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for q in gold["queries"]:
        refuse = q.get("unanswerable") or q.get("expected_answer", "") == ""
        res = {"answer": "" if refuse else q["expected_answer"],
               "citations": [] if refuse else list(q["expected_citations"]),
               "confidence": 0.0 if refuse else 1.0}
        (out / f"query_{q['n']:02d}_output.json").write_text(json.dumps(res, ensure_ascii=False) + "\n",
                                                             encoding="utf-8")
    print(f"wrote {len(gold['queries'])} outputs to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
