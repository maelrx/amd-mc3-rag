#!/usr/bin/env python3
"""Score outputs the way the MC3 brief describes.

Answer: uppercased, whitespace removed, and the characters - . · _ removed, then
compared with the expected answer or any alias. Citations: exact set. A question
scores 20 only when both hold; there is no partial credit.

    python3 eval/score.py sample-questions.json out_dir/
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def norm(s: str) -> str:
    return re.sub(r"[\s\-.·_]", "", str(s)).upper()


def unit_free(s: str) -> str:
    """Brief: 94, 94 C and 94°C all match. Strip a trailing degree/unit token for the
    alias check only; the primary comparison stays strict."""
    return norm(re.sub(r"\s*(°\s*[CF]?|degrees?|deg)\s*$", "", str(s), flags=re.I))


def score(questions: dict, out_dir: Path) -> dict:
    rows, total = [], 0
    for q in questions["queries"]:
        qid = f"query_{q['n']:02d}"
        f = out_dir / f"{qid}_output.json"
        try:
            got = json.loads(f.read_text())
            ans, cites = got["answer"], got["citations"]
            malformed = not isinstance(ans, str) or not isinstance(cites, list)
        except Exception:
            ans, cites, malformed = None, None, True
        exp = q.get("expected_answer", "")
        accepted = {norm(exp)} | {norm(a) for a in q.get("answer_aliases", [])} | {unit_free(exp)}
        if malformed:
            ok_a = ok_c = False
        elif exp == "":
            ok_a = ans == ""
            ok_c = cites == []
        else:
            ok_a = norm(str(ans)) in accepted or unit_free(str(ans)) in accepted
            ok_c = set(cites or []) == set(q.get("expected_citations", []))
        pts = 20 if ok_a and ok_c else 0
        total += pts
        rows.append({"id": qid, "pts": pts, "answer_ok": ok_a, "citations_ok": ok_c,
                     "got": None if malformed else {"answer": ans, "citations": cites},
                     "expected": {"answer": exp, "citations": q.get("expected_citations", [])},
                     "category": q.get("category", "")})
    return {"total": total, "max": 20 * len(questions["queries"]), "rows": rows}


if __name__ == "__main__":
    res = score(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
    for r in res["rows"]:
        flag = "OK " if r["pts"] else ("ANS" if not r["answer_ok"] else "CIT")
        print(f"{flag} {r['id']} got={r['got']} exp={r['expected']} [{r['category']}]")
    print(f"TOTAL {res['total']}/{res['max']}")
