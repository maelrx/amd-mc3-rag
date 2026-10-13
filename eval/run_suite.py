#!/usr/bin/env python3
"""Index a corpus and answer a questions file in-process, then score it like the harness would.

    python3 eval/run_suite.py --corpus eval/suites/s1/corpus --questions eval/suites/s1/questions.json \\
        --out /tmp/run_s1 [--reader oracle|lexical|adversarial|refuse|real] [--model DIR] [--soft 22] [--limit N]

Writes OUT/index/ (pipeline.index_corpus), OUT/query_NN_output.json (same shape as app.py) and OUT/report.json.
Prints one line per question, TOTAL, a histogram of reasons, p50/p95 seconds and, for canaries
(expected_answer == ""), the number of FALSE ANSWERS (non-empty answers). ``--reader real`` needs a GPU stack.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "app", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402
import pipeline  # noqa: E402
from budget import Deadline, Trace  # noqa: E402
from eval import score as score_mod  # noqa: E402
from eval.stub_reader import StubReader  # noqa: E402

IMAGE_WORDS = ("jpg", "png", "image", "multimodal", "pinout", "label")


def make_reader(kind: str, data: dict, model: str | None):
    if kind == "real":
        import reader as reader_mod
        rd = reader_mod.Reader(model or config.MODEL_DIR)
        rd.load()
        rd.self_test_sdpa()
        rd.warmup()
        return rd
    return StubReader.from_questions(data, kind)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def is_image_question(q: dict) -> bool:
    cat = q.get("category", "").lower()
    return any(w in cat for w in IMAGE_WORDS)


def run(corpus: Path, questions: Path, out: Path, reader_kind: str = "oracle", model: str | None = None,
        soft: float = 22.0, limit: int | None = None, reuse_index: bool = False, verbose: bool = True) -> dict:
    corpus, out = Path(corpus), Path(out)
    data = json.loads(Path(questions).read_text(encoding="utf-8"))
    if limit:
        data = dict(data, queries=data["queries"][:limit])
    out.mkdir(parents=True, exist_ok=True)
    index_dir = out / "index"
    if index_dir.exists() and not reuse_index:
        shutil.rmtree(index_dir)
    reader = make_reader(reader_kind, data, model)

    t0 = time.time()
    idx = pipeline.index_corpus(corpus, index_dir, reader, Deadline.after(config.INDEX_BUDGET_S))
    t_index = time.time() - t0
    if verbose:
        print(f"indexed {len(idx.files)} files, {len(idx.records)} records in {t_index:.1f}s")
    return run_questions(corpus, questions, data, out, reader, idx, t_index, reader_kind, soft, verbose)


def run_questions(corpus: Path, questions: Path, data: dict, out: Path, reader, idx, t_index: float,
                  reader_kind: str = "oracle", soft: float = 22.0, verbose: bool = True,
                  extra_seconds=None, on_question=None) -> dict:
    """Answer ``data["queries"]`` with an already built reader and index, write OUT/query_NN_output.json and
    OUT/report.json. ``extra_seconds()`` (optional) returns a cumulative counter of seconds a caching reader
    saved; the per-question delta is reported as ``seconds_est`` (= seconds + saved) next to the wall-clock
    ``seconds`` (and ``p50_est``/``p95_est``/``max_est``). ``on_question(qid, per_q_row)`` is a progress hook."""
    corpus, out = Path(corpus), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per_q = {}
    for q in data["queries"]:
        qid = f"query_{q['n']:02d}"
        trace = Trace()
        saved0 = extra_seconds() if extra_seconds else 0.0
        t = time.time()
        res = pipeline.answer(idx, reader, q["query"], Deadline.after(soft), corpus, trace)
        secs = time.time() - t
        (out / f"{qid}_output.json").write_text(json.dumps(res.to_output(), ensure_ascii=False) + "\n",
                                                encoding="utf-8")
        per_q[qid] = {"seconds": round(secs, 3), "reason": res.trace.get("reason", ""),
                      "attempts": res.trace.get("attempts", 0), "n": q["n"],
                      "support_rid": res.trace.get("support_rid"),
                      "notes": res.trace.get("notes", {}),
                      "stages": res.trace.get("stages", [])}
        if extra_seconds:
            per_q[qid]["seconds_est"] = round(secs + extra_seconds() - saved0, 3)
        if on_question:
            on_question(qid, per_q[qid])

    sc = score_mod.score(data, out)
    by_n = {q["n"]: q for q in data["queries"]}
    secs = [v["seconds"] for v in per_q.values()]
    reasons = Counter(v["reason"] for v in per_q.values())
    canary_reasons = Counter(per_q[r["id"]]["reason"] for r in sc["rows"] if by_n[per_q[r["id"]]["n"]]["expected_answer"] == "")
    false_answers = [r["id"] for r in sc["rows"]
                     if by_n[per_q[r["id"]]["n"]]["expected_answer"] == "" and r["got"] and r["got"]["answer"] != ""]
    answerable = [r for r in sc["rows"] if by_n[per_q[r["id"]]["n"]]["expected_answer"] != ""]
    text_ans = [r for r in answerable if not is_image_question(by_n[per_q[r["id"]]["n"]])]
    n_canary = len(sc["rows"]) - len(answerable)
    report = {
        "reader": reader_kind, "corpus": str(corpus), "questions": str(questions),
        "total": sc["total"], "max": sc["max"],
        "answerable": len(answerable), "answerable_correct": sum(1 for r in answerable if r["pts"]),
        "text_answerable": len(text_ans), "text_correct": sum(1 for r in text_ans if r["pts"]),
        "canaries": n_canary, "false_answers": len(false_answers), "false_answer_ids": false_answers,
        "reasons": dict(reasons), "canary_reasons": dict(canary_reasons),
        "seconds": {"index": round(t_index, 2), "p50": round(pct(secs, 50), 3), "p95": round(pct(secs, 95), 3),
                    "max": round(max(secs), 3) if secs else 0.0},
        "rows": [dict(r, **per_q[r["id"]], category=r["category"]) for r in sc["rows"]],
    }
    if extra_seconds:
        est = [v["seconds_est"] for v in per_q.values()]
        report["seconds"].update({"p50_est": round(pct(est, 50), 3), "p95_est": round(pct(est, 95), 3),
                                  "max_est": round(max(est), 3) if est else 0.0})
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print_report(report)
    return report


def print_report(rep: dict) -> None:
    for r in rep["rows"]:
        flag = "OK " if r["pts"] else ("ANS" if not r["answer_ok"] else "CIT")
        print(f"{flag} {r['id']} {r['seconds']:5.2f}s {r['reason']:<18} got={r['got']} exp={r['expected']} "
              f"[{r['category']}]")
    print(f"TOTAL {rep['total']}/{rep['max']}")
    print(f"answerable: {rep['answerable_correct']}/{rep['answerable']}  "
          f"(text-only: {rep['text_correct']}/{rep['text_answerable']})")
    print("reasons: " + ", ".join(f"{k or '-'}={v}" for k, v in sorted(rep["reasons"].items(), key=lambda kv: -kv[1])))
    if rep["canaries"]:
        print("canary reasons: " + ", ".join(f"{k or '-'}={v}" for k, v in sorted(rep["canary_reasons"].items(),
                                                                                 key=lambda kv: -kv[1])))
    print(f"canaries: {rep['canaries']}  FALSE ANSWERS: {rep['false_answers']} {rep['false_answer_ids']}")
    s = rep["seconds"]
    print(f"seconds/question: p50={s['p50']} p95={s['p95']} max={s['max']}  index={s['index']}s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reader", default="oracle", choices=["oracle", "lexical", "adversarial", "refuse", "real"])
    ap.add_argument("--model", default=None, help="model dir for --reader real (default MC3_MODEL)")
    ap.add_argument("--soft", type=float, default=config.ANSWER_SOFT_S, help="per-question soft deadline, seconds")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reuse-index", action="store_true", help="keep OUT/index from a previous run")
    args = ap.parse_args(argv)
    run(args.corpus, args.questions, args.out, args.reader, args.model, args.soft, args.limit, args.reuse_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
