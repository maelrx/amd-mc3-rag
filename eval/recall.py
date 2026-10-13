#!/usr/bin/env python3
"""Package recall: is every gold file (and a record holding the gold value) inside the package?

    python3 eval/recall.py eval/suites/s1 eval/suites/s2 [--ocr]
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import index as index_mod  # noqa: E402
import parse  # noqa: E402
import retrieve  # noqa: E402
import snap  # noqa: E402


def run(suite: Path, ocr: bool) -> tuple[int, int, list[str]]:
    corpus = suite / "corpus"
    if ocr:
        import pipeline
        idx = pipeline.index_corpus(corpus, Path(tempfile.mkdtemp()))
    else:
        idx = index_mod.build(parse.walk(corpus))
    gold = json.loads((suite / "questions.json").read_text())["queries"]
    hit = tot = 0
    misses = []
    for g in gold:
        if not g["expected_answer"]:
            continue
        tot += 1
        q = retrieve.analyze(g["query"], idx)
        pkg = retrieve.assemble(idx, q, 1)
        files_ok = all(f in pkg.files for f in g["expected_citations"])
        value_file = g["expected_citations"][-1]
        rec_ok = any(idx.record(r).file == value_file and snap.find(g["expected_answer"], idx.record(r).text)
                     for r in pkg.rids) or idx.file_by_path[value_file].kind == "image"
        if files_ok and rec_ok:
            hit += 1
        else:
            hits = retrieve.hop(idx, q, retrieve.retrieve(idx, q, 1))
            rank = next((i for i, h in enumerate(hits) if h.file == value_file), None)
            misses.append(f"  q{g['n']:02d} files_ok={files_ok} rec_ok={rec_ok} gold_rank={rank} "
                          f"pkg={[f.split('/')[-1] for f in pkg.files]} :: {g['query'][:70]}")
    return hit, tot, misses


if __name__ == "__main__":
    ocr = "--ocr" in sys.argv
    for s in [a for a in sys.argv[1:] if not a.startswith("--")]:
        h, t, m = run(Path(s), ocr)
        print(f"{s}: recall {h}/{t}")
        print("\n".join(m))
