"""eval/gen_corpus.py v2: opt-in traps T25-T28 (xlsx join, chain with decoy, several logs, PDF family table),
--n-images, --max-refusals. The gold answer must be in the text app/parse.py extracts from the cited file."""

import json
import re
from pathlib import Path

import pytest

import parse
from conftest import ROOT
from eval import gen_corpus as gc

NEW = ["T25", "T26", "T27", "T28"]
SEEDS = (201, 202)

DEFAULT_TRAPS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
                 "T13", "T14b", "T15b", "T17b", "T18b", "T19b", "T20", "T21", "T22", "T23", "T24"]


def _norm(s: str) -> str:
    return re.sub(r"[\s\-.·_]", "", str(s)).upper()


def _rendered(seed: int, tmp: Path, traps=NEW, n_files=24, n_images=0, max_refusals=None):
    ctx = gc.build(seed, 0, traps, n_files, n_images, max_refusals)
    assert gc.validate(ctx) == []
    gold = gc.render(ctx, tmp)
    return ctx, gold, tmp / "corpus"


@pytest.fixture(scope="module", params=SEEDS)
def suite(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(f"p1_{request.param}")
    ctx, gold, corpus = _rendered(request.param, tmp)
    extracted = {}

    def text_of(rel: str) -> str:
        if rel not in extracted:
            rec = parse.extract(corpus, corpus / rel)
            assert rec["status"] == "ok", (rel, rec)
            extracted[rel] = rec["text"]
        return extracted[rel]
    return {"ctx": ctx, "gold": gold, "corpus": corpus, "text": text_of, "seed": request.param}


def _trap_of(suite, q):
    traps = json.loads((suite["corpus"].parent / "traps.json").read_text())
    return traps[str(q["n"])]


# ---------------------------------------------------------------- defaults unchanged

def test_default_trap_list_is_unchanged():
    assert gc.TRAPS == DEFAULT_TRAPS
    assert not set(NEW) & set(gc.TRAPS)
    assert gc.EXTRA_TRAPS == NEW


def test_cli_default_runs_no_new_trap(monkeypatch, tmp_path):
    seen = {}
    real_build = gc.build

    def spy(seed, attempt, traps, *a, **kw):
        seen["traps"] = list(traps)
        return real_build(seed, attempt, ["T1"], 12)       # keep the CLI test fast
    monkeypatch.setattr(gc, "build", spy)
    assert gc.main(["--seed", "3", "--out", str(tmp_path)]) == 0
    assert seen["traps"] == DEFAULT_TRAPS


def test_t28_conflicts_with_datasheet_traps():
    with pytest.raises(SystemExit):
        gc.main(["--seed", "1", "--out", "/nonexistent", "--traps", "T1,T28"])


def test_unknown_trap_still_rejected():
    with pytest.raises(SystemExit):
        gc.main(["--seed", "1", "--out", "/nonexistent", "--traps", "T99"])


# ---------------------------------------------------------------- gold vs files

def test_every_new_trap_has_questions_and_a_canary(suite):
    tags = {}
    for q in suite["gold"]["queries"]:
        tags.setdefault(_trap_of(suite, q), []).append(q)
    for t in NEW:
        assert any(not q.get("unanswerable") for q in tags[t]), t
        assert any(q.get("unanswerable") for q in tags[f"{t}c"]), t


def test_gold_answer_is_in_the_cited_extracted_text(suite):
    n = 0
    for q in suite["gold"]["queries"]:
        if _trap_of(suite, q) not in NEW:
            continue
        n += 1
        assert q["expected_citations"], q["query"]
        blob = _norm("\n".join(suite["text"](c) for c in q["expected_citations"]))
        assert any(_norm(a) in blob for a in [q["expected_answer"], *q["answer_aliases"]]), q["query"]
        for c in q["expected_citations"]:
            assert (suite["corpus"] / c).is_file()
    assert n >= 15


def test_canaries_have_empty_gold_and_their_facts_are_absent(suite):
    for q in suite["gold"]["queries"]:
        t = _trap_of(suite, q)
        if t in {f"{x}c" for x in NEW}:
            assert q.get("unanswerable") is True
            assert q["expected_answer"] == "" and q["expected_citations"] == []


def test_generation_is_deterministic(tmp_path):
    a = gc.build(201, 0, NEW, 24)
    b = gc.build(201, 0, NEW, 24)
    assert [q["query"] for q in a.queries] == [q["query"] for q in b.queries]
    assert {p: d.text() for p, d in a.docs.items()} == {p: d.text() for p, d in b.docs.items()}


# ---------------------------------------------------------------- T25

def test_t25_join_workbook(suite):
    path = suite["ctx"].cache["T25"]["path"]
    text = suite["text"](path)
    for sheet in ("Parts", "LeadTimes", "Costs"):
        assert f"[sheet {sheet}]" in text
    doc = suite["ctx"].docs[path]
    lead = next(s for s in doc.sheets if s["name"] == "LeadTimes")
    costs = next(s for s in doc.sheets if s["name"] == "Costs")
    parts = next(s for s in doc.sheets if s["name"] == "Parts")
    assert "product" not in lead["header"] and "product" not in costs["header"]    # keyed ONLY by part_number
    assert lead["header"][0] == costs["header"][0] == "part_number" and "lead_time_weeks" in lead["header"]
    assert len({r[0] for r in parts["rows"]}) == 3                                  # three sibling products
    assert {r[2] for r in parts["rows"]} == {r[0] for r in lead["rows"]} == {r[0] for r in costs["rows"]}


def test_t25_questions_name_product_and_component(suite):
    qs = [q for q in suite["gold"]["queries"] if _trap_of(suite, q) == "T25"]
    assert len(qs) >= 4
    doc = suite["ctx"].docs[suite["ctx"].cache["T25"]["path"]]
    parts = next(s for s in doc.sheets if s["name"] == "Parts")
    lead = {r[0]: r[1] for r in next(s for s in doc.sheets if s["name"] == "LeadTimes")["rows"]}
    by_pc = {(r[0], r[1].lower()): lead[r[2]] for r in parts["rows"]}
    for q in qs:
        m = re.match(r"What is the lead time, in weeks, of the (.+) for the (\S+)\?", q["query"])
        if not m:
            continue
        comp, prod = m.groups()
        assert str(by_pc[(prod, comp)]) == q["expected_answer"]
        # siblings carry a different value for the same component (near-miss)
        sibs = {v for (p, c), v in by_pc.items() if c == comp and p != prod}
        assert sibs and str(q["expected_answer"]) not in {str(v) for v in sibs}
        assert q["expected_citations"] == [suite["ctx"].cache["T25"]["path"]]


# ---------------------------------------------------------------- T26

def test_t26_chain_has_a_topical_decoy(suite):
    ctx = suite["ctx"]
    csv_path = ctx.cache["T26"]["csv"]
    rows = [r.split(",") for r in (suite["corpus"] / csv_path).read_text().splitlines()[1:]]
    chains = [q for q in suite["gold"]["queries"] if _trap_of(suite, q) == "T26"
              and q["query"].startswith("The production log shows")]
    assert len(chains) == 3
    for q in chains:
        noun = re.match(r"The production log shows a (.+) incident\.", q["query"]).group(1)
        log, csv = q["expected_citations"]
        assert csv == csv_path and log.startswith("logs/")
        real = [r for r in rows if r[-1] == q["expected_answer"]]
        assert len(real) == 1
        assert real[0][0] in suite["text"](log)                       # log names the real ticket
        assert noun.lower() not in real[0][3].lower()                  # real summary does not repeat the topic words
        decoys = [r for r in rows if noun.lower() in r[3].lower()]
        assert decoys and all(d[-1] != q["expected_answer"] and d[0] != real[0][0] for d in decoys)
        assert all(d[0] not in suite["text"](log) for d in decoys)


def test_t26_canary_event_is_in_no_log(suite):
    c = suite["ctx"].cache["T26"]
    can = next(q for q in suite["gold"]["queries"] if c["noun_absent"] in q["query"])
    assert can["expected_answer"] == ""
    assert all(c["noun_absent"] not in suite["text"](p) for p in suite["ctx"].docs if p.startswith("logs/prod_serving"))
    assert c["ticket_absent"] in suite["text"](c["csv"])                # the decoy-only ticket exists


# ---------------------------------------------------------------- T27

def test_t27_several_logs_with_distinct_codes(suite):
    qs = [q for q in suite["gold"]["queries"] if _trap_of(suite, q) == "T27"]
    assert len(qs) == 4
    logs = {q["expected_citations"][0] for q in qs}
    codes = {q["expected_answer"] for q in qs}
    assert len(logs) == 4 and len(codes) == 4
    dated = [q for q in qs if re.search(r"on \d{4}-\d\d-\d\d\?$", q["query"])]
    assert len(dated) == 3
    for q in qs:
        log = q["expected_citations"][0]
        assert re.search(rf"ERROR\s+{q['expected_answer']}:", suite["text"](log))
        m = re.search(r"on (\d{4}-\d\d-\d\d)\?$", q["query"])
        if m:
            assert m.group(1) in log


def test_t27_same_event_on_two_dates_has_two_codes(suite):
    qs = [q for q in suite["gold"]["queries"] if _trap_of(suite, q) == "T27"]
    stems = {}
    for q in qs:
        stems.setdefault(re.sub(r" on \d{4}-\d\d-\d\d\?$", "", q["query"]), []).append(q["expected_answer"])
    assert any(len(v) == 2 and v[0] != v[1] for v in stems.values())


# ---------------------------------------------------------------- T28

def test_t28_family_table_pdf(suite):
    c = suite["ctx"].cache["T28"]
    text = suite["text"](c["path"])
    header = next(ln for ln in text.splitlines() if ln.startswith("Parameter"))
    cols = header.split()[1:]
    assert cols == [c["lo"], c["mid"], c["hi"]]
    jt = next(ln for ln in text.splitlines() if ln.startswith("Maximum junction temperature"))
    assert len(re.findall(r"\d+ C", jt)) == 3                       # one value per product, side by side
    assert any(ln.startswith("Maximum coolant inlet") and re.search(r"inlet\s+-\s{2,}", ln) for ln in text.splitlines())
    r1 = c["path"].replace("_r2.pdf", "_r1_WITHDRAWN.pdf")
    assert (suite["corpus"] / r1).is_file()
    t1 = suite["text"](r1)
    assert "WITHDRAWN" in t1
    jt1 = next(ln for ln in t1.splitlines() if ln.startswith("Maximum junction temperature"))
    assert re.findall(r"\d+ C", jt1) != re.findall(r"\d+ C", jt)


def test_t28_questions_pick_the_right_column(suite):
    c = suite["ctx"].cache["T28"]
    text = suite["text"](c["path"])
    prods = [c["lo"], c["mid"], c["hi"]]
    n = 0
    for q in suite["gold"]["queries"]:
        if _trap_of(suite, q) != "T28":
            continue
        n += 1
        prod = next(p for p in prods if p in q["query"])
        label = {"junction": "Maximum junction", "board power": "Board power", "capacity": "Memory capacity",
                 "bandwidth": "Peak memory", "connector": "Auxiliary"}
        key = next(v for k, v in label.items() if k in q["query"])
        row = next(ln for ln in text.splitlines() if ln.startswith(key))
        cells = re.split(r"\s{2,}", row.strip())[1:]
        assert cells[prods.index(prod)].split(" ")[0] == q["expected_answer"].split(" ")[0]
        assert q["expected_citations"] == [c["path"]]
    assert n == 5


# ---------------------------------------------------------------- --n-images / --max-refusals

def test_n_images_adds_answer_images_and_lookalikes(tmp_path):
    ctx, gold, corpus = _rendered(5, tmp_path, ["T1"], 30, n_images=14)
    imgs = [p for p, d in ctx.docs.items() if d.kind in ("png", "jpg")]
    assert len(imgs) == 14
    iq = [q for q in ctx.queries if q["trap"] == "IMG"]
    ans = [q for q in iq if not q["unanswerable"]]
    assert len(ans) == 10 and len(iq) - len(ans) == 3
    texts = {p: d.text() for p, d in ctx.docs.items()}
    for q in ans:
        assert len(q["expected_citations"]) == 1 and q["expected_citations"][0] in imgs and q["imgonly"]
        if gc.is_ident(q["expected_answer"]):
            assert not any(gc.bound_re(q["expected_answer"]).search(t) for t in texts.values())
    assert all((corpus / p).stat().st_size > 1000 for p in imgs)
    assert len({q["expected_citations"][0] for q in ans}) == 9


def test_default_has_no_images_and_n_images_default_is_off(tmp_path):
    ctx = gc.build(7, 0, ["T1"], 20)
    assert not [d for d in ctx.docs.values() if d.kind in ("png", "jpg")]
    assert not [q for q in ctx.queries if q["trap"] == "IMG"]


def test_max_refusals_caps_canaries_only():
    full = gc.build(9, 0, ["T1", "T6"], 20)
    cap = gc.build(9, 0, ["T1", "T6"], 20, max_refusals=3)
    assert sum(q["unanswerable"] for q in cap.queries) == 3
    assert [q for q in full.queries if not q["unanswerable"]] == [q for q in cap.queries if not q["unanswerable"]]
