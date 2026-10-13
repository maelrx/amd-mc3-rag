"""eval/ab_suite.py and eval/calib_gate_b.py on CPU: stub oracle reader, sample corpus, synthetic reports."""

import json
import time
from pathlib import Path

import pytest

import config
from budget import Deadline
from eval import ab_suite, calib_gate_b
from eval.stub_reader import StubReader

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [{"name": "v5", "env": {}}, {"name": "rank", "env": {"MC3_RANK": "cov_first"}}]


def sample_suites():
    return ab_suite.resolve_suites(["sample"], ROOT)


def test_two_configs_reports_cache_hits_and_restore(tmp_path):
    before = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    summ = ab_suite.run_ab(sample_suites(), CONFIGS, "oracle", None, tmp_path / "out", tmp_path / "cache",
                           limit=6, groups={"TUNE": ["sample"]})
    for cfg in ("v5", "rank"):
        rep = json.loads((tmp_path / "out" / cfg / "sample" / "report.json").read_text())
        assert rep["total"] == 120 and len(rep["rows"]) == 6 and "seconds_est" in rep["rows"][0]
        assert (tmp_path / "out" / cfg / "sample" / "query_01_output.json").exists()
    assert json.loads((tmp_path / "out" / "ab_summary.json").read_text())["baseline"] == "v5"
    c = summ["cache"]
    assert c["misses"] > 0 and c["hits"] > 0          # config 2 re-reads what config 1 already asked
    assert summ["configs"]["rank"]["suites"]["sample"]["vs_baseline"]["net_questions"] == 0
    assert summ["groups"]["TUNE"]["rank"]["total"] == 120
    assert "A/B summary" in ab_suite.format_table(summ)
    assert {k: getattr(config, k) for k in dir(config) if k.isupper()} == before


def test_cache_persists_across_processes(tmp_path):
    ab_suite.run_ab(sample_suites(), CONFIGS[:1], "oracle", None, tmp_path / "o1", tmp_path / "cache", limit=4)
    summ = ab_suite.run_ab(sample_suites(), CONFIGS[:1], "oracle", None, tmp_path / "o2", tmp_path / "cache", limit=4)
    assert summ["cache"]["misses"] == summ["cache"]["skipped"] and summ["cache"]["hits"] > 0   # only empty vqa answers are never cached


def test_applied_maps_env_and_restores():
    old = (config.RANK, config.XLSX_JOIN, config.GATE_B, config.GATE_B_MODE, config.GATE_B_THETA1,
           config.MAX_HOPS, config.BM25_K1)
    env = {"MC3_RANK": "cov_first", "MC3_XLSX_JOIN": "1", "MC3_GATE_B": "1", "MC3_GATE_B_T1": "-1.5",
           "MC3_MAX_HOPS": "3", "MC3_BM25_K1": "1.2"}
    with ab_suite.applied(env):
        assert config.RANK == "cov_first" and config.XLSX_JOIN is True
        assert config.GATE_B_MODE == "1" and config.GATE_B is True and config.GATE_B_THETA1 == -1.5
        assert config.MAX_HOPS == 3 and config.BM25_K1 == 1.2
    with ab_suite.applied({"MC3_GATE_B": "log"}):
        assert config.GATE_B_MODE == "log" and config.GATE_B is False
    assert old == (config.RANK, config.XLSX_JOIN, config.GATE_B, config.GATE_B_MODE, config.GATE_B_THETA1,
                   config.MAX_HOPS, config.BM25_K1)
    with pytest.raises(ValueError):
        ab_suite.applied({"MC3_NOPE": "1"})


def test_index_dir_depends_on_index_flags_only():
    base = ab_suite.index_dir_name("s1")
    with ab_suite.applied({"MC3_RANK": "cov_first", "MC3_CHAIN_GUARD": "1"}):
        assert ab_suite.index_dir_name("s1") == base
    with ab_suite.applied({"MC3_XLSX_JOIN": "1"}):
        assert ab_suite.index_dir_name("s1") != base
    with ab_suite.applied({"MC3_PDF_COLS": "1"}):
        assert ab_suite.index_dir_name("s1") != base


class Spy:
    """Minimal inner reader: no logprob, counts calls, one attribute to forward."""
    force_math = False

    def __init__(self):
        self.n = 0
        self.family = "spy"

    def read(self, package_text, question, images, deadline, max_new_tokens=96):
        self.n += 1
        from schema import ReaderOut
        if question == "slow":
            time.sleep(0.3)
        return ReaderOut("A", [1], [], raw='{"answer":"A"}', n_new_tokens=3, mean_logprob=-0.5)

    def vqa(self, question, image, deadline, max_new_tokens=48, longest_edge=None):
        self.n += 1
        return "v"


def test_cached_reader_semantics(tmp_path):
    from PIL import Image
    spy = Spy()
    cr = ab_suite.CachedReader(spy, ab_suite.Cache(tmp_path))
    assert cr.has_vision is True and cr.family == "spy"
    cr.force_math = True
    assert spy.force_math is True
    assert not hasattr(cr, "logprob")                   # missing methods stay missing
    img = Image.new("RGB", (8, 8), (1, 2, 3))
    img2 = Image.new("RGB", (8, 8), (9, 9, 9))
    a = cr.read("pkg", "q", [("f.png", img)], Deadline.after(5))
    b = cr.read("pkg", "q", [("f.png", img)], Deadline.after(1))     # deadline is not part of the key
    assert a == b and spy.n == 1 and cr._cache.hits == 1
    cr.read("pkg", "q", [("f.png", img2)], Deadline.after(5))        # other pixels
    cr.read("pkg", "q", [("f.png", img)], Deadline.after(5), max_new_tokens=10)   # other token budget
    assert spy.n == 3
    assert cr.vqa("q", ("f.png", img), Deadline.after(5)) == "v" and cr.vqa("q", ("f.png", img), None) == "v"
    assert spy.n == 4
    # cut by the deadline -> not cached
    cr.read("pkg", "slow", [], Deadline.after(0.1))
    cr.read("pkg", "slow", [], Deadline.after(0.1))
    assert spy.n == 6 and cr._cache.skipped == 2
    # a fresh cache object over the same dir sees the persisted entries
    cr2 = ab_suite.CachedReader(Spy(), ab_suite.Cache(tmp_path))
    assert cr2.read("pkg", "q", [("f.png", img)], Deadline.after(5)).mean_logprob == -0.5
    assert cr2._cache.hits == 1 and cr2._inner.n == 0


def test_cached_reader_with_stub_logprob(tmp_path):
    cr = ab_suite.CachedReader(StubReader("oracle", {}), ab_suite.Cache(tmp_path))
    assert cr.logprob("p", "q", [], "x") == -1.0 and cr.logprob("p", "q", [], "x") == -1.0
    assert cr._cache.hits == 1 and cr._cache.misses == 1


def _rep(rows, fa, total=None):
    rows = [{"id": i, "pts": p} for i, p in rows]
    return {"rows": rows, "total": total if total is not None else sum(p for _, p in [(r["id"], r["pts"]) for r in rows]),
            "max": 20 * len(rows), "answerable": len(rows), "answerable_correct": sum(1 for r in rows if r["pts"]),
            "canaries": 2, "false_answer_ids": fa, "seconds": {"p50": 1.0, "p95": 2.0, "max": 3.0}}


def test_summary_deltas_on_synthetic_reports():
    base = {"s1": _rep([("query_01", 20), ("query_02", 0), ("query_03", 20)], ["query_09"]),
            "s2": _rep([("query_01", 0), ("query_02", 20)], [])}
    new = {"s1": _rep([("query_01", 0), ("query_02", 20), ("query_03", 20)], ["query_09", "query_10"]),
           "s2": _rep([("query_01", 20), ("query_02", 20)], [])}
    summ = ab_suite.summarize({"base": base, "new": new}, {"G": ["s1", "s2", "zzz"]})
    d = summ["configs"]["new"]["suites"]["s1"]["vs_baseline"]
    assert d == {"gained": ["query_02"], "lost": ["query_01"], "new_false_answers": ["query_10"], "net_questions": 0}
    assert summ["configs"]["new"]["suites"]["s2"]["vs_baseline"]["gained"] == ["query_01"]
    assert "vs_baseline" not in summ["configs"]["base"]["suites"]["s1"]
    g = summ["groups"]["G"]["new"]
    assert g["suites"] == ["s1", "s2"] and g["total"] == 80 and g["max"] == 100
    assert g["vs_baseline"] == {"gained": 2, "lost": 1, "new_false_answers": 1, "net_questions": 1}
    assert "NEW FALSE" in ab_suite.format_table(summ)


def test_parse_groups_and_configs():
    assert ab_suite.parse_groups(["TUNE=sample,s1", "H=h1;P=p0,p1"]) == {
        "TUNE": ["sample", "s1"], "H": ["h1"], "P": ["p0", "p1"]}
    assert [c["name"] for c in ab_suite.load_configs(json.dumps(CONFIGS))] == ["v5", "rank"]
    with pytest.raises(SystemExit):
        ab_suite.load_configs('[{"name":"a"},{"name":"a"}]')


def _row(label, s1, gain, i="q"):
    ans = {"correct": ("X", 20, "X"), "false": ("X", 0, ""), "wrong": ("Y", 0, "X")}[label]
    return {"id": i, "pts": ans[1], "got": {"answer": ans[0], "citations": []}, "expected": {"answer": ans[2]},
            "notes": {k: v for k, v in (("gate_b_s1", s1), ("context_gain", gain)) if v is not None}}


def test_calibration_grid_on_synthetic_rows(tmp_path):
    tune = {"rows": [_row("correct", -0.3, 5.0), _row("correct", -1.2, 4.0), _row("correct", -2.5, 0.5),
                     _row("false", -2.8, -1.0), _row("false", -2.9, 0.0), _row("false", -0.9, 1.0),
                     _row("correct", None, None),
                     {"id": "r", "pts": 20, "got": {"answer": "", "citations": []}, "expected": {"answer": ""}}]}
    hold = {"rows": [_row("correct", -0.5, 3.0), _row("false", -2.95, -0.5), _row("false", -0.2, 6.0)]}
    p1, p2 = tmp_path / "t.json", tmp_path / "h.json"
    p1.write_text(json.dumps(tune))
    p2.write_text(json.dumps(hold))
    rows = calib_gate_b.collect([str(tmp_path / "t*.json")])
    assert len(rows) == 7                                # the refusal row is skipped
    best = calib_gate_b.search(rows)
    assert best["correct_refused"] == 0 and best["false_refused"] == 2     # the -0.9/gain 1.0 canary is not separable
    assert best["theta1"] <= -2.5 and best["theta2"] <= 0.5
    assert calib_gate_b.passes({"s1": None, "gain": None}, -1, 1) and not calib_gate_b.passes({"s1": -2, "gain": None}, -1, 1)
    hc = calib_gate_b.counts(calib_gate_b.collect([str(p2)]), best["theta1"], best["theta2"])
    assert hc["correct_refused"] == 0 and hc["false_refused"] == 1 and hc["false"] == 2
    assert calib_gate_b.main(["--tune", str(p1), "--holdout", str(p2)]) == 0


def test_colab_ab_phase_registered(monkeypatch, tmp_path):
    import importlib.util
    monkeypatch.setenv("MC3_BASE", str(tmp_path))
    monkeypatch.delenv("MC3_PHASE", raising=False)
    spec = importlib.util.spec_from_file_location("colab_gpu_ab", ROOT / "gpu" / "colab_gpu.py")
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)
    assert "ab" in cg.PHASES and "ab" not in cg.ALL_PHASES and "\n  ab " in cg.__doc__
    monkeypatch.setenv("MC3_AB_CONFIGS", '[{"name":"v5"}]')
    assert cg.ab_configs_arg() == '[{"name":"v5"}]'
    monkeypatch.setenv("MC3_AB_CONFIGS", "eval/ab_configs.json")
    assert cg.ab_configs_arg() == str(cg.SRC / "eval/ab_configs.json")
    assert cg.phase_ab() is False                        # src not extracted -> skip, no crash
