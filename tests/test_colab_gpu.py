"""gpu/colab_gpu.py: model selection env, weights completeness, per-model run layout and report (no GPU, no network)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_cg(monkeypatch, base, **env):
    for k in ("MC3_MODEL_REPO", "MC3_MODEL_REV", "MC3_MODEL_DIR", "MC3_SUITES"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MC3_BASE", str(base))
    monkeypatch.setenv("MC3_MODEL", "restored-by-monkeypatch")   # the module overwrites it at import
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MC3_PHASE", raising=False)
    spec = importlib.util.spec_from_file_location("colab_gpu_under_test", ROOT / "gpu" / "colab_gpu.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # importing does not run main() outside `colab exec`
    return mod


def test_default_model_is_shipped_qwen3vl(monkeypatch, tmp_path):
    cg = load_cg(monkeypatch, tmp_path)
    assert cg.REPO_ID == "Qwen/Qwen3-VL-8B-Instruct"
    assert cg.REVISION == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    assert cg.MODEL_DIR == tmp_path / "models" / "Qwen3-VL-8B-Instruct"
    assert cg.MODEL_SLUG == "Qwen3-VL-8B-Instruct"


def test_model_env_overrides(monkeypatch, tmp_path):
    cg = load_cg(monkeypatch, tmp_path, MC3_MODEL_REPO="google/gemma-4-12b-it", MC3_MODEL_REV="abc123")
    assert (cg.REPO_ID, cg.REVISION) == ("google/gemma-4-12b-it", "abc123")
    assert cg.MODEL_DIR == tmp_path / "models" / "gemma-4-12b-it" and cg.MODEL_SLUG == "gemma-4-12b-it"
    cg = load_cg(monkeypatch, tmp_path, MC3_MODEL_REPO="Qwen/Qwen3.5-9B", MC3_MODEL_DIR="/data/q35")
    assert cg.REVISION == "main" and str(cg.MODEL_DIR) == "/data/q35" and cg.MODEL_SLUG == "q35"
    # empty values (what `--env X=` leaves behind) mean "unset"
    cg = load_cg(monkeypatch, tmp_path, MC3_MODEL_REPO="", MC3_MODEL_REV="", MC3_MODEL_DIR="")
    assert cg.REPO_ID == "Qwen/Qwen3-VL-8B-Instruct" and cg.MODEL_SLUG == "Qwen3-VL-8B-Instruct"


def test_weights_complete_sharded_and_single_file(monkeypatch, tmp_path):
    cg = load_cg(monkeypatch, tmp_path)
    d = cg.MODEL_DIR
    d.mkdir(parents=True)
    assert cg._weights_complete()[0] is False
    (d / "config.json").write_text("{}")
    (d / "tokenizer.json").write_text("{}")
    (d / "preprocessor_config.json").write_text("{}")
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}}))
    (d / "s1.safetensors").write_text("x")
    ok, why = cg._weights_complete()
    assert not ok and "s2.safetensors" in why
    (d / "s2.safetensors").write_text("x")
    assert cg._weights_complete()[0] is True

    cg = load_cg(monkeypatch, tmp_path, MC3_MODEL_REPO="google/gemma-4-12b-it")      # single model.safetensors, no index
    d = cg.MODEL_DIR
    d.mkdir(parents=True)
    for f in ("config.json", "tokenizer.json", "processor_config.json"):
        (d / f).write_text("{}")
    assert cg._weights_complete()[0] is False
    (d / "model.safetensors").write_text("x")
    assert cg._weights_complete()[0] is True


def test_suite_list_subset(monkeypatch, tmp_path):
    cg = load_cg(monkeypatch, tmp_path, MC3_SUITES="sample,s2,nope")
    for n in ("s1", "s2"):
        (cg.SRC / "eval" / "suites" / n).mkdir(parents=True)
        (cg.SRC / "eval" / "suites" / n / "questions.json").write_text("{}")
    assert [n for n, _, _ in cg.suite_list()] == ["sample", "s2"]


def test_report_groups_by_model_and_legacy(monkeypatch, tmp_path, capsys):
    cg = load_cg(monkeypatch, tmp_path)
    run = cg.RUNS / "r1"
    for model, score in (("Qwen3-VL-8B-Instruct", "7/10"), ("gemma-4-12b-it", "9/10")):
        d = run / model / "s1"
        d.mkdir(parents=True)
        (d / "suite_stdout.txt").write_text(f"x\nTOTAL {score}\n")
    (run / "old").mkdir()
    (run / "old" / "suite_stdout.txt").write_text("TOTAL 1/2\n")
    (cg.RUNS / "CURRENT").write_text("r1")
    assert cg.phase_report() is True
    out = capsys.readouterr().out
    assert "=== model Qwen3-VL-8B-Instruct ===" in out and "=== model gemma-4-12b-it ===" in out
    assert "=== model (legacy) ===" in out and "A/B score" in out
    latest = json.loads((cg.RUNS / "latest.json").read_text())
    assert latest["models"]["gemma-4-12b-it"]["suites"]["s1"]["score"] == 9
    assert latest["suites"]["s1"]["score"] == 7                  # single-model view = the selected (default) model
