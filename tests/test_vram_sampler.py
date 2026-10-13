"""gpu/vram_sampler.py with every GPU/CLI call mocked (no GPU needed)."""

import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu"))
import vram_sampler as vs  # noqa: E402


AMD_SMI_NEW = json.dumps([{"gpu": 0, "mem_usage": {
    "total_vram": {"value": 49152, "unit": "MB"}, "used_vram": {"value": 20480, "unit": "MB"},
    "free_vram": {"value": 28672, "unit": "MB"}, "used_visible_vram": {"value": 99999, "unit": "MB"}}}])
ROCM_SMI = json.dumps({"card0": {"VRAM Total Memory (B)": "51539607552",
                                 "VRAM Total Used Memory (B)": "10737418240"}})


def test_parse_nvidia():
    assert vs.parse_nvidia_smi("1234\n") == 1234.0
    assert vs.parse_nvidia_smi("100\n4000\n") == 4000.0
    assert vs.parse_nvidia_smi("") is None
    assert vs.parse_nvidia_smi("No devices were found") is None


def test_parse_amd_smi_new_and_rocm_smi():
    assert vs.parse_amd_json(AMD_SMI_NEW) == 20480.0          # not total, not visible, not free
    assert vs.parse_amd_json(ROCM_SMI) == 10240.0             # bytes -> MiB
    assert vs.parse_amd_json("not json") is None
    assert vs.parse_amd_json("[]") is None


def test_amd_fallback_to_rocm_smi(monkeypatch):
    calls = []

    def fake_run(argv):
        calls.append(argv[0])
        return ROCM_SMI if argv[0] == "rocm-smi" else None  # amd-smi fails

    monkeypatch.setattr(vs.shutil, "which", lambda n: "/bin/" + n)
    monkeypatch.setattr(vs, "_run", fake_run)
    assert vs.read_amd() == 10240.0
    assert calls == ["amd-smi", "rocm-smi"]


def test_sampler_csv_peak_and_stop(tmp_path, monkeypatch):
    seq = iter([1000.0, 5000.0, 3000.0] + [2000.0] * 50)
    monkeypatch.setattr(vs, "read_nvidia", lambda: next(seq))
    monkeypatch.setattr(vs, "read_amd", lambda: None)
    fake_cuda = SimpleNamespace(is_available=lambda: True,
                                memory_allocated=lambda: 300 * (1 << 20),
                                max_memory_allocated=lambda: 800 * (1 << 20))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    out = tmp_path / "sub" / "vram.csv"
    s = vs.start(str(out), interval=0.05)
    time.sleep(0.4)
    info = s.stop()
    assert info["peak_mib"] == 5000.0
    assert info["samples"] >= 3
    assert "nvidia" in info["source"] and "torch" in info["source"]
    assert info["peak_by_source_mib"]["torch"] == 800.0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == info["samples"]
    assert rows[0]["nvidia_used_mib"] == "1000.0" and rows[0]["torch_alloc_mib"] == "300.0"


def test_amd_source_and_dead_source_is_dropped(monkeypatch):
    n = {"nv": 0}

    def nv():
        n["nv"] += 1
        return None

    monkeypatch.setattr(vs, "read_nvidia", nv)
    monkeypatch.setattr(vs, "read_amd", lambda: 7000.0)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    s = vs.Sampler(None, 0.01)
    for _ in range(6):
        s.sample_now()
    assert n["nv"] == vs.MAX_FAILS                              # stops calling a source that never answers
    info = s.stop()
    assert info["peak_mib"] == 7000.0 and info["source"] == "amd"


def test_never_crashes(monkeypatch, tmp_path):
    def boom():
        raise RuntimeError("driver fell over")

    monkeypatch.setattr(vs, "read_nvidia", boom)
    monkeypatch.setattr(vs, "read_amd", boom)
    s = vs.start(str(tmp_path / "x.csv"), interval=0.02)
    time.sleep(0.1)
    info = s.stop()                                              # no exception
    assert info["peak_mib"] == 0.0 and "error" in info
    # unwritable path: still returns a usable sampler
    bad = vs.start("/proc/nope/x.csv", interval=0.02)
    assert isinstance(bad.stop(), dict)
