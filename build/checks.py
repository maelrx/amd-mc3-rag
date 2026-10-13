#!/usr/bin/env python3
"""Checks executed INSIDE the image (CPython 3.14): `deps`, `ocr`, `weights <model_dir>`, `all`.

stdlib at import time; the dependencies are imported lazily by each check.
Exit code 0 = all good, 1 = any failure.

  MC3_SKIP_TORCH_CHECK=1   skip the '+rocm' assertion (local CPU venv / Colab CUDA)
"""
import importlib
import json
import os
import sys

DEFAULT_MODEL = os.environ.get("MC3_MODEL", "/models/Qwen3-VL-8B-Instruct")

# (import name, label) -- the label is only for the printout
MODULES = [
    "transformers", "accelerate", "safetensors", "tokenizers", "huggingface_hub",
    "openpyxl", "pypdf", "cryptography", "pypdfium2", "onnxruntime", "rapidocr", "cv2", "PIL", "numpy",
]

# files that must exist next to the shards (Qwen3-VL repo layout)
MODEL_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "model.safetensors.index.json",
]


def check_deps():
    ok = True
    try:
        import torch
        v = torch.__version__
        if os.environ.get("MC3_SKIP_TORCH_CHECK") == "1":
            print(f"torch {v} (rocm check skipped)")
        elif "+rocm" in v:
            print(f"torch {v} OK (+rocm)")
        else:
            print(f"FAIL torch {v}: '+rocm' not in version (pip replaced the base torch?)")
            ok = False
    except Exception as e:
        print(f"FAIL torch: {type(e).__name__}: {e}")
        ok = False
    for name in MODULES:
        try:
            m = importlib.import_module(name)
            ver = getattr(m, "__version__", None) or getattr(m, "VERSION", None) or "?"
            print(f"{name} {ver}")
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            ok = False
    return ok


def _offline():
    """No network for the rest of the process: env + socket.socket raises."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import socket

    class _Blocked(socket.socket):
        def __init__(self, *a, **k):
            raise OSError("network disabled by checks.py ocr")

    socket.socket = _Blocked


def _norm(s):
    return "".join(ch for ch in s.upper() if ch not in " -")


def check_ocr():
    _offline()
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        text = "REV-C2 B14"
        font = ImageFont.load_default()
        # default font is tiny: render at 1x, then upscale 3x (nearest keeps edges crisp)
        small = Image.new("L", (110, 22), 255)
        ImageDraw.Draw(small).text((4, 4), text, fill=0, font=font)
        img = small.resize((small.width * 3, small.height * 3), Image.NEAREST).convert("RGB")
        from rapidocr import RapidOCR
        engine = RapidOCR()
        res = engine(np.array(img))
        txts = list(getattr(res, "txts", None) or [])
        print("detected:", txts)
        got = _norm("".join(txts))
        miss = [t for t in ("REVC2", "B14") if t not in got]
        if miss:
            print(f"FAIL ocr: tokens not found: {miss} (normalised text: {got!r})")
            return False
        print("ocr OK: REV-C2 and B14 read offline")
        return True
    except Exception as e:
        print(f"FAIL ocr: {type(e).__name__}: {e}")
        return False


def check_weights(model_dir):
    ok = True
    if not os.path.isdir(model_dir):
        print(f"FAIL weights: {model_dir} is not a directory")
        return False
    for f in MODEL_FILES:
        p = os.path.join(model_dir, f)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            print(f"FAIL missing/empty: {f}")
            ok = False
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    total = 0
    try:
        with open(idx) as fh:
            wm = json.load(fh)["weight_map"]
        shards = sorted(set(wm.values()))
        for s in shards:
            p = os.path.join(model_dir, s)
            size = os.path.getsize(p) if os.path.isfile(p) else 0
            if size <= 0:
                print(f"FAIL shard missing/empty: {s}")
                ok = False
            else:
                total += size
                print(f"{s} {size / 2**30:.2f} GiB")
        print(f"{len(shards)} shards, {len(wm)} tensors, {total / 2**30:.2f} GiB")
    except Exception as e:
        print(f"FAIL index: {type(e).__name__}: {e}")
        ok = False
    if os.path.exists(os.path.join(model_dir, ".cache")):
        print("WARN .cache present in model dir (wasted layer space)")
    if ok:
        print("weights OK")
    return ok


def main(argv):
    if not argv or argv[0] not in ("deps", "ocr", "weights", "all"):
        print(__doc__)
        return 2
    cmd = argv[0]
    mdir = argv[1] if len(argv) > 1 else DEFAULT_MODEL
    results = {}
    if cmd in ("deps", "all"):
        results["deps"] = check_deps()
    if cmd in ("ocr", "all"):
        results["ocr"] = check_ocr()
    if cmd in ("weights", "all"):
        results["weights"] = check_weights(mdir)
    print("RESULT", json.dumps(results))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
