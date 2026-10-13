"""Make app/ importable as top-level modules, exactly like /app inside the container."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_CORPUS = ROOT / "sample-corpus"
SAMPLE_QUESTIONS = ROOT / "sample-questions.json"
