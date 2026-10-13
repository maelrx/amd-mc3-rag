"""Tunables with environment overrides. Stdlib only."""

from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# paths (defaults match the starter kit)
OUTPUT_DIR = os.environ.get("MC2_OUTPUT_DIR", "/app/output")
INDEX_DIR = os.environ.get("MC3_INDEX_DIR", "/app/index")
MODEL_DIR = os.environ.get("MC3_MODEL", "/models/Qwen3-VL-8B-Instruct")
SOCK = os.environ.get("MC3_SOCK", "/tmp/mc3.sock")
READER = os.environ.get("MC3_READER", "auto")       # auto | stub | none
STATUS_PATH = os.environ.get("MC3_STATUS", "/tmp/mc3.status")   # json mirror of the server state

# time budgets (seconds)
INDEX_BUDGET_S = _f("MC3_INDEX_BUDGET_S", 360.0)    # --index wall budget from exec
OCR_BUDGET_S = _f("MC3_OCR_BUDGET_S", 240.0)        # client-side RapidOCR budget
VLM_INDEX_MARGIN_S = _f("MC3_VLM_MARGIN_S", 20.0)   # daemon stops transcribing at deadline - margin
ANSWER_SOFT_S = _f("MC3_ANSWER_SOFT_S", 22.0)       # daemon returns best-so-far
CLIENT_TIMEOUT_S = _f("MC3_CLIENT_TIMEOUT_S", 23.0) # socket timeout in app.py --query
FALLBACK_CAP_S = _f("MC3_FALLBACK_CAP_S", 2.0)
FALLBACK_LATEST_S = _f("MC3_FALLBACK_LATEST_S", 24.0)  # past this, write the refusal instead of trying
RETRY_MIN_REMAINING_S = _f("MC3_RETRY_MIN_S", 10.0)
COVERAGE_MIN = _f("MC3_COVERAGE_MIN", 0.4)        # below this, take a second look without the support record
COVERAGE_REFUSE = _f("MC3_COVERAGE_REFUSE", 0.0)   # >0: refuse a text-only, bridge-less answer whose support covers less
VQA_MIN_REMAINING_S = _f("MC3_VQA_MIN_S", 5.0)
LOO_MIN_REMAINING_S = _f("MC3_LOO_MIN_S", 8.0)

# retrieval
BM25_K1 = _f("MC3_BM25_K1", 1.5)
BM25_B = _f("MC3_BM25_B", 0.75)
RRF_K = _i("MC3_RRF_K", 60)
RARE_DF = _i("MC3_RARE_DF", 5)
BM25_TOP_RECORDS = _i("MC3_BM25_TOP", 150)
PKG_FILES = (_i("MC3_PKG_FILES_1", 8), _i("MC3_PKG_FILES_2", 12))
PKG_NEIGHBORS = (_i("MC3_PKG_NEIGH_1", 2), _i("MC3_PKG_NEIGH_2", 4))
PKG_TOKENS = (_i("MC3_PKG_TOKENS_1", 8000), _i("MC3_PKG_TOKENS_2", 14000))
PKG_WHOLE_FILE_MAX_RECORDS = _i("MC3_PKG_WHOLE", 40)
CHARS_PER_TOKEN = _f("MC3_CHARS_PER_TOKEN", 3.5)
MAX_HOPS = _i("MC3_MAX_HOPS", 2)
MAX_IMAGES = _i("MC3_MAX_IMAGES", 2)

# reader
MAX_NEW_TOKENS = _i("MC3_MAX_NEW_TOKENS", 96)
TRANSCRIBE_MAX_NEW_TOKENS = _i("MC3_TRANSCRIBE_TOKENS", 512)
IMAGE_LONGEST_EDGE_READ = _i("MC3_IMG_EDGE_READ", 1_228_800)      # pixels (~1200 visual tokens)
IMAGE_LONGEST_EDGE_TRANSCRIBE = _i("MC3_IMG_EDGE_TRANSCRIBE", 2_097_152)
OCR_THREADS = _i("MC3_OCR_THREADS", 4)
OCR_MIN_CONF = _f("MC3_OCR_MIN_CONF", 0.85)

# gates
GATE_B_MODE = os.environ.get("MC3_GATE_B", "0")    # "0" off | "log" compute + record, never refuse | "1" enforce
GATE_B = GATE_B_MODE == "1"
GATE_B_THETA1 = _f("MC3_GATE_B_T1", -0.7)
GATE_B_THETA2 = _f("MC3_GATE_B_T2", 2.0)

# hypothesis flags (A/B against v5; every default reproduces v5 exactly). Always read as config.X at call time.
XLSX_JOIN = os.environ.get("MC3_XLSX_JOIN", "0") == "1"      # H1: cross-sheet join records + cross-sheet formula refs
PDF_COLS = os.environ.get("MC3_PDF_COLS", "0") == "1"        # H5: per-product records from PDF family tables
RANK = os.environ.get("MC3_RANK", "default")                 # H4: "cov_first" = coverage before hop target
CHAIN_GUARD = os.environ.get("MC3_CHAIN_GUARD", "0") == "1"  # H2: named-kind chain guard against topical decoys
LABEL_STRIP = os.environ.get("MC3_LABEL_STRIP", "0") == "1"  # H7: drop a printed field label the question names
DATE_COND = os.environ.get("MC3_DATE_COND", "0") == "1"      # H8: a date in the question must appear in the support record or its path

# H6: model "warrant" check after the verified answer (YES/NO question to the reader). Read as config.X at call time.
WARRANT = os.environ.get("MC3_WARRANT", "0")         # "0" off | "log" compute + record, never refuse | "1" enforce
WARRANT_THETA = _f("MC3_WARRANT_T", 0.0)             # refuse when log P(YES) - log P(NO) < theta

# confidence values reported (never scored)
CONF_SINGLE = 0.9
CONF_CHAIN = 0.8
CONF_IMAGE = 0.6
