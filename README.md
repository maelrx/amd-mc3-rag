# Proof-Carrying RAG on ROCm (AMD AI League, Match 3)

A retrieval-augmented question answering container for AMD GPUs. Every answer is treated as a claim that must be
proven from the corpus before it is written; citations come from record identity and chain necessity.

- **Model**: Qwen3-VL-8B-Instruct, bf16, greedy, resident behind a unix socket (`app/server.py`, PID 1 supervisor).
- **Client**: `app/app.py` is a thin stdlib client; it always writes a valid `{"answer","citations","confidence"}` JSON
  within the 30 s budget (refusal-first atomic write, deterministic fallback).
- **Indexing** (`app/parse.py`, `app/index.py`, `app/vision.py`): defensive corpus walk (unreadable, encrypted,
  unknown and empty inputs are recorded, never fatal); PDF, DOCX tables, XLSX sheets, CSV, logs, Python literals and
  images become atomic records; RapidOCR on CPU plus VLM transcription for images.
- **Retrieval** (`app/retrieve.py`): BM25 fused with an exact rare-ID index (RRF), ID hops across files for chains,
  deterministic revision supersession (withdrawn / draft / stale rows / hidden sheets).
- **Verification** (`app/verify.py`, `app/cite.py`, `app/snap.py`): token-boundary span in an eligible record,
  question conditions (product, quantity, unit) in the same record, code literal over stale comment, false-premise
  and withdrawn-only checks; the final answer is the verbatim source span cleaned to the bare value.

## v6 additions (on in the image via ENV; code defaults reproduce v5)

| Flag | What it does |
|---|---|
| `MC3_XLSX_JOIN=1` | Cross-sheet joins in workbooks: a row keyed only by part number (lead times, costs) is joined to the row of another sheet that names the product, so "lead time of the PSU for the TQ-40" is answerable; cross-sheet formulas are evaluated against the referenced sheet. |
| `MC3_PDF_COLS=1` | PDF family tables with one column per product become per-product records. |
| `MC3_CHAIN_GUARD=1` | When the question names a source kind ("the production log shows ...") and the answer came from a topical decoy, re-read the ID chain (log line → ticket) and cite every link. |
| `MC3_WARRANT=1`, `MC3_WARRANT_T=-6` | After the deterministic verifier accepts an answer, the model is asked a YES/NO question over the evidence; log P(YES) − log P(NO) below −6 refuses (unanswerable questions stop leaking). |
| `MC3_LABEL_STRIP=1` | A printed field label that the question itself names ("BIN D-14-2" for "which bin location") is stripped from the value. |

Validated on the real model (A100/G4) against v5 over 13 suites (~1,200 questions, generated adversarial corpora
plus held-out seeds and a 300-file / 61-image scale corpus): +45 questions, 0 regressions, false answers on
unanswerable questions 30 → 3, p95 5.4 s per question, peak VRAM 22.5 GiB. The ROCm stack was validated on a
Radeon Pro W7900D.

## Build and tooling

The image is assembled with `crane` on top of the mandated base
`rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` (`build/colab_build.py`, `build/append_app.sh`);
`Dockerfile` documents the equivalent layers. Dependencies are installed with `--no-deps` so torch stays `+rocm`.

Evaluation tooling: `eval/` (adversarial corpus generator, suite runner, A/B runner with a reader cache, warrant
calibration, harness simulator, scorer), `gpu/` (Colab/AMD validation sessions), `tests/` (pytest).

Tests: `python3 -m pytest -q tests` needs the starter kit's sample next to the repo root (not redistributed here):
copy `mc3-starter-kit/mc3-corpus` to `sample-corpus/` and `mc3-starter-kit/sample-questions.json` to the root, then
run `mc3-starter-kit/setup.sh`-style fixups (empty `archive/`, `chmod 000 vendor/internal_audit.txt`). Generated
suites go to `eval/suites/` via `python3 eval/gen_corpus.py --seed N --out eval/suites/<name>`.
