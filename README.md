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

Build: the image is assembled with `crane` on top of the mandated base
`rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` (`build/colab_build.py`, `build/append_app.sh`);
`Dockerfile` documents the equivalent layers. Dependencies are installed with `--no-deps` so torch stays `+rocm`.

Evaluation tooling: `eval/` (adversarial corpus generator, suite runner, harness simulator, scorer), `gpu/`
(Colab/AMD validation sessions), `tests/` (pytest).
