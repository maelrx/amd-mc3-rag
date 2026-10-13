#!/usr/bin/env bash
# Push only the app layer: REF:heavy-noapp (deps+model) + 20-app.tar -> REF:TAG.
#   build/append_app.sh docker.io/<user>/mc3 /path/to/repo v1
# SRC_DIR must contain app/ (and optionally build/checks.py, shipped as /app/checks.py).
# Needs crane on PATH and an authenticated registry (crane auth login ...). Works locally and on Colab:
# `crane append -b <remote ref>` only reads the base manifest/config, no layer is downloaded.
set -euo pipefail

if [ $# -ne 3 ]; then
  echo "usage: $0 REF SRC_DIR TAG   (e.g. docker.io/user/mc3 . v1)" >&2
  exit 2
fi
REF=$1; SRC=$2; TAG=$3
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
command -v crane >/dev/null || { echo "crane not on PATH" >&2; exit 1; }
[ -d "$SRC/app" ] || { echo "$SRC/app not found" >&2; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
# same tar rules as the full build (deterministic, no tests/__pycache__, empty corpus/output/index)
python3 "$HERE/colab_build.py" --app-tar "$SRC" "$WORK/20-app.tar"
(tar tf "$WORK/20-app.tar" || true) | head -5

crane append -b "$REF:heavy-noapp" -f "$WORK/20-app.tar" -t "$REF:$TAG"
# append keeps the base config; re-apply CMD/ENV/WORKDIR anyway (idempotent)
crane mutate "$REF:$TAG" --cmd python3,/app/server.py --workdir /app \
  --env MC3_MODEL=/models/Qwen3-VL-8B-Instruct --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONUNBUFFERED=1 --env TOKENIZERS_PARALLELISM=false --env OMP_NUM_THREADS=8 \
  -t "$REF:$TAG"
crane config "$REF:$TAG" | jq .config
echo "pushed $REF:$TAG  digest: $(crane digest "$REF:$TAG")"
