#!/usr/bin/env bash
# OPTIONAL smoke test of the REAL pushed image on a Colab CPU VM, without Docker, using udocker (PRoot).
#   bash udocker_smoke.sh docker.io/<user>/mc3:v1 [/content/sample-corpus] [/content/sample-questions.json]
# Proves: anonymous pull works (udocker has no credentials here), the cp314 imports resolve, server.py starts
# in CPU mode (no GPU -> no model, deterministic fallback), app.py --index and --query write a valid JSON.
# It does NOT prove GPU/ROCm behaviour. Cannot be tested on the dev machine; read it before trusting it.
#
# DISK: pull stores the compressed layers (~37 GB) and `create` extracts them (~47 GB). A 100 GB Colab disk
# fits only with nothing else on it: the script aborts if < 90 GB are free. Run it on a fresh VM (or after
# `colab_build.py` finished and its stage-*/tars dirs are deleted).
# PRoot may fail on some syscalls: if `udocker run` errors out, try `udocker setup --execmode=F1 mc3` or give up
# and rely on the local cp314 venv test + Colab GPU run.
set -uo pipefail

IMG=${1:?usage: $0 REF:TAG [SAMPLE_CORPUS] [SAMPLE_QUESTIONS_JSON]}
CORPUS=${2:-/content/sample-corpus}
QUESTIONS=${3:-/content/sample-questions.json}
NAME=mc3
OUT=/content/mc3-smoke-out
[ -d "$CORPUS" ] || { echo "sample corpus $CORPUS not found (colab upload it)"; exit 1; }
mkdir -p "$OUT"

FREE_GB=$(df -BG --output=avail /content | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge 90 ] || { echo "only ${FREE_GB} GB free, need >= 90"; exit 1; }

pip install -q udocker && udocker install >/dev/null 2>&1 || true   # udocker downloads its own PRoot tarball
export UDOCKER_LOGLEVEL=2

udocker pull "$IMG"                         # anonymous pull = proof the repo/package is public
udocker rm -f $NAME >/dev/null 2>&1 || true
udocker create --name=$NAME "$IMG"

# udocker maps host /tmp, so the unix socket /tmp/mc3.sock is shared by separate `udocker run` calls
# (same as the harness' docker exec into one container). Corpus and output are bind-mounted.
RUN=(udocker run --env=MC2_OUTPUT_DIR=/app/output --env=MC3_INDEX_DIR=/app/index
     -v "$CORPUS":/app/corpus -v "$OUT":/app/output $NAME)

# 1) what docker would do at start: the image CMD, in the background
"${RUN[@]}" python3 /app/server.py > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

# 2) the same checks the build runs, inside the real image (torch must report +rocm)
"${RUN[@]}" python3 /app/checks.py deps || echo "WARN checks deps failed (see above)"
"${RUN[@]}" python3 /app/checks.py ocr  || echo "WARN checks ocr failed"
"${RUN[@]}" python3 /app/checks.py weights /models/Qwen3-VL-8B-Instruct || echo "WARN checks weights failed"

# 3) index (CPU), as the harness does once
time "${RUN[@]}" python3 /app/app.py --index /app/corpus

# 4) one query per question from the sample file (first 3), fresh process each time
python3 - "$QUESTIONS" <<'PY' > "$OUT/queries.tsv"
import json, sys
d = json.load(open(sys.argv[1]))
for i, q in enumerate(d["queries"][:3], 1):
    print(f"query_{q.get('n', i):02d}\t{q['query']}")
PY
while IFS=$'\t' read -r QID Q; do
  echo "== $QID: $Q"
  "${RUN[@]}" python3 /app/app.py --corpus /app/corpus --query-id "$QID" --query "$Q"
  cat "$OUT/${QID}_output.json"; echo
done < "$OUT/queries.tsv"

echo "server log tail:"; tail -n 20 "$OUT/server.log"
