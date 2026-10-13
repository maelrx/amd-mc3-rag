#!/usr/bin/env bash
# AMD free-notebook session runner (ROCm, no Docker). Budget: < 90 min of the 3 h/day quota; idempotent.
#
#   bash gpu/amd_session.sh                    # phases 0..5
#   PHASES="0 1 2" bash gpu/amd_session.sh     # subset (space or comma separated)
#
#   0 env      GPU name, python/torch versions, pick persistent dir P (/persistent, else /workspace)
#   1 venv     $P/venv --system-site-packages; pip install --no-deps -r app/requirements.txt; assert +rocm
#   2 weights  HF reachability probe (120 s); ok -> snapshot_download in background (log + progress);
#              blocked -> print what to do and exit 2 (Colab covers the functional validation)
#   3 bench    gpu/bench.py (VRAM sampler = amd-smi/rocm-smi every 2 s) -> $P/runs/<ts>/bench.json
#   4 suite    eval/run_suite.py on sample-corpus and every eval/suites/*/ with questions.json
#   5 pack     tar the run dir to $P/runs/<ts>.tgz and tell you to turn the session off
# Env: MC3_P (force persistent dir)  MC3_TS / MC3_NEW_RUN=1 (run dir)  MC3_BUDGET_MIN (default 85)
#      MC3_WEIGHTS_TIMEOUT_S (default 2400)  HF_ENDPOINT (HF mirror, if your notebook has one)
# Everything is logged to $P/runs/<ts>/session.log.

set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASES="${PHASES:-0 1 2 3 4 5}"
PHASES="${PHASES//,/ }"
MODEL_REPO="Qwen/Qwen3-VL-8B-Instruct"
MODEL_REV="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
BUDGET_MIN="${MC3_BUDGET_MIN:-85}"
WEIGHTS_TIMEOUT_S="${MC3_WEIGHTS_TIMEOUT_S:-2400}"
T_START=$(date +%s)

# ---------------------------------------------------------------- persistent dir, run dir, logging
if [ -n "${MC3_P:-}" ]; then
  P="$MC3_P"
elif [ -d /persistent ] && [ -w /persistent ]; then
  P=/persistent
elif [ -d /workspace ] && [ -w /workspace ]; then
  P=/workspace
else
  P="$HOME/mc3-persist"
  echo "WARN: neither /persistent nor /workspace is writable; using $P (may NOT survive the session)"
fi
mkdir -p "$P/runs" "$P/models" || { echo "cannot create dirs under $P"; exit 1; }
if [ -n "${MC3_TS:-}" ]; then
  TS="$MC3_TS"
elif [ "${MC3_NEW_RUN:-0}" != "1" ] && [ -f "$P/runs/CURRENT" ]; then
  TS="$(cat "$P/runs/CURRENT")"
else
  TS="$(date +%Y%m%d-%H%M%S)"
fi
echo "$TS" > "$P/runs/CURRENT"
RUN="$P/runs/$TS"
mkdir -p "$RUN"
LOG="$RUN/session.log"
exec > >(tee -a "$LOG") 2>&1

MODEL="$P/models/Qwen3-VL-8B-Instruct"
VENV="$P/venv"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MC3_MODEL="$MODEL"
export PYTHONPATH="$REPO/app:$REPO${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$P/site-packages" ] && export PYTHONPATH="$P/site-packages:$PYTHONPATH"   # fallback install target of phase 1

log()     { echo "[$(date +%H:%M:%S)] $*"; }
banner()  { echo; echo "=================== $* ==================="; }
elapsed_min() { echo $(( ( $(date +%s) - T_START ) / 60 )); }
want()    { for p in $PHASES; do [ "$p" = "$1" ] && return 0; done; return 1; }
pybin()   { if [ -x "$VENV/bin/python" ]; then echo "$VENV/bin/python"; else echo python3; fi; }
py()      { "$(pybin)" "$@"; }
turn_off() { echo; echo "############################################################"; echo "#  TURN OFF THE SESSION NOW  (Turn-off Session in the UI)  #"; echo "############################################################"; }
within_budget() { # $1 = minutes of headroom needed
  [ "$(elapsed_min)" -le $(( BUDGET_MIN - $1 )) ]
}

weights_complete() {
  py - "$MODEL" <<'PY'
import json, sys, pathlib
m = pathlib.Path(sys.argv[1])
try:
    idx = json.loads((m / "model.safetensors.index.json").read_text())
    need = set(idx["weight_map"].values()) | {"config.json", "preprocessor_config.json", "tokenizer.json"}
    miss = sorted(f for f in need if not (m / f).exists())
    print("missing:", miss) if miss else print("complete:", len(need), "files")
    sys.exit(1 if miss else 0)
except Exception as e:
    print("incomplete:", e); sys.exit(1)
PY
}

log "repo=$REPO  P=$P  run=$RUN  phases='$PHASES'  budget=${BUDGET_MIN} min"

# ---------------------------------------------------------------- phase 0: env
if want 0; then
  banner "phase 0: env"
  date; uname -a
  echo "--- GPU"
  (rocm-smi --showproductname 2>&1 | head -20) || true
  (amd-smi static --asic 2>&1 | head -30) || true
  (rocm-smi --showmeminfo vram 2>&1 | head -12) || true
  echo "--- python / torch"
  python3 --version
  python3 - <<'PY' || echo "WARN: torch import failed"
import torch, sys
print("torch", torch.__version__, "| hip", getattr(torch.version, "hip", None), "| cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, "| arch", getattr(p, "gcnArchName", "?"), "| VRAM GiB", round(p.total_memory / 2**30, 1))
print("+rocm tag:", "+rocm" in torch.__version__)
try:
    import transformers; print("transformers", transformers.__version__)
except Exception as e:
    print("transformers: not installed (phase 1 installs it)")
PY
  echo "--- persistent dir / disk / net env"
  df -h "$P" | tail -2
  free -g | head -2 || true
  env | grep -iE '^(http|https|no)_proxy=|^PIP_|^HF_|^HUGGING' | sed -E 's#(://)[^@/]*@#\1***@#' || true
  pip config list 2>/dev/null || true
fi

# ---------------------------------------------------------------- phase 1: venv + deps
if want 1; then
  banner "phase 1: venv + requirements (--no-deps)"
  REQ="$REPO/app/requirements.txt"
  [ -f "$REQ" ] || { log "ERROR $REQ not found (is the repo tar extracted next to gpu/?)"; exit 1; }
  PIPMODE=venv
  if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    rm -rf "$VENV"
    if python3 -m venv --system-site-packages "$VENV" && "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
      log "created $VENV"
    else
      rm -rf "$VENV"
      PIPMODE=target
      log "WARN: venv (with pip) unavailable here -> fallback: pip install --no-deps --target $P/site-packages (+PYTHONPATH)"
    fi
  fi
  TORCH_BEFORE="$(py -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)"
  SUM="$(sha1sum "$REQ" | cut -c1-12)"
  MARK="$VENV/.req-$SUM"; [ "$PIPMODE" = target ] && MARK="$P/site-packages/.req-$SUM"
  mkdir -p "$(dirname "$MARK")"
  if [ -f "$MARK" ]; then
    log "requirements $SUM already installed"
  else
    PIP=("$(pybin)" -m pip install --no-cache-dir --no-deps --disable-pip-version-check --default-timeout 60 --retries 3)
    [ "$PIPMODE" = target ] && PIP+=(--upgrade --target "$P/site-packages")
    log "pip install --no-deps -r app/requirements.txt  (torch before: $TORCH_BEFORE; proxy may be slow)"
    if "${PIP[@]}" -r "$REQ"; then
      touch "$MARK"
    else
      log "bulk install failed -> one requirement at a time to find the culprit"
      FAILED=()
      while IFS= read -r line; do
        line="${line%%#*}"; line="$(echo "$line" | xargs 2>/dev/null || true)"
        [ -z "$line" ] && continue
        case "$line" in -*) continue ;; esac
        "${PIP[@]}" "$line" || FAILED+=("$line")
      done < "$REQ"
      if [ "${#FAILED[@]}" -gt 0 ]; then
        log "FAILED requirements: ${FAILED[*]}"
      else
        touch "$MARK"
      fi
    fi
  fi
  TORCH_AFTER="$(py -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)"
  log "torch after: $TORCH_AFTER"
  if ! py -c "import torch,sys; sys.exit(0 if '+rocm' in torch.__version__ else 1)"; then
    log "ERROR: torch is not a +rocm build ($TORCH_AFTER). Either this is not the ROCm notebook, or a pip step replaced it (never pip install without --no-deps here)."
    exit 1
  fi
  py - <<'PY' || log "WARN: some imports failed (see above); bench will say what breaks"
import importlib
bad = []
for m in ("transformers", "accelerate", "safetensors", "tokenizers", "huggingface_hub", "PIL", "pypdf", "pypdfium2",
          "openpyxl", "onnxruntime", "rapidocr", "cv2"):
    try:
        mod = importlib.import_module(m); print(f"  {m:<18} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        bad.append(m); print(f"  {m:<18} FAILED {type(e).__name__}: {e}")
raise SystemExit(1 if bad else 0)
PY
fi

# ---------------------------------------------------------------- phase 2: HF reachability + weights
if want 2; then
  banner "phase 2: Hugging Face reachability + weights"
  if weights_complete; then
    log "weights already complete at $MODEL (skipping download)"
  else
    log "probing HF (120 s limit): config.json @ ${MODEL_REV:0:8}"
    if timeout 120 "$(pybin)" -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen3-VL-8B-Instruct','config.json', revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', local_dir='$P/models/Qwen3-VL-8B-Instruct')"; then
      log "HF reachable -> full snapshot_download in background (log: $RUN/weights.log)"
      cat > "$P/dl_weights.py" <<PY
import time
from huggingface_hub import snapshot_download
t0 = time.time()
for kind, kw in (("meta", {"ignore_patterns": ["*.safetensors"]}), ("shards", {"allow_patterns": ["*.safetensors"]})):
    snapshot_download("$MODEL_REPO", revision="$MODEL_REV", local_dir="$MODEL", max_workers=8, **kw)
    print(f"[weights] {kind} done at {time.time()-t0:.0f}s", flush=True)
print("[weights] DONE", flush=True)
PY
      nohup "$(pybin)" -u "$P/dl_weights.py" > "$RUN/weights.log" 2>&1 &
      DLPID=$!
      WAITED=0
      while kill -0 "$DLPID" 2>/dev/null; do
        sleep 20; WAITED=$((WAITED + 20))
        log "downloading... $(du -sm "$MODEL" 2>/dev/null | cut -f1) MiB of ~17500 MiB  (${WAITED}s)"
        if [ "$WAITED" -ge "$WEIGHTS_TIMEOUT_S" ]; then
          log "ERROR: download exceeded ${WEIGHTS_TIMEOUT_S}s -> killing. Re-run with PHASES='2 3 4' to resume (HF resumes partial files)."
          kill "$DLPID" 2>/dev/null; turn_off; exit 3
        fi
      done
      tail -3 "$RUN/weights.log"
      rm -rf "$MODEL/.cache"
      if weights_complete; then
        log "weights OK"
      else
        log "ERROR: weights incomplete after download (see $RUN/weights.log)"; turn_off; exit 3
      fi
    else
      cat <<EOF

!! Hugging Face is NOT reachable from this notebook (probe failed or hit the 120 s timeout).
   What to do:
   * Do not burn the quota here. The functional validation (10/10 on the sample, suites, canaries, bench)
     is covered by the Colab GPU session (CUDA); ROCm is covered only by the SDPA self-test + a 2x latency margin.
   * If you have an HF mirror/proxy for this notebook: export HF_ENDPOINT=https://<mirror> and re-run:
         PHASES="2 3 4 5" bash gpu/amd_session.sh
   * If you can get the weights onto the notebook another way (notebook file upload, rclone/Drive, scp),
     put the Qwen3-VL-8B-Instruct directory (rev ${MODEL_REV:0:8}) at:
         $MODEL
     then re-run  PHASES="3 4 5" bash gpu/amd_session.sh
   * Otherwise: Turn-off Session, and rely on Colab.
   Probe log and env are in $LOG
EOF
      turn_off
      exit 2
    fi
  fi
fi

# ---------------------------------------------------------------- phase 3: bench
if want 3; then
  banner "phase 3: bench (ROCm)  [elapsed $(elapsed_min) min]"
  if ! weights_complete >/dev/null; then
    log "SKIP bench: weights not at $MODEL (phase 2)"
  elif ! within_budget 25; then
    log "SKIP bench: only $((BUDGET_MIN - $(elapsed_min))) min of budget left"
  else
    (cd "$REPO" && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 py gpu/bench.py --model "$MODEL" --out "$RUN/bench.json" --skip-suite)
    log "bench exit code $? (3 = ran, a gate failed)"
  fi
fi

# ---------------------------------------------------------------- phase 4: suites
if want 4; then
  banner "phase 4: suites  [elapsed $(elapsed_min) min]"
  if ! weights_complete >/dev/null; then
    log "SKIP suites: weights not at $MODEL"
  elif [ ! -f "$REPO/eval/run_suite.py" ]; then
    log "SKIP suites: eval/run_suite.py not present"
  else
    SUITES=("sample|$REPO/sample-corpus|$REPO/sample-questions.json")
    for d in "$REPO"/eval/suites/*/; do
      [ -f "${d}questions.json" ] || continue
      c="$d"; for cand in corpus mc3-corpus; do [ -d "${d}$cand" ] && { c="${d}$cand"; break; }; done
      SUITES+=("$(basename "$d")|${c%/}|${d}questions.json")
    done
    for s in "${SUITES[@]}"; do
      IFS='|' read -r name corpus questions <<< "$s"
      within_budget 8 || { log "SKIP suite $name: budget"; continue; }
      mkdir -p "$RUN/$name"
      log "--- suite $name"
      (cd "$REPO" && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 py eval/run_suite.py --corpus "$corpus" --questions "$questions" \
          --out "$RUN/$name" --reader real --model "$MODEL") 2>&1 | tee "$RUN/$name/suite_stdout.txt"
      log "suite $name: $(grep '^TOTAL ' "$RUN/$name/suite_stdout.txt" | tail -1)"
    done
  fi
fi

# ---------------------------------------------------------------- phase 5: pack
if want 5; then
  banner "phase 5: pack logs  [elapsed $(elapsed_min) min]"
  # keep this last: session.log is still being written by tee, so snapshot after a flush
  sleep 1
  tar -czf "$P/runs/$TS.tgz" -C "$P/runs" "$TS" && ls -l "$P/runs/$TS.tgz"
  log "download it from the notebook file browser: $P/runs/$TS.tgz   (and $RUN/bench.json)"
  log "total elapsed: $(elapsed_min) min"
  turn_off
fi
