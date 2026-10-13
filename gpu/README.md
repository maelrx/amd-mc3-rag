# gpu/ - GPU validation sessions (Colab CUDA, AMD notebook ROCm)

| File | What |
|---|---|
| `colab_gpu.py` | runs inside the Colab Jupyter kernel; `MC3_PHASE` = `setup,weights,bench,suite,report` (idempotent, re-runnable) |
| `bench.py` | load / SDPA self-test / warm-up / transcription / read latency 4k-8k-14k / logprob cost / VRAM peak / gates / optional suite |
| `vram_sampler.py` | background VRAM sampler (torch + nvidia-smi, or amd-smi / rocm-smi); never raises |
| `amd_session.sh` | AMD notebook runner (phases 0-5, < 90 min, resumable) |

Gates (spec section 10, CUDA numbers): load <= 90 s, warm-up <= 30 s, read#1 p95 <= 8 s at 8k tokens,
transcription <= 30 s/image, VRAM peak <= 40 GiB. `bench.py` exits 0 (all PASS), 3 (a gate FAILED), 2 (bad args).

## A. Colab GPU (run from this repo's root on your machine)

Three things differ from what you might expect from a shell:

* The phase list must reach the **remote kernel**: use `--env MC3_PHASE=...` (a `MC3_PHASE=... colab exec` prefix only sets
  it locally and is ignored).
* `colab exec` has a **30 s default timeout**: always pass `--timeout 3600` (weights/pip/suite are long).
* `src.tar` is built with `--ignore-failed-read` because the sample corpus has a `chmod 000` trap file that `tar` cannot read
  as a normal user (it is simply left out of the tar; Colab runs as root, so the trap would not bite there anyway).

```bash
# 0. (optional) adversarial suites; eval/suites/ must exist BEFORE the tar so the suite phase finds them
python3 eval/gen_corpus.py --seed 1 --out eval/suites/s1

# 1. pack the code (re-run this + step 3 after every local change)
tar --ignore-failed-read --exclude=__pycache__ --exclude='*.pyc' -cf /tmp/mc3-src.tar \
    app eval gpu sample-corpus sample-questions.json

# 2. allocate (A100 40 GB preferred ~11 CU/h; on a 400 error use L4 24 GB ~2.5 CU/h; T4 is not enough)
colab new -s mc3-gpu --gpu A100          # fallback: colab new -s mc3-gpu --gpu L4
colab status -s mc3-gpu                  # confirm the hardware
echo "import os; os.makedirs('/content/mc3', exist_ok=True)" | colab exec -s mc3-gpu

# 3. upload code
colab upload -s mc3-gpu /tmp/mc3-src.tar /content/mc3/src.tar

# 4. deps + weights (pip pins from app/requirements.txt, torch frozen; snapshot_download 8B @ 0c351dd, ~17.5 GB)
colab exec -s mc3-gpu --timeout 3600 --env MC3_PHASE=setup,weights -f gpu/colab_gpu.py

# 5. bench -> /content/mc3/runs/<ts>/bench.json
colab exec -s mc3-gpu --timeout 3600 --env MC3_PHASE=bench -f gpu/colab_gpu.py

# 6. suites: sample-corpus, then every eval/suites/*/ that has questions.json  (--reader real)
colab exec -s mc3-gpu --timeout 3600 --env MC3_PHASE=suite -f gpu/colab_gpu.py

# 7. one summary table + /content/mc3/runs/latest.json (bench + every report.json / query_NN_output.json embedded)
colab exec -s mc3-gpu --timeout 600 --env MC3_PHASE=report -f gpu/colab_gpu.py
colab download -s mc3-gpu /content/mc3/runs/latest.json runs/colab-latest.json

# 8. ALWAYS release the VM (billed while alive)
colab stop -s mc3-gpu
```

Everything in one go: `--env MC3_PHASE=all` (= setup,weights,bench,suite,report).

Iterating after a fix: repeat steps 1 and 3, then
`colab exec -s mc3-gpu --timeout 3600 --env MC3_PHASE=suite,report --env MC3_UPDATE=1 -f gpu/colab_gpu.py`
(`MC3_UPDATE=1` re-untars `src.tar`; a changed tar is also detected automatically). Weights, pip and the run directory are kept.

Other knobs (pass with `--env K=V`; `MC3_UPDATE` and `MC3_NEW_RUN` are one-shot):

| Var | Effect |
|---|---|
| `MC3_NEW_RUN=1` | start a fresh `runs/<ts>` (default: reuse the session's current run, so bench + suite + report land together) |
| `MC3_SUITES=sample,s1` | restrict the suite phase (`sample` = sample-corpus) |
| `MC3_BENCH_SUITE=1` | also run the sample suite inside bench.py (otherwise bench runs `--skip-suite`) |
| `MC3_SKIP_PIP=1` | skip pip in `setup` |
| `MC3_BASE=/content/mc3` | state root (logs: `runs/<ts>/bench.log`, `<suite>/suite_stdout.txt`) |

Look at progress / state: `colab ls -s mc3-gpu /content/mc3/runs`, `colab log -s mc3-gpu -n 20`.

## B. AMD free notebook (ROCm; 3 h/day, session idle time burns quota)

1. On your machine make the tar (step 1 above; add `eval/suites` only if you generated suites) and upload it through the
   notebook file browser (or any route you have), then in the notebook **terminal**:

```bash
cd /persistent 2>/dev/null || cd /workspace          # whichever exists and persists
mkdir -p mc3 && tar xf ~/mc3-src.tar -C mc3          # adjust to where the upload landed
cd mc3
bash gpu/amd_session.sh                              # phases 0 1 2 3 4 5, log: $P/runs/<ts>/session.log
```

2. Subsets / resume (idempotent: the venv, the pip marker and complete weights are reused):

```bash
PHASES="0 1 2" bash gpu/amd_session.sh      # env + deps + HF probe/download only (cheap first look)
PHASES="3 4 5"  bash gpu/amd_session.sh     # bench + suites + pack, weights already in $P/models
MC3_NEW_RUN=1 bash gpu/amd_session.sh       # new run dir (default: continue $P/runs/CURRENT)
```

What each phase does: 0 prints the GPU (rocm-smi/amd-smi), python/torch versions and chooses `P` (`/persistent`, else
`/workspace`); 1 builds `$P/venv --system-site-packages`, `pip install --no-deps -r app/requirements.txt` (the image's exact pins,
the proxy may be slow; falls back to `--target $P/site-packages` if venv/ensurepip is missing) and **asserts `+rocm`**;
2 probes Hugging Face with `hf_hub_download(config.json)` under `timeout 120`: on success it downloads the full
snapshot in the background (progress every 20 s, log `weights.log`), otherwise it prints what to do and **exits 2**
(Colab then covers the functional validation; ROCm stays covered by the SDPA self-test and the x2 latency margin);
3 runs `gpu/bench.py --skip-suite` (the sampler uses `amd-smi metric --mem --json`, fallback `rocm-smi`); 4 runs the
sample suite and every `eval/suites/*/`; 5 writes `$P/runs/<ts>.tgz` and prints **TURN OFF THE SESSION NOW**.
Exit codes: 1 = hard failure (no venv/torch not +rocm), 2 = HF blocked, 3 = weights download failed/timed out.

3. Bring results back: download `$P/runs/<ts>.tgz` (contains `bench.json`, `vram.csv`, suite reports, `session.log`) from the file
   browser, then press **Turn-off Session**.

Budget guard: `MC3_BUDGET_MIN` (default 85) - bench is skipped when fewer than 25 min remain, each suite when fewer than 8.
HF mirror: `export HF_ENDPOINT=https://<mirror>` before phase 2. Download timeout: `MC3_WEIGHTS_TIMEOUT_S` (default 2400).

## C. Running bench.py by hand

```bash
PYTHONPATH=app:. python3 gpu/bench.py --model /path/Qwen3-VL-8B-Instruct --out runs/$(date +%Y%m%d-%H%M%S)/bench.json [--corpus sample-corpus] [--skip-suite] [--reps 5]
```

Output: live table, `bench.json` (rewritten after each phase so a crash keeps earlier phases), `vram.csv` (2 s samples),
`vram_suite.csv` + `suite_sample/` when the suite runs. Reader methods that swallow their own errors (`read`, `transcribe`,
`warmup`, `self_test_sdpa`) are detected and recorded as phase errors, and errored reads never count towards the latency gates.
