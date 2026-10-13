# REFERENCE ONLY: never built here (no Docker on the dev machine, disk too small). The real image is assembled
# with `crane append` by build/colab_build.py, whose layers mirror this file one to one:
#   00-deps.tar (pip install) / 10-model-meta.tar / 11..14-shard N.tar / 20-app.tar (COPY app/ + checks.py).
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=8 \
    MC3_MODEL=/models/Qwen3-VL-8B-Instruct HF_HOME=/tmp/hf
WORKDIR /app

# --no-deps is mandatory: otherwise pip swaps the base torch 2.13.0+rocm for a CUDA build
COPY app/requirements.txt build/checks.py /app/
RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt && python3 /app/checks.py deps
RUN python3 /app/checks.py ocr

ARG MODEL_REPO=Qwen/Qwen3-VL-8B-Instruct
ARG MODEL_REV=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
RUN python3 -c "from huggingface_hub import snapshot_download as d; d('$MODEL_REPO', revision='$MODEL_REV', local_dir='$MC3_MODEL', allow_patterns=['*.json', '*.txt'])" \
 && rm -rf $MC3_MODEL/.cache /tmp/hf
RUN python3 -c "from huggingface_hub import snapshot_download as d; d('$MODEL_REPO', revision='$MODEL_REV', local_dir='$MC3_MODEL', allow_patterns=['model-00001-of-00004.safetensors'])" \
 && rm -rf $MC3_MODEL/.cache /tmp/hf
RUN python3 -c "from huggingface_hub import snapshot_download as d; d('$MODEL_REPO', revision='$MODEL_REV', local_dir='$MC3_MODEL', allow_patterns=['model-00002-of-00004.safetensors'])" \
 && rm -rf $MC3_MODEL/.cache /tmp/hf
RUN python3 -c "from huggingface_hub import snapshot_download as d; d('$MODEL_REPO', revision='$MODEL_REV', local_dir='$MC3_MODEL', allow_patterns=['model-00003-of-00004.safetensors'])" \
 && rm -rf $MC3_MODEL/.cache /tmp/hf
RUN python3 -c "from huggingface_hub import snapshot_download as d; d('$MODEL_REPO', revision='$MODEL_REV', local_dir='$MC3_MODEL', allow_patterns=['model-00004-of-00004.safetensors'])" \
 && rm -rf $MC3_MODEL/.cache /tmp/hf
RUN python3 /app/checks.py weights $MC3_MODEL

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY app/ /app/
RUN python3 -m compileall -q /app && mkdir -p /app/corpus /app/output /app/index
HEALTHCHECK --interval=10s --timeout=3s --start-period=600s CMD test -S /tmp/mc3.sock || exit 1
CMD ["python3", "/app/server.py"]
