#!/usr/bin/env bash
# Launcher for the v0.4.0 Python 3.12 baseline image (see
# docker/Dockerfile.vllm018_py312 for the rationale). It keeps vLLM 0.18.0
# and torch 2.10.0+cu128 while enabling cudagraph decode on Python 3.12.
#
# First time:   ./docker/run_docker_vllm018_py312.sh build
# Interactive:  ./docker/run_docker_vllm018_py312.sh shell
# Serve:        ./docker/run_docker_vllm018_py312.sh serve \
#                   --model /mnt/models/Qwen2.5-7B-Instruct \
#                   --dtype float16 \
#                   --attention-backend TRITON_ATTN --tensor-parallel-size 1 \
#                   --max-num-seqs 1 --gpu-memory-utilization 0.80 \
#                   --max-model-len 32768 --no-enable-chunked-prefill \
#                   --host 0.0.0.0 --port 8002
#
# Mounts (every run):
#   <project root> -> /work                  (mounted rw)
#   /mnt/models    -> /mnt/models            (read-only)
#   ~/.cache/vllm018-py312-* -> compile caches inside the container
#                                (separate from both the cu128 image's caches
#                                AND 1catai's caches; py3.10 vs py3.12 + torch
#                                ABI differences mean cross-image cache reuse
#                                would risk stale-hit corruption)
#
# Ports:
#   $PORT (default 8002) -> $PORT inside the container.
#   (Default differs from the cu128 image's 8000 so this can run alongside
#   our existing serves without colliding.)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-vllm-v100-py312-test:cu128}"

MODEL_MOUNT=()
if [[ -d /mnt/models ]]; then
    MODEL_MOUNT=(-v /mnt/models:/mnt/models:ro)
fi

# Persistent cache mounts for this image. Kept fully separate from the cu128
# image's caches (~/.cache/vllm-*) and the 1catai image's caches
# (~/.cache/1catai-vllm-*) because the cache keys depend on python +
# torch version; cross-image reuse would risk stale-hit corruption.
# `extensions` is only used by serve-fp8 mode (JIT-compiled fp8_dequant .cu)
# but we mount it unconditionally for simplicity.
for sub in triton torch torchinductor extensions; do
    mkdir -p "$HOME/.cache/vllm018-py312-$sub"
done
CACHE_MOUNTS=(
    -v "$HOME/.cache/vllm018-py312-triton:/root/.triton"
    -v "$HOME/.cache/vllm018-py312-torch:/root/.cache/torch"
    -v "$HOME/.cache/vllm018-py312-extensions:/root/.cache/torch_extensions"
    -v "$HOME/.cache/vllm018-py312-torchinductor:/tmp/torchinductor_root"
)

VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

GPUS="${GPUS:-all}"
if [[ "$GPUS" == "all" ]]; then
    GPU_FLAG=(--gpus all)
else
    GPU_FLAG=(--gpus "\"device=$GPUS\"")
fi

PORT="${PORT:-8002}"

RUNTIME_ENV=(
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
    -e VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD"
)

# FP8_ENV: extra environment used only by `serve-fp8` mode. Mirrors the
# DEV_ENV array in docker/run_docker.sh so the fp8_w8a16_sm70 monkey-patches
# see exactly the same configuration knobs as on the cu128 stack. All have
# safe defaults; override any of them by exporting in the calling shell.
FP8_ENV=(
    -e PYTHONPATH=/work/src
    -e VLLM_V100_FP8_DEBUG_SHAPES="${VLLM_V100_FP8_DEBUG_SHAPES:-}"
    -e VLLM_V100_FP8_DEBUG_APPLY="${VLLM_V100_FP8_DEBUG_APPLY:-}"
    -e VLLM_V100_FP8_APPLY_MAG="${VLLM_V100_FP8_APPLY_MAG:-10000}"
    -e VLLM_V100_FP8_APPLY_WARN="${VLLM_V100_FP8_APPLY_WARN:-1000}"
    -e VLLM_V100_FP8_HASH_LAYERS="${VLLM_V100_FP8_HASH_LAYERS:-off}"
    -e FP8_WMMA_MIN_M="${FP8_WMMA_MIN_M:-64}"
    -e FP8_WMMA_COUNTER_LOG_EVERY="${FP8_WMMA_COUNTER_LOG_EVERY:-1000}"
    -e VLLM_V100_FP8_MOE_FALLBACK="${VLLM_V100_FP8_MOE_FALLBACK:-1}"
    -e VLLM_V100_FP8_MOE_ACTIVE_LIST="${VLLM_V100_FP8_MOE_ACTIVE_LIST:-1}"
    -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM="${VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM:-1}"
    -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS="${VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS:-128}"
    -e VLLM_V100_FP8_MOE_GROUPED_K_SPLIT="${VLLM_V100_FP8_MOE_GROUPED_K_SPLIT:-auto}"
    -e VLLM_V100_FP8_MOE_GROUPED_LOG_ONCE="${VLLM_V100_FP8_MOE_GROUPED_LOG_ONCE:-1}"
    -e VLLM_V100_FP8_MOE_DEBUG="${VLLM_V100_FP8_MOE_DEBUG:-0}"
    -e VLLM_V100_FP8_MOE_PROFILE="${VLLM_V100_FP8_MOE_PROFILE:-0}"
    -e VLLM_V100_FP8_MOE_PROFILE_EVERY="${VLLM_V100_FP8_MOE_PROFILE_EVERY:-64}"
    -e VLLM_V100_FP8_MOE_PROFILE_WARMUP_CALLS="${VLLM_V100_FP8_MOE_PROFILE_WARMUP_CALLS:-200}"
    -e VLLM_V100_FP8_MOE_DECODE_M_MAX="${VLLM_V100_FP8_MOE_DECODE_M_MAX:-8}"
    -e VLLM_V100_FP8_MOE_PROFILE_ACTIVE_STAT="${VLLM_V100_FP8_MOE_PROFILE_ACTIVE_STAT:-0}"
    -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP="${VLLM_V100_FP8_MOE_FAST_ROUTE_PREP:-1}"
    -e VLLM_V100_FP8_DECODE_BREAKDOWN="${VLLM_V100_FP8_DECODE_BREAKDOWN:-0}"
    -e VLLM_V100_FP8_DECODE_BREAKDOWN_EVERY="${VLLM_V100_FP8_DECODE_BREAKDOWN_EVERY:-32}"
    -e VLLM_V100_FP8_DECODE_BREAKDOWN_MOE_SUBS="${VLLM_V100_FP8_DECODE_BREAKDOWN_MOE_SUBS:-1}"
    -e VLLM_V100_FP8_DEBUG_SHARED_EXPERTS="${VLLM_V100_FP8_DEBUG_SHARED_EXPERTS:-0}"
    -e VLLM_V100_FP8_MOE_OTHER_PROFILE="${VLLM_V100_FP8_MOE_OTHER_PROFILE:-0}"
    -e VLLM_V100_FP8_DECODE_BREAKDOWN_GDN_SUBS="${VLLM_V100_FP8_DECODE_BREAKDOWN_GDN_SUBS:-1}"
    -e VLLM_V100_FP8_ROW_PARALLEL_AR_PROFILE="${VLLM_V100_FP8_ROW_PARALLEL_AR_PROFILE:-0}"
)

# v0.4.1: optional Qwen3.5 MTP speculative decoding. Opt-in via
# ENABLE_QWEN_MTP=1. Default OFF. When enabled, appends
#   --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
# to the forwarded vllm serve args. See README.md "v0.4.1: Optional
# MTP speculative decoding" and docs/SESSION_LOG.md Stage 4 for
# workload-shape guidance, exactness caveats, and validation matrix.
SPECULATIVE_ARGS=()
if [[ "${ENABLE_QWEN_MTP:-0}" == "1" ]]; then
    SPECULATIVE_ARGS=(
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
    )
fi

case "${1:-help}" in
    build)
        docker build -t "$IMAGE" -f "$HERE/Dockerfile.vllm018_py312" "$PROJECT_ROOT"
        ;;
    shell)
        docker run --rm -it "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            -p ${PORT}:${PORT} \
            --shm-size=8g \
            "${RUNTIME_ENV[@]}" \
            "$IMAGE" \
            bash
        ;;
    serve)
        # Plain `vllm serve` — no monkey-patches. Useful for FP16/GPTQ
        # baselines and stock-vLLM comparisons. ENABLE_QWEN_MTP=1 adds
        # spec-decode config; see top-of-file note.
        shift || true
        docker run --rm -i "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            -p ${PORT}:${PORT} \
            --shm-size=8g \
            "${RUNTIME_ENV[@]}" \
            "$IMAGE" \
            vllm serve "${SPECULATIVE_ARGS[@]}" "$@"
        ;;
    serve-fp8)
        # FP8 W8A16 sm_70 serve: imports the fp8_w8a16_sm70 monkey-patches
        # from the mounted repo (PYTHONPATH=/work/src) and invokes them via
        # `python3 -m fp8_w8a16_sm70.vllm_serve`. The .cu kernel JIT-compiles
        # on first launch and lives in the torch_extensions cache mount.
        # Same arg-forwarding shape as `serve`; the patches handle FP8
        # capability gate bypass + W8A16 dequant + MoE optimizations.
        # ENABLE_QWEN_MTP=1 adds spec-decode config; see top-of-file note.
        shift || true
        docker run --rm -i "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            -p ${PORT}:${PORT} \
            --shm-size=8g \
            "${RUNTIME_ENV[@]}" \
            "${FP8_ENV[@]}" \
            "$IMAGE" \
            python3 -m fp8_w8a16_sm70.vllm_serve "${SPECULATIVE_ARGS[@]}" "$@"
        ;;
    py)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${RUNTIME_ENV[@]}" \
            "$IMAGE" \
            python3 "$@"
        ;;
    *)
        cat >&2 <<USAGE
usage: $0 <mode> [args...]

  build      build the baseline image ($IMAGE) — vllm 0.18 on py3.12
  shell      interactive bash inside the image
  serve      run \`vllm serve\` with forwarded args (no monkey-patches)
  serve-fp8  run \`python3 -m fp8_w8a16_sm70.vllm_serve\` with our monkey-patches
             (mounts repo at /work, PYTHONPATH=/work/src, all VLLM_V100_FP8_* env vars set)
  py         run an arbitrary python script in the image

env vars:
  GPUS="0"             restrict to device indices (default: all)
  PORT=8002            host port for serve/shell (default: 8002 — distinct from
                       the cu128 image's 8000 and 1catai's 8001 so all three
                       can run in parallel on different GPUs)
  ENABLE_QWEN_MTP=1    v0.4.1 opt-in: enable Qwen3.5 MTP speculative decoding.
                       Default OFF. Appends --speculative-config to forwarded args.
                       See README.md and docs/SESSION_LOG.md Stage 4.
USAGE
        exit 1
        ;;
esac
