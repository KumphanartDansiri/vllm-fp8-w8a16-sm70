#!/usr/bin/env bash
# Launcher for the vLLM 0.19.0 sm_70 SOURCE-BUILD image (see
# docker/Dockerfile.vllm019_py312 for why 0.19 must be built from source on
# V100). Keeps Python 3.12 + torch 2.10.0+cu128 so the fp8_w8a16_sm70
# monkey-patches stay ABI-compatible with the 0.18 image.
#
# First time:   ./docker/run_docker_vllm019_py312.sh build    # ~30-90 min
# Interactive:  ./docker/run_docker_vllm019_py312.sh shell
# Serve:        ./docker/run_docker_vllm019_py312.sh serve     <vllm serve args...>
# FP8 serve:    ./docker/run_docker_vllm019_py312.sh serve-fp8 <vllm serve args...>
#
# Coexists with the 0.18 stack: distinct IMAGE, distinct cache dirs
# (~/.cache/vllm019-py312-*), distinct default PORT (8003 vs 0.18's 8002).
#
# Mounts (every run):
#   <project root> -> /work  (rw; PYTHONPATH=/work/src for serve-fp8)
#   /mnt/models    -> /mnt/models (ro)
#   ~/.cache/vllm019-py312-* -> compile caches (separate from 0.18 — torch ABI
#                               is the same but the vllm version fingerprint
#                               differs, so cross-image reuse would risk
#                               stale-hit corruption).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-vllm-v100-py312:vllm019}"
# Where the vLLM 0.19 source checkout lives — this is the docker BUILD CONTEXT.
VLLM_SRC="${VLLM_SRC:-$HOME/vllm-0.19}"
# nvcc parallelism for the source build; lower it if a training job is running.
MAX_JOBS="${MAX_JOBS:-8}"

MODEL_MOUNT=()
if [[ -d /mnt/models ]]; then
    MODEL_MOUNT=(-v /mnt/models:/mnt/models:ro)
fi

for sub in triton torch torchinductor extensions; do
    mkdir -p "$HOME/.cache/vllm019-py312-$sub"
done
CACHE_MOUNTS=(
    -v "$HOME/.cache/vllm019-py312-triton:/root/.triton"
    -v "$HOME/.cache/vllm019-py312-torch:/root/.cache/torch"
    -v "$HOME/.cache/vllm019-py312-extensions:/root/.cache/torch_extensions"
    -v "$HOME/.cache/vllm019-py312-torchinductor:/tmp/torchinductor_root"
)

VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

GPUS="${GPUS:-all}"
if [[ "$GPUS" == "all" ]]; then
    GPU_FLAG=(--gpus all)
else
    GPU_FLAG=(--gpus "\"device=$GPUS\"")
fi

PORT="${PORT:-8003}"

# Optional fixed container name so a driver script (tools/smoke_vllm019.sh) can
# `docker stop` exactly the server it started, instead of guessing by image.
NAME_FLAG=()
if [[ -n "${CONTAINER_NAME:-}" ]]; then
    NAME_FLAG=(--name "$CONTAINER_NAME")
fi

RUNTIME_ENV=(
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
    -e VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD"
)

# Extra env used only by serve-fp8 — mirrors the 0.18 launcher's FP8_ENV so the
# monkey-patches see identical knobs. All have safe defaults.
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

case "${1:-help}" in
    build)
        if [[ ! -d "$VLLM_SRC" ]]; then
            echo "[019] ERROR: vLLM source not found at VLLM_SRC=$VLLM_SRC" >&2
            echo "      Point VLLM_SRC at your vllm-0.19 checkout." >&2
            exit 1
        fi
        # Optional: override transformers (for Gemma 4, which needs 5.x — see
        # Dockerfile). Tag such builds distinctly via IMAGE= so they don't
        # clobber the FP8-validated default image.
        TF_ARG=()
        if [[ -n "${TRANSFORMERS_VERSION:-}" ]]; then
            TF_ARG=(--build-arg TRANSFORMERS_VERSION="$TRANSFORMERS_VERSION")
            echo "[019] transformers override: $TRANSFORMERS_VERSION (unsupported by vLLM pin)"
        fi
        echo "[019] building $IMAGE from source at $VLLM_SRC (MAX_JOBS=$MAX_JOBS) — this is slow."
        docker build -t "$IMAGE" \
            --build-arg MAX_JOBS="$MAX_JOBS" \
            "${TF_ARG[@]}" \
            -f "$HERE/Dockerfile.vllm019_py312" "$VLLM_SRC"
        ;;
    shell)
        docker run --rm -it "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            -p ${PORT}:${PORT} --shm-size=8g \
            "${RUNTIME_ENV[@]}" "$IMAGE" bash
        ;;
    serve)
        # Plain `vllm serve` — no monkey-patches. For Gemma 4 / FP16 / GPTQ
        # baselines and stock-vLLM comparisons on the 0.19 build.
        shift || true
        docker run --rm -i "${NAME_FLAG[@]}" "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            -p ${PORT}:${PORT} --shm-size=8g \
            "${RUNTIME_ENV[@]}" "$IMAGE" \
            vllm serve "$@"
        ;;
    serve-fp8)
        # FP8 W8A16 sm_70 serve: imports the fp8_w8a16_sm70 monkey-patches from
        # the mounted repo (PYTHONPATH=/work/src) via the same entry point as
        # the 0.18 stack. The .cu kernel JIT-compiles on first launch into the
        # torch_extensions cache mount.
        shift || true
        docker run --rm -i "${NAME_FLAG[@]}" "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            -p ${PORT}:${PORT} --shm-size=8g \
            "${RUNTIME_ENV[@]}" "${FP8_ENV[@]}" "$IMAGE" \
            python3 -m fp8_w8a16_sm70.vllm_serve "$@"
        ;;
    py)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${RUNTIME_ENV[@]}" "$IMAGE" python3 "$@"
        ;;
    *)
        cat >&2 <<USAGE
usage: $0 <mode> [args...]

  build      build the sm_70 source image ($IMAGE) — vllm 0.19 from $VLLM_SRC
  shell      interactive bash inside the image
  serve      run \`vllm serve\` with forwarded args (no monkey-patches)
  serve-fp8  run \`python3 -m fp8_w8a16_sm70.vllm_serve\` with our monkey-patches

env vars:
  GPUS="0,1,2,3"   restrict to device indices (default: all)
  PORT=8003        host port (default 8003 — distinct from 0.18's 8002)
  VLLM_SRC=PATH    vllm-0.19 source checkout / build context (default ~/vllm-0.19)
  MAX_JOBS=8       nvcc build parallelism (lower if training is running)
  IMAGE=...        override image tag
USAGE
        exit 1
        ;;
esac
