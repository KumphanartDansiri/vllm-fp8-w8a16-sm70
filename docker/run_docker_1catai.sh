#!/usr/bin/env bash
# Spin up the 1Cat-vLLM v1.0.0 wheel-runtime image (Python 3.12 + torch 2.9.1)
# to compare against our cu128 monkey-patched stack on Qwen3.5/3.6-class MoE.
#
# First time:   ./docker/run_docker_1catai.sh build
# Interactive:  ./docker/run_docker_1catai.sh shell
# Serve:        ./docker/run_docker_1catai.sh serve --model /mnt/models/Qwen3.5-122B-A10B-FP8 \
#                   --tensor-parallel-size 8 --host 0.0.0.0 --port 8000
#
# Mounts (every run):
#   <project root> -> /work                  (mounted rw)
#   /mnt/models    -> /mnt/models            (read-only)
#   ~/.cache/1catai-vllm-* -> compile caches inside the container
#                              (separate from our cu128 image so torch 2.9 vs 2.10
#                              fingerprints don't collide)
#
# Ports:
#   $PORT (default 8000) -> 8000 inside the container.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
IMAGE="1catai-vllm-v100:cu128"

MODEL_MOUNT=()
if [[ -d /mnt/models ]]; then
    MODEL_MOUNT=(-v /mnt/models:/mnt/models:ro)
fi

# Persistent cache mounts for the 1catai image. KEPT SEPARATE from our cu128
# image's caches: torch 2.9.1 + py3.12 vs torch 2.10 + py3.10 means kernel /
# inductor cache keys differ; cross-version reuse would risk stale-hit
# corruption. Names parallel run_docker.sh for easy mental mapping.
for sub in triton torch torchinductor; do
    mkdir -p "$HOME/.cache/1catai-vllm-$sub"
done
CACHE_MOUNTS=(
    -v "$HOME/.cache/1catai-vllm-triton:/root/.triton"
    -v "$HOME/.cache/1catai-vllm-torch:/root/.cache/torch"
    -v "$HOME/.cache/1catai-vllm-torchinductor:/tmp/torchinductor_root"
)

VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

GPUS="${GPUS:-all}"
if [[ "$GPUS" == "all" ]]; then
    GPU_FLAG=(--gpus all)
else
    GPU_FLAG=(--gpus "\"device=$GPUS\"")
fi

PORT="${PORT:-8000}"

RUNTIME_ENV=(
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
    -e VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD"
)

case "${1:-help}" in
    build)
        docker build -t "$IMAGE" -f "$HERE/Dockerfile.1catai" "$PROJECT_ROOT"
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
        # Plain `vllm serve` — no monkey-patches. All args after `serve` are
        # forwarded as-is. Stays in foreground so the caller (bench_1catai.sh)
        # can `tee` stdout/stderr into a log file.
        shift || true
        docker run --rm -i "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            -p ${PORT}:${PORT} \
            --shm-size=8g \
            "${RUNTIME_ENV[@]}" \
            "$IMAGE" \
            vllm serve "$@"
        ;;
    py)
        # Run an arbitrary python script in the image (debug/diagnostic).
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

  build      build the 1catai-vllm wheel-runtime image ($IMAGE)
  shell      interactive bash inside the image
  serve      run \`vllm serve\` with forwarded args (no monkey-patches)
  py         run an arbitrary python script in the image

env vars:
  GPUS="0,1,2,3"  restrict to device indices (default: all)
  PORT=8001       host port for serve/shell (default: 8000)
USAGE
        exit 1
        ;;
esac
