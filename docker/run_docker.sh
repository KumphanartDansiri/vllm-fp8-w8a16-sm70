#!/usr/bin/env bash
# Build the cu124 / cu128 docker images for V100 FP8 W8A16 work, and run
# tests / benches / the vLLM serve in them.
#
# First time:   ./docker/run_docker.sh build-dev
# Serve FP8:    ./docker/run_docker.sh serve --model /mnt/models/Qwen3.6-27B-FP8 ...
# Run a test:   ./docker/run_docker.sh dev-test tests/test_fp8.py
# Interactive:  ./docker/run_docker.sh dev-shell
#
# Note: this is the legacy py3.10/eager fallback launcher. The current v0.4.0
# performance baseline is docker/run_docker_vllm018_py312.sh.
#
# Mounts (every run):
#   <project root> -> /work                  (mounted rw)
#   /mnt/models    -> /mnt/models            (read-only, if it exists)
#   ~/.cache/vllm-* -> Triton/torch/extension caches inside the container
#                       (preserves the ~14-min cold autotune on `serve` restarts)
#
# Inside the container PYTHONPATH=/work/src so `import fp8_w8a16_sm70` works
# without an editable pip install.
#
# Ports:
#   $PORT (default 8000) -> 8000 inside the container, used by `serve`/`dev-shell`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"           # = .../docker
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"                          # repo root
IMAGE="vllm-v100-fp8-test:cu124"
DEV_IMAGE="vllm-v100-dev:cu128"

# Mount /mnt/models only if it exists, so basic tests work on hosts without it.
MODEL_MOUNT=()
if [[ -d /mnt/models ]]; then
    MODEL_MOUNT=(-v /mnt/models:/mnt/models:ro)
fi

# Persistent cache mounts for the dev (cu128) image. Without these, every
# `serve` restart spends ~14 minutes re-JIT-compiling Triton/FLA kernels and
# our fp8_dequant extension. With these, the first run still pays the full
# cost but subsequent runs reuse the cached binaries.
#   ~/.cache/vllm-triton        -> /root/.triton             (direct Triton cache)
#   ~/.cache/vllm-torch         -> /root/.cache/torch        (torch dynamo / inductor)
#   ~/.cache/vllm-extensions    -> /root/.cache/torch_extensions (our kernel JIT)
#   ~/.cache/vllm-torchinductor -> /tmp/torchinductor_root   (CRITICAL — this is where
#                                   FLA Mamba + inductor-fused Triton kernels actually
#                                   live; /tmp is ephemeral in docker, so without this
#                                   mount every restart pays ~14 min Triton compile)
for sub in triton torch extensions torchinductor; do
    mkdir -p "$HOME/.cache/vllm-$sub"
done
CACHE_MOUNTS=(
    -v "$HOME/.cache/vllm-triton:/root/.triton"
    -v "$HOME/.cache/vllm-torch:/root/.cache/torch"
    -v "$HOME/.cache/vllm-extensions:/root/.cache/torch_extensions"
    -v "$HOME/.cache/vllm-torchinductor:/tmp/torchinductor_root"
)

# vLLM TP workers communicate through shared-memory queues. Keep a generous
# default for slow V100 cold paths, while still allowing callers to override it.
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# GPU isolation. Default "all". Set GPUS="0,1,2,3" to restrict device indices.
GPUS="${GPUS:-all}"
if [[ "$GPUS" == "all" ]]; then
    GPU_FLAG=(--gpus all)
else
    # Wrap in literal double-quotes so docker treats `device=4,5,6,7` as one
    # value instead of comma-splitting into a Count + DeviceIDs conflict.
    GPU_FLAG=(--gpus "\"device=$GPUS\"")
fi

# Host port that the serve mode publishes. Default 8000.
PORT="${PORT:-8000}"

# Common -e flags for the dev image (vllm serve / dev-shell / bench scripts).
# PYTHONPATH=/work/src makes `import fp8_w8a16_sm70` work everywhere in container.
DEV_ENV=(
    -e PYTHONPATH=/work/src
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
    -e VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD"
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
    -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS="${VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS:-32}"
    -e VLLM_V100_FP8_MOE_GROUPED_K_SPLIT="${VLLM_V100_FP8_MOE_GROUPED_K_SPLIT:-auto}"
    -e VLLM_V100_FP8_MOE_GROUPED_LOG_ONCE="${VLLM_V100_FP8_MOE_GROUPED_LOG_ONCE:-1}"
    -e VLLM_V100_FP8_MOE_DEBUG="${VLLM_V100_FP8_MOE_DEBUG:-0}"
    -e VLLM_V100_FP8_MOE_PROFILE="${VLLM_V100_FP8_MOE_PROFILE:-0}"
    -e VLLM_V100_FP8_MOE_PROFILE_EVERY="${VLLM_V100_FP8_MOE_PROFILE_EVERY:-64}"
    -e VLLM_V100_FP8_MOE_PROFILE_WARMUP_CALLS="${VLLM_V100_FP8_MOE_PROFILE_WARMUP_CALLS:-200}"
    -e VLLM_V100_FP8_MOE_DECODE_M_MAX="${VLLM_V100_FP8_MOE_DECODE_M_MAX:-8}"
    -e VLLM_V100_FP8_MOE_PROFILE_ACTIVE_STAT="${VLLM_V100_FP8_MOE_PROFILE_ACTIVE_STAT:-0}"
    -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP="${VLLM_V100_FP8_MOE_FAST_ROUTE_PREP:-0}"
    -e VLLM_V100_FP8_DECODE_BREAKDOWN="${VLLM_V100_FP8_DECODE_BREAKDOWN:-0}"
    -e VLLM_V100_FP8_DECODE_BREAKDOWN_EVERY="${VLLM_V100_FP8_DECODE_BREAKDOWN_EVERY:-32}"
    -e VLLM_V100_FP8_DECODE_BREAKDOWN_MOE_SUBS="${VLLM_V100_FP8_DECODE_BREAKDOWN_MOE_SUBS:-1}"
    -e VLLM_V100_FP8_DEBUG_SHARED_EXPERTS="${VLLM_V100_FP8_DEBUG_SHARED_EXPERTS:-0}"
    -e VLLM_V100_FP8_MOE_OTHER_PROFILE="${VLLM_V100_FP8_MOE_OTHER_PROFILE:-0}"
    -e VLLM_V100_FP8_DECODE_BREAKDOWN_GDN_SUBS="${VLLM_V100_FP8_DECODE_BREAKDOWN_GDN_SUBS:-1}"
    -e VLLM_V100_FP8_ROW_PARALLEL_AR_PROFILE="${VLLM_V100_FP8_ROW_PARALLEL_AR_PROFILE:-0}"
    -e VLLM_V100_FP8_COALESCED_GEMV="${VLLM_V100_FP8_COALESCED_GEMV:-0}"
    -e VLLM_V100_FP8_COALESCED_UNROLL="${VLLM_V100_FP8_COALESCED_UNROLL:-2}"
    -e VLLM_V100_FP8_COALESCED_M_UNROLL="${VLLM_V100_FP8_COALESCED_M_UNROLL:-${VLLM_V100_FP8_COALESCED_UNROLL:-2}}"
    -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX="${VLLM_V100_FP8_COALESCED_GEMV_M_MAX:-1}"
    -e MODEL_DIR="${MODEL_DIR:-/mnt/models/Qwen3.6-27B-FP8}"
    -e TP_SIZE="${TP_SIZE:-4}"
    -e BENCH_M="${BENCH_M:-4096}"
    -e BENCH_SWEEP="${BENCH_SWEEP:-0}"
    -e TOL_ABS="${TOL_ABS:-5e-2}"
)

# Common -e flags for the cu124 correctness-test image (smaller env).
CU124_ENV=(
    -e PYTHONPATH=/work/src
)

case "${1:-test}" in
    build)
        docker build -t "$IMAGE" -f "$HERE/Dockerfile" "$PROJECT_ROOT"
        ;;
    build-dev)
        docker build -t "$DEV_IMAGE" -f "$HERE/Dockerfile.dev" "$PROJECT_ROOT"
        ;;
    test)
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${CU124_ENV[@]}" \
            "$IMAGE" \
            python3 tests/test_fp8.py
        ;;
    inspect)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${CU124_ENV[@]}" \
            "$IMAGE" \
            python3 tools/inspect_fp8_model.py "$@"
        ;;
    matmul)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${CU124_ENV[@]}" \
            "$IMAGE" \
            python3 tests/test_phase5_matmul.py "$@"
        ;;
    phaseb)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${CU124_ENV[@]}" \
            "$IMAGE" \
            python3 tests/test_phase_b_module.py "$@"
        ;;
    a1)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${CU124_ENV[@]}" \
            "$IMAGE" \
            python3 tests/test_phase_a1.py "$@"
        ;;
    a2)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${CU124_ENV[@]}" \
            "$IMAGE" \
            python3 tests/test_phase_a2.py "$@"
        ;;
    a3)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${CU124_ENV[@]}" \
            "$IMAGE" \
            python3 tests/test_phase_a3.py "$@"
        ;;
    mlp)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${CU124_ENV[@]}" \
            "$IMAGE" \
            python3 tests/test_phase_c_mlp.py "$@"
        ;;
    shell)
        docker run --rm -it "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${CU124_ENV[@]}" \
            "$IMAGE" \
            bash
        ;;
    dev-shell)
        # Interactive shell inside the dev image (vllm 0.18 + torch 2.10 + cu128).
        # Mirrors aiagent's verified production stack — see docs/AIAGENT_ENV.md.
        docker run --rm -it "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            -p ${PORT}:${PORT} \
            --shm-size=8g \
            "${DEV_ENV[@]}" \
            "$DEV_IMAGE" \
            bash
        ;;
    dev-test)
        # Run an arbitrary script in the dev image. Default: tests/test_fp8.py
        #   ./docker/run_docker.sh dev-test benches/bench_profile_scale.py
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${DEV_ENV[@]}" \
            "$DEV_IMAGE" \
            python3 "${@:-tests/test_fp8.py}"
        ;;
    attn-test)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${DEV_ENV[@]}" \
            "$DEV_IMAGE" \
            python3 tests/test_attention_shapes.py "$@"
        ;;
    dump-hashes)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            "${DEV_ENV[@]}" \
            "$DEV_IMAGE" \
            python3 tools/dump_offline_hashes.py "$@"
        ;;
    serve)
        # Run vLLM with the FP8 W8A16 sm_70 monkey-patches applied.
        # All args after `serve` are forwarded to vLLM's api_server.
        # Example:
        #   ./docker/run_docker.sh serve \
        #       --model /mnt/models/Qwen3.6-27B-FP8 \
        #       --quantization fp8 --dtype float16 --enforce-eager \
        #       --attention-backend TRITON_ATTN --tensor-parallel-size 4 \
        #       --max-num-seqs 1 --gpu-memory-utilization 0.80 \
        #       --max-model-len 4096 --no-enable-chunked-prefill \
        #       --disable-custom-all-reduce --host 0.0.0.0 --port 8000
        shift || true
        # -i (no -t) so this works under background/tmux/CI where there is no
        # TTY. Stdin is not needed for the serve once started.
        docker run --rm -i "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$PROJECT_ROOT":/work -w /work \
            -p ${PORT}:${PORT} \
            --shm-size=8g \
            "${DEV_ENV[@]}" \
            "$DEV_IMAGE" \
            python3 -m fp8_w8a16_sm70.vllm_serve "$@"
        ;;
    *)
        echo "usage: $0 <mode> [args...]"                                              >&2
        echo ""                                                                        >&2
        echo "  cu124 image ($IMAGE) — kernel correctness tests:"                      >&2
        echo "    build      build the cu124 image"                                    >&2
        echo "    test       run tests/test_fp8.py"                                    >&2
        echo "    inspect    run tools/inspect_fp8_model.py"                           >&2
        echo "    matmul     run tests/test_phase5_matmul.py"                          >&2
        echo "    phaseb     run tests/test_phase_b_module.py"                         >&2
        echo "    a1|a2|a3   run tests/test_phase_a{1,2,3}.py"                         >&2
        echo "    mlp        run tests/test_phase_c_mlp.py"                            >&2
        echo "    shell      interactive bash inside cu124 image"                      >&2
        echo ""                                                                        >&2
        echo "  cu128 image ($DEV_IMAGE) — vllm integration:"                          >&2
        echo "    build-dev  build the cu128 dev image"                                >&2
        echo "    dev-shell  interactive bash inside dev image"                        >&2
        echo "    dev-test   run any python script in dev image (default test_fp8.py)" >&2
        echo "    serve      run vllm_serve (monkey-patches + vllm api_server)"        >&2
        echo "    attn-test  run tests/test_attention_shapes.py"                       >&2
        echo "    dump-hashes run tools/dump_offline_hashes.py"                        >&2
        echo ""                                                                        >&2
        echo "  env vars:"                                                             >&2
        echo "    GPUS=\"0,1,2,3\"  restrict to device indices (default: all)"        >&2
        echo "    PORT=8001         host port for serve/dev-shell (default: 8000)"     >&2
        echo "    FP8_WMMA_MIN_M    dispatch threshold for WMMA (default: 64; 99999=off)" >&2
        exit 1
        ;;
esac
