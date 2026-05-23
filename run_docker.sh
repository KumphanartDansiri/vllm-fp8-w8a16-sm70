#!/usr/bin/env bash
# Build the image once, then run the FP8->FP16 hello-world test on V100.
#
# First time: ./run_docker.sh build
# Then:       ./run_docker.sh test
#
# Mounts:
#   ./             -> /work          (this experiment dir, rw)
#   /mnt/models    -> /mnt/models    (read-only; for serving real FP8 models later)
#
# Ports:
#   8000 -> 8000   (vLLM OpenAI-compatible API default; only bound in `serve` mode)
#
# If your user isn't in the `docker` group, prefix with `sudo` or run:
#   sudo usermod -aG docker $USER && newgrp docker
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="vllm-v100-fp8-test:cu124"
DEV_IMAGE="vllm-v100-dev:cu128"

# Mount /mnt/models only if it exists, so `test` keeps working on hosts without it.
MODEL_MOUNT=()
if [[ -d /mnt/models ]]; then
    MODEL_MOUNT=(-v /mnt/models:/mnt/models:ro)
fi

# Persistent cache mounts for the dev (cu128) image. Without these, every
# `serve` restart spends ~14 minutes re-JIT-compiling Triton/FLA kernels and
# our fp8_dequant extension. With these, the first run still pays the full
# cost but subsequent runs reuse the cached binaries.
#   ~/.cache/vllm-triton        -> /root/.triton             (Triton compile cache for direct triton calls)
#   ~/.cache/vllm-torch         -> /root/.cache/torch        (torch dynamo / inductor)
#   ~/.cache/vllm-extensions    -> /root/.cache/torch_extensions (our kernel JIT)
#   ~/.cache/vllm-torchinductor -> /tmp/torchinductor_root   (CRITICAL — this is where the FLA Mamba +
#                                                              inductor-fused Triton kernels actually live;
#                                                              /tmp is ephemeral in docker, so without this
#                                                              mount every restart pays ~14 min Triton compile)
CACHE_MOUNTS=()
for sub in triton torch extensions torchinductor; do
    host_dir="$HOME/.cache/vllm-$sub"
    mkdir -p "$host_dir"
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

# GPU isolation. Default "all" matches prior behavior. Set GPUS="0,1,2,3" to
# restrict the container to those device indices — useful when something else
# (e.g. aiagent's GPTQ-Int4 baseline) is running on the other GPUs.
GPUS="${GPUS:-all}"
if [[ "$GPUS" == "all" ]]; then
    GPU_FLAG=(--gpus all)
else
    # Docker's --gpus value parser comma-splits the value into key=value
    # pairs, so `device=4,5,6,7` becomes [device=4, 5, 6, 7] and the bare
    # numbers parse as Count, conflicting with device=4 as DeviceIDs
    # ("cannot set both Count and DeviceIDs"). The canonical fix is to
    # wrap the value in literal double-quotes so docker treats the whole
    # device=... string as one value before its own comma-splitting.
    GPU_FLAG=(--gpus "\"device=$GPUS\"")
fi

# Host port that the serve mode publishes. Default 8000. Set PORT=8001 to run
# alongside another vLLM server on 8000.
PORT="${PORT:-8000}"

case "${1:-test}" in
    build)
        docker build -t "$IMAGE" "$HERE"
        ;;
    build-dev)
        docker build -t "$DEV_IMAGE" -f "$HERE/Dockerfile.dev" "$HERE"
        ;;
    test)
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$HERE":/work \
            -w /work \
            "$IMAGE" \
            python3 test_fp8.py
        ;;
    inspect)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$HERE":/work \
            -w /work \
            "$IMAGE" \
            python3 inspect_fp8_model.py "$@"
        ;;
    matmul)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$HERE":/work \
            -w /work \
            "$IMAGE" \
            python3 test_phase5_matmul.py "$@"
        ;;
    phaseb)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$HERE":/work \
            -w /work \
            "$IMAGE" \
            python3 test_phase_b_module.py "$@"
        ;;
    a1)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$HERE":/work \
            -w /work \
            "$IMAGE" \
            python3 test_phase_a1.py "$@"
        ;;
    a2)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$HERE":/work \
            -w /work \
            "$IMAGE" \
            python3 test_phase_a2.py "$@"
        ;;
    a3)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$HERE":/work \
            -w /work \
            "$IMAGE" \
            python3 test_phase_a3.py "$@"
        ;;
    mlp)
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$HERE":/work \
            -w /work \
            "$IMAGE" \
            python3 test_phase_c_mlp.py "$@"
        ;;
    shell)
        docker run --rm -it "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            -v "$HERE":/work \
            -w /work \
            "$IMAGE" \
            bash
        ;;
    dev-shell)
        # Interactive shell inside the dev image (vllm 0.18 + torch 2.10 + cu128).
        # Mirrors aiagent's verified production stack — see AIAGENT_ENV.md.
        # Use this for vLLM integration work.
        docker run --rm -it "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$HERE":/work \
            -w /work \
            -p ${PORT}:8000 \
            --shm-size=8g \
            -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS" \
            -e VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" \
            -e VLLM_V100_FP8_DEBUG_SHAPES="${VLLM_V100_FP8_DEBUG_SHAPES:-}" \
            -e VLLM_V100_FP8_DEBUG_APPLY="${VLLM_V100_FP8_DEBUG_APPLY:-}" \
            -e VLLM_V100_FP8_APPLY_MAG="${VLLM_V100_FP8_APPLY_MAG:-10000}" \
            -e VLLM_V100_FP8_APPLY_WARN="${VLLM_V100_FP8_APPLY_WARN:-1000}" \
            -e VLLM_V100_FP8_HASH_LAYERS="${VLLM_V100_FP8_HASH_LAYERS:-off}" \
            "$DEV_IMAGE" \
            bash
        ;;
    dev-test)
        # Run a script in the dev image (defaults to test_fp8.py if none given).
        # Useful for re-validating the kernel JIT-compiles against torch 2.10 +
        # cu128 before doing vllm integration work.
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$HERE":/work \
            -w /work \
            -e MODEL_DIR="${MODEL_DIR:-/mnt/models/Qwen3.6-27B-FP8}" \
            -e TP_SIZE="${TP_SIZE:-4}" \
            -e BENCH_M="${BENCH_M:-4096}" \
            -e BENCH_SWEEP="${BENCH_SWEEP:-0}" \
            "$DEV_IMAGE" \
            python3 "${@:-test_fp8.py}"
        ;;
    attn-test)
        # Offline kernel-vs-reference test for the exact attention/linear_attn
        # shapes the 27B-FP8 model exercises at TP=4. See
        # test_attention_shapes.py for the test plan.
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$HERE":/work \
            -w /work \
            -e MODEL_DIR="${MODEL_DIR:-/mnt/models/Qwen3.6-27B-FP8}" \
            -e TP_SIZE="${TP_SIZE:-4}" \
            -e TOL_ABS="${TOL_ABS:-5e-2}" \
            "$DEV_IMAGE" \
            python3 test_attention_shapes.py "$@"
        ;;
    dump-hashes)
        # Dump offline-emulator hashes (no GPU/kernel) for the 5 target layers
        # × 4 ranks. Diff against the runtime hashes that serve_fp8_v100.py
        # prints (VLLM_V100_FP8_HASH_LAYERS=on, default). See
        # dump_offline_hashes.py for the layer set + sharding rules.
        shift || true
        docker run --rm "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$HERE":/work \
            -w /work \
            -e MODEL_DIR="${MODEL_DIR:-/mnt/models/Qwen3.6-27B-FP8}" \
            -e TP_SIZE="${TP_SIZE:-4}" \
            "$DEV_IMAGE" \
            python3 dump_offline_hashes.py "$@"
        ;;
    serve)
        # Run a vllm command inside the dev image (port 8000 forwarded for the
        # OpenAI-compatible API).
        # CACHE_MOUNTS persist Triton + torch JIT caches across restarts —
        # second `serve` is ~30s startup instead of the first one's ~14 min.
        # Examples:
        #   ./run_docker.sh serve serve_fp8_v100.py --model /mnt/models/Qwen3.5-4B-FP8 ...
        #   ./run_docker.sh serve -m vllm.entrypoints.openai.api_server --model ...
        shift || true
        # NOTE: -i (no -t) so this works under background/tmux/CI where there
        # is no TTY. Stdin is not needed for the serve once started.
        docker run --rm -i "${GPU_FLAG[@]}" \
            "${MODEL_MOUNT[@]}" \
            "${CACHE_MOUNTS[@]}" \
            -v "$HERE":/work \
            -w /work \
            -p ${PORT}:8000 \
            --shm-size=8g \
            -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS" \
            -e VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" \
            -e VLLM_V100_FP8_DEBUG_SHAPES="${VLLM_V100_FP8_DEBUG_SHAPES:-}" \
            -e VLLM_V100_FP8_DEBUG_APPLY="${VLLM_V100_FP8_DEBUG_APPLY:-}" \
            -e VLLM_V100_FP8_APPLY_MAG="${VLLM_V100_FP8_APPLY_MAG:-10000}" \
            -e VLLM_V100_FP8_APPLY_WARN="${VLLM_V100_FP8_APPLY_WARN:-1000}" \
            -e VLLM_V100_FP8_HASH_LAYERS="${VLLM_V100_FP8_HASH_LAYERS:-off}" \
            -e FP8_WMMA_MIN_M="${FP8_WMMA_MIN_M:-64}" \
            -e FP8_WMMA_COUNTER_LOG_EVERY="${FP8_WMMA_COUNTER_LOG_EVERY:-1000}" \
            "$DEV_IMAGE" \
            python3 "$@"
        ;;
    *)
        echo "usage: $0 <mode> [args...]"                                              >&2
        echo ""                                                                        >&2
        echo "  cu124 image (vllm-v100-fp8-test:cu124) — kernel correctness tests:"    >&2
        echo "    build      build the cu124 image"                                    >&2
        echo "    test       run test_fp8.py (Phase 1+2 dequant)"                      >&2
        echo "    inspect    run inspect_fp8_model.py"                                 >&2
        echo "    matmul     run test_phase5_matmul.py"                                >&2
        echo "    phaseb     run test_phase_b_module.py"                               >&2
        echo "    a1|a2|a3   run test_phase_a{1,2,3}.py"                               >&2
        echo "    mlp        run test_phase_c_mlp.py"                                  >&2
        echo "    shell      interactive bash"                                         >&2
        echo ""                                                                        >&2
        echo "  cu128 image (vllm-v100-dev:cu128) — vllm integration, AIAGENT_ENV.md:" >&2
        echo "    build-dev  build the cu128 dev image"                                >&2
        echo "    dev-shell  interactive bash inside dev image"                        >&2
        echo "    dev-test   run a script in dev image (default: test_fp8.py)"        >&2
        echo "    serve      run a python script with port \$PORT forwarded"          >&2
        echo "    attn-test  run test_attention_shapes.py (offline kernel-vs-ref test)" >&2
        echo "    dump-hashes run dump_offline_hashes.py (no GPU, fast hash dump for diff)" >&2
        echo ""                                                                        >&2
        echo "  env vars:"                                                             >&2
        echo "    GPUS=\"0,1,2,3\"  restrict to specific device indices (default: all)" >&2
        echo "    PORT=8001         host port for serve/dev-shell      (default: 8000)" >&2
        exit 1
        ;;
esac
