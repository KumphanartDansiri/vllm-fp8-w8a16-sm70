#!/usr/bin/env bash
# Profile one representative Qwen3.6 FP8 decode GEMV kernel with Nsight Compute.
#
# Defaults to GPU 4 to avoid GPUs 0-3 when another agent is active.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROJECT_ROOT="$(pwd)"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
GPU="${GPU:-4}"
CACHE_TAG="${CACHE_TAG:-021}"
SHAPE="${SHAPE:-attn}"
OP="${OP:-a3}"
K_SPLIT="${K_SPLIT:-8}"
ITERS="${ITERS:-1}"
WARMUP="${WARMUP:-10}"
OUT_DIR="${OUT_DIR:-/tmp/v100_qwen27b_fp8_ncu}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_${OP}_${SHAPE}_ks${K_SPLIT}}"

METRICS="${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_miss.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct}"

mkdir -p "$OUT_DIR"
for s in torchext triton torch inductor; do
    mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"
done

docker image inspect "$IMAGE" >/dev/null

used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
if [[ "${used:-9999}" -gt 2000 ]]; then
    echo "[qwen27b-a3-ncu] GPU $GPU busy (${used} MiB used); choose GPU=..." >&2
    exit 1
fi

echo "[qwen27b-a3-ncu] gpu/image : $GPU / $IMAGE"
echo "[qwen27b-a3-ncu] op/shape  : $OP / $SHAPE k_split=$K_SPLIT"
echo "[qwen27b-a3-ncu] out       : $OUT_DIR/$RUN_TAG.txt"

docker run --rm -i --name qwen27b_fp8_a3_ncu_probe --gpus "\"device=$GPU\"" \
    --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" ncu \
        --profile-from-start off \
        --target-processes all \
        --kernel-name-base function \
        --kernel-name 'regex:fp8_w8a16_gemv_coalesced_kernel|fp8_w8a16_gemm_a3_kernel|fp8_w8a16_gemm_a1_kernel|void cutlass::Kernel|ampere|volta|gemv|gemm|cublas' \
        --launch-count "$ITERS" \
        --metrics "$METRICS" \
        --csv \
        python3 tools/qwen27b_fp8_a3_ncu_probe.py \
            --op "$OP" --shape "$SHAPE" --k-split "$K_SPLIT" \
            --iters "$ITERS" --warmup "$WARMUP" \
    2>&1 | tee "$OUT_DIR/$RUN_TAG.txt"

echo "[qwen27b-a3-ncu] wrote $OUT_DIR/$RUN_TAG.txt"
