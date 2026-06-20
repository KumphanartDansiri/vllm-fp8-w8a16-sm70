#!/usr/bin/env bash
# Profile GLM-Air grouped MoE w13 decode kernels with Nsight Compute.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROJECT_ROOT="$(pwd)"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
GPU="${GPU:-4}"
CACHE_TAG="${CACHE_TAG:-021}"
OP="${OP:-coalesced}"
R="${R:-64}"
E="${E:-128}"
N="${N:-352}"
K="${K:-4096}"
BH="${BH:-1}"
EIDS_MODE="${EIDS_MODE:-random}"
HOT_EXPERTS="${HOT_EXPERTS:-8}"
WARMUP="${WARMUP:-20}"
OUT_DIR="${OUT_DIR:-/tmp/v100_grouped_moe_ncu}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_${OP}_R${R}_N${N}_K${K}_bh${BH}}"

METRICS="${METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_miss.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct}"

mkdir -p "$OUT_DIR"
for s in torchext triton torch inductor; do
    mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"
done

docker image inspect "$IMAGE" >/dev/null

used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
if [[ "${used:-9999}" -gt 2000 ]]; then
    echo "[grouped-moe-ncu] GPU $GPU busy (${used} MiB used); choose GPU=..." >&2
    exit 1
fi

echo "[grouped-moe-ncu] gpu/image : $GPU / $IMAGE"
echo "[grouped-moe-ncu] op/shape  : $OP R=$R E=$E N=$N K=$K block_h=$BH eids=$EIDS_MODE hot=$HOT_EXPERTS"
echo "[grouped-moe-ncu] out       : $OUT_DIR/$RUN_TAG.txt"

docker run --rm -i --name grouped_moe_ncu_probe --gpus "\"device=$GPU\"" \
    --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e VLLM_V100_FP8_COALESCED_UNROLL="${VLLM_V100_FP8_COALESCED_UNROLL:-4}" \
    "$IMAGE" ncu \
        --profile-from-start off \
        --target-processes all \
        --kernel-name-base function \
        --kernel-name 'regex:fp8_w8a16_grouped_gemv_coalesced_mtile_kernel|fp8_w8a16_grouped_tiled_gemm_kernel|fp8_w8a16_grouped_gemv_coalesced_kernel|fp8_w8a16_grouped_routed_gemm_a3_kernel' \
        --launch-count 1 \
        --metrics "$METRICS" \
        --csv \
        python3 tools/grouped_moe_ncu_probe.py \
            --op "$OP" --r "$R" --e "$E" --n "$N" --k "$K" \
            --block-h "$BH" --eids-mode "$EIDS_MODE" \
            --hot-experts "$HOT_EXPERTS" --warmup "$WARMUP" \
    2>&1 | tee "$OUT_DIR/$RUN_TAG.txt"

echo "[grouped-moe-ncu] wrote $OUT_DIR/$RUN_TAG.txt"
