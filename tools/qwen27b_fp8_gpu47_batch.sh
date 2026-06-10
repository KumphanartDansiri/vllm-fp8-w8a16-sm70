#!/usr/bin/env bash
# Qwen 27B FP8 batch test lane for the shared V100 box.
#
# This script deliberately uses GPUs 4,5,6,7 and port 8027 by default so it can
# run beside another agent's Gemma-4 work on GPUs 0,1,2,3 / port 8021.
#
# Usage:
#   ./tools/qwen27b_fp8_gpu47_batch.sh
#   MODEL=/mnt/models/Qwen/Qwen3.6-27B-FP8 ./tools/qwen27b_fp8_gpu47_batch.sh
#
# Env:
#   MODEL          default: /mnt/models/Qwen/Qwen3.6-27B-FP8
#   GPUS           default: 4,5,6,7
#   TP_SIZE        default: number of GPUs in GPUS
#   PORT           default: 8027
#   TAG            default: qwen27b_fp8_gpu47
#   N_CURLS        default: 4
#   MAX_TOKENS     default: 200
#   MAX_MODEL_LEN  default: 4096
#   MAX_NUM_SEQS   default: 8
#   CT_RESIDENT    default: 1, enables compressed-tensors FP8-resident channel path

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPUS="${GPUS:-4,5,6,7}"
PORT="${PORT:-8027}"
TAG="${TAG:-qwen27b_fp8_gpu47}"
RUN_TAG="${RUN_TAG:-${TAG}_$(date +%Y%m%d_%H%M%S)}"
N_CURLS="${N_CURLS:-4}"
MAX_TOKENS="${MAX_TOKENS:-200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
CT_RESIDENT="${CT_RESIDENT:-1}"
V100_BENCH_ROOT="${V100_BENCH_ROOT:-/tmp/v100_qwen27b_fp8_gpu47}"

if [[ -z "${MODEL:-}" ]]; then
    MODEL="/mnt/models/Qwen/Qwen3.6-27B-FP8"
fi

gpu_count() {
    awk -F',' '{print NF}' <<<"$GPUS"
}

guard_gpus_idle() {
    local gpu used pids busy=0
    IFS=',' read -ra ids <<<"$GPUS"
    for gpu in "${ids[@]}"; do
        used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
        pids=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '/^[0-9]+$/ {n++} END{print n+0}')
        if [[ "${used:-9999}" -gt 2000 || "${pids:-1}" -gt 0 ]]; then
            echo "[qwen27b-gpu47] GPU $gpu busy: used=${used:-?}MiB compute_pids=${pids:-?}" >&2
            busy=1
        fi
    done
    [[ "$busy" -eq 0 ]]
}

guard_port_free() {
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[qwen27b-gpu47] port $PORT already has a healthy vLLM server" >&2
        return 1
    fi
}

if [[ ! -f "$MODEL/config.json" ]]; then
    echo "[qwen27b-gpu47] missing model config: $MODEL" >&2
    exit 1
fi

TP_SIZE="${TP_SIZE:-$(gpu_count)}"
mkdir -p "$V100_BENCH_ROOT"

guard_gpus_idle || {
    echo "[qwen27b-gpu47] refusing to collide. Current GPU memory:" >&2
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >&2
    exit 1
}
guard_port_free || exit 1

echo "[qwen27b-gpu47] model          : $MODEL"
echo "[qwen27b-gpu47] gpus/tp/port   : $GPUS / $TP_SIZE / $PORT"
echo "[qwen27b-gpu47] output root    : $V100_BENCH_ROOT"
echo "[qwen27b-gpu47] tag            : $RUN_TAG"
echo "[qwen27b-gpu47] ct resident    : $CT_RESIDENT"

serve_pid=""
cleanup() {
    if [[ -n "$serve_pid" ]] && kill -0 "$serve_pid" 2>/dev/null; then
        echo "[qwen27b-gpu47] stopping serve pid $serve_pid"
        kill "$serve_pid" 2>/dev/null || true
        wait "$serve_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

MODEL="$MODEL" \
GPUS="$GPUS" \
TP_SIZE="$TP_SIZE" \
PORT="$PORT" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
MAX_NUM_SEQS="$MAX_NUM_SEQS" \
MAX_TOKENS="$MAX_TOKENS" \
VLLM_V100_CT_FP8_RESIDENT="$CT_RESIDENT" \
V100_BENCH_ROOT="$V100_BENCH_ROOT" \
./tools/bench_v100.sh serve "$RUN_TAG" &
serve_pid=$!

echo "[qwen27b-gpu47] serve pid      : $serve_pid"

serve_log=""
for _ in $(seq 1 30); do
    serve_log=$(ls -1 "$V100_BENCH_ROOT"/*_"$RUN_TAG"/serve.log 2>/dev/null | head -1 || true)
    [[ -n "$serve_log" ]] && break
    if ! kill -0 "$serve_pid" 2>/dev/null; then
        echo "[qwen27b-gpu47] serve process exited before creating a run directory" >&2
        exit 1
    fi
    sleep 1
done
if [[ -z "$serve_log" ]]; then
    echo "[qwen27b-gpu47] timed out waiting for bench run directory for tag $RUN_TAG" >&2
    exit 1
fi

MODEL="$MODEL" \
PORT="$PORT" \
MAX_TOKENS="$MAX_TOKENS" \
V100_BENCH_ROOT="$V100_BENCH_ROOT" \
./tools/bench_v100.sh curls "$RUN_TAG" "$N_CURLS" \
    "Write a concise technical explanation of why FP8 dequantization can be slow on V100."

echo "[qwen27b-gpu47] done. Latest run:"
readlink -f "$V100_BENCH_ROOT/latest" 2>/dev/null || true
