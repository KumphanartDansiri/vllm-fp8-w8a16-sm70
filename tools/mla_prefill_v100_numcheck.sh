#!/usr/bin/env bash
# ======================================================================================
# Stage the ai-bond flash_attn_v100_cuda.so and run the MLA-prefill numerical check
# (tools/mla_prefill_v100_numcheck.py) inside the cu126 serving image. NO model load —
# pure kernel-vs-fp32-reference for the exact shapes vLLM's MLA prefill emits. ~1 GPU,
# seconds. Run this BEFORE the full GLM-4.7-Flash serve probe (cheap go/no-go).
#
# Usage:  ./tools/mla_prefill_v100_numcheck.sh
# Env:    IMAGE FA_DIR
# ======================================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
FA_DIR="${FA_DIR:-/home/kumphanartd/flash-attention-v100}"
SO_GLOB="$FA_DIR/build/lib.linux-x86_64-cpython-312/flash_attn_v100_cuda.*.so"
OUT=/tmp/v100_mla_numcheck
mkdir -p "$OUT"
note() { echo "[mla-numcheck] $*"; }

clean_box_guard() {
    local used pids
    used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
    pids=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [[ "${used:-9999}" -le 2000 && "${pids:-1}" -eq 0 ]]
}

docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "ABORT: image $IMAGE missing"; exit 1; }
# shellcheck disable=SC2086
ls $SO_GLOB >/dev/null 2>&1 || { note "ABORT: ai-bond .so not built at $SO_GLOB (build it first)"; exit 1; }
clean_box_guard || { note "SKIP: GPU0 busy (>2GB used or other CUDA apps)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 1; }

# Stage ONLY the .so (the ai-bond `flash_attn` python shim must NOT be importable).
rm -rf "$OUT/pylib"; mkdir -p "$OUT/pylib"
# shellcheck disable=SC2086
cp $SO_GLOB "$OUT/pylib/" || { note "ABORT: failed to stage .so"; exit 1; }
note "staged $(ls "$OUT/pylib")"

docker rm -f mla_numcheck >/dev/null 2>&1 || true
docker run --rm -i --name mla_numcheck --gpus '"device=0"' \
    -v "$PROJECT_ROOT":/work -w /work \
    -v "$OUT/pylib":/falib:ro \
    -e PYTHONPATH=/work/src:/falib \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" \
    python3 tools/mla_prefill_v100_numcheck.py \
    </dev/null 2>&1 | tee "$OUT/SUMMARY.txt"
rc=${PIPESTATUS[0]}
note "exit=$rc  (full log: $OUT/SUMMARY.txt)"
exit "$rc"
