#!/usr/bin/env bash
# Run ViT attention microbench on one V100:
#   Torch SDPA D=72 vs ai-bond FA-V100 with D=72 padded to D=128.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROJECT_ROOT="$(pwd)"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
GPU="${GPU:-4}"
FA_DIR="${FA_DIR:-/home/kumphanartd/flash-attention-v100}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$PROJECT_ROOT/results/vit_fa_v100_d72_${TS}}"
CACHE_TAG="${CACHE_TAG:-021}"

mkdir -p "$OUT"
for s in torchext triton torch inductor; do
    mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"
done

docker image inspect "$IMAGE" >/dev/null

used="$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')"
if [[ "${used:-9999}" -gt 2000 ]]; then
    echo "[vit-fa] GPU $GPU busy (${used} MiB used); choose GPU=..." >&2
    exit 1
fi

so_glob="$FA_DIR/build/lib.linux-x86_64-cpython-312/flash_attn_v100_cuda."*.so
if ! compgen -G "$so_glob" >/dev/null; then
    echo "[vit-fa] missing flash_attn_v100_cuda build under $FA_DIR/build" >&2
    exit 4
fi

# Stage ONLY the low-level extension. Do not expose ai-bond's flash_attn Python
# shims to vLLM/Python import resolution.
docker run --rm -v "$FA_DIR":/fasrc:ro -v "$OUT":/out alpine sh -c \
    "mkdir -p /out/pylib && cp /fasrc/build/lib.linux-x86_64-cpython-312/flash_attn_v100_cuda.*.so /out/pylib/ && chown -R $(id -u):$(id -g) /out/pylib"

echo "[vit-fa] results -> $OUT"
docker run --rm -i --name "vit_fa_v100_d72_${TS}" --gpus "\"device=$GPU\"" \
    -v "$PROJECT_ROOT":/work -w /work \
    -v "$OUT/pylib":/fa-pylib:ro \
    -e PYTHONPATH="/fa-pylib:/work/src" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python3 tools/vit_fa_v100_d72_microbench.py "$@" \
    2>&1 | tee "$OUT/bench.log"
