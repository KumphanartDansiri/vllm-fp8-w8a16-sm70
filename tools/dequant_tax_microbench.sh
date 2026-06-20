#!/usr/bin/env bash
# Run the FP8-vs-GPTQ standalone dequant tax microbench on one V100.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROJECT_ROOT="$(pwd)"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
GPU="${GPU:-4}"
CACHE_TAG="${CACHE_TAG:-dequant_tax}"

for s in torchext triton torch inductor; do
    mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"
done

docker image inspect "$IMAGE" >/dev/null

used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
if [[ "${used:-9999}" -gt 2000 ]]; then
    echo "[dequant-tax] GPU $GPU busy (${used} MiB used); choose GPU=..." >&2
    exit 1
fi

docker run --rm -i --name dequant_tax_microbench --gpus "\"device=$GPU\"" \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python3 tools/dequant_tax_microbench.py "$@"
