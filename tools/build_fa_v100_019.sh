#!/usr/bin/env bash
# Rebuild the ai-bond flash-attention-v100 .so against vLLM 0.19's torch ABI (torch 2.10/cu128),
# so the V100 MLA-prefill hook (and the FA arm) work on the 0.19 engine. The default 0.21 build
# in $FA_DIR is built against torch 2.11/cu126 and fails to import on 0.19 with
#   undefined symbol: _ZNK3c104cuda10CUDAStream5queryEv  (c10::cuda::CUDAStream::query)
# -> on 0.19 the MLA-prefill hook then can't load and vLLM falls back to stock FlashAttention
#    MLA prefill, which raises "FlashAttention only supports Ampere GPUs or newer." on V100.
# This writes a separate 0.19-ABI build to FA_DIR_019 (perf_v2_cell.sh stages it for ENGINE=019).
# Needs a FREE GPU (setup.py checks torch.cuda.is_available()). ~10-20 min.
#   bash tools/build_fa_v100_019.sh            # GPU=6 by default
set -uo pipefail
FA_SRC="${FA_SRC:-/home/kumphanartd/flash-attention-v100}"
FA_DIR_019="${FA_DIR_019:-/home/kumphanartd/flash-attention-v100-019}"
IMAGE="${IMAGE:-vllm-v100-py312:vllm019-tf5}"
GPU="${GPU:-6}"

echo "[build-019] src=$FA_SRC -> $FA_DIR_019  image=$IMAGE  gpu=$GPU"
rm -rf "$FA_DIR_019"; mkdir -p "$FA_DIR_019"
rsync -a --exclude 'build/' --exclude '*.so' --exclude '.git/' "$FA_SRC/" "$FA_DIR_019/"
docker run --rm --gpus "\"device=$GPU\"" --entrypoint bash -e MAX_JOBS=6 -e NVCC_THREADS=2 \
  -v "$FA_DIR_019":/fa -w /fa "$IMAGE" -c \
  "python3 setup.py build_ext --inplace 2>&1; rc=\$?; chown -R $(id -u):$(id -g) /fa 2>/dev/null; echo FABUILD_EXIT=\$rc"
echo "[build-019] artifact:"
ls -l "$FA_DIR_019"/build/lib.linux-x86_64-cpython-312/flash_attn_v100_cuda.*.so 2>/dev/null \
  && echo "[build-019] OK" || echo "[build-019] FAILED (no .so)"
