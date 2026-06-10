#!/usr/bin/env bash
# Phase 4 Stage 2 STANDALONE correctness: the WMMA kernel extended to CHANNEL scale
# (block_h=1, per-output-row scale) must match an FP32 dequant reference on GLM
# dense Linear shapes, AND the original block_h=128 path must still pass (Qwen
# no-regression). This validates the kernel BEFORE wiring it into the dispatch.
#
# Usage: ./tools/ct_fp8_channel_wmma_numtest_vllm021.sh
# Env: IMAGE GPU CACHE_TAG
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"; GPU="${GPU:-0}"; CACHE_TAG="${CACHE_TAG:-021}"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE missing"; exit 1; }
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
[[ "${used:-9999}" -le 2000 ]] || echo "WARN: GPU $GPU has ${used} MiB used. Set GPU= to a free one."

docker run --rm -i --name ct_channel_wmma_numtest --gpus "\"device=$GPU\"" \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python3 - <<'PY'
import torch, torch.nn.functional as F
from fp8_w8a16_sm70.ext_loader import load_kernel
ext = load_kernel(name="fp8_dequant_ext_vllm")
assert hasattr(ext, "fp8_w8a16_gemm_wmma_poc")
dev="cuda"; torch.backends.cuda.matmul.allow_tf32=False

def chan_case(M, N, K, seed=0):
    g=torch.Generator(device=dev).manual_seed(seed)
    cs=torch.rand(N,1,generator=g,device=dev,dtype=torch.float16)*0.02+0.01     # channel [N,1]
    wq=(torch.randn(N,K,generator=g,device=dev,dtype=torch.float16)*0.1).to(torch.float8_e4m3fn)
    A=torch.randn(M,K,generator=g,device=dev,dtype=torch.float16)*0.1
    wu8=wq.view(torch.uint8).reshape(-1).contiguous()
    scale_blk=cs.expand(N, K//128).contiguous().reshape(-1)                     # [N,K/128] fake-block
    out=ext.fp8_w8a16_gemm_wmma_poc(A, wu8, scale_blk, N, K, 1, 128).float()    # block_h=1 CHANNEL
    ref=(A.float() @ (wq.float()*cs.float()).T)
    l2=((out-ref).norm()/ref.norm().clamp_min(1e-12)).item()
    cos=F.cosine_similarity(out.flatten(),ref.flatten(),dim=0).item()
    ok=(l2<2e-2 and cos>0.9999)
    print(f"[{'PASS' if ok else 'FAIL'}] CHANNEL M={M} N={N} K={K} | L2rel={l2:.4f} cos={cos:.5f}")
    return ok

def block_case(M, N, K, seed=0):  # block_h=128 no-regression (Qwen)
    g=torch.Generator(device=dev).manual_seed(seed)
    Nb,Kb=N//128,K//128
    bs=torch.rand(Nb,Kb,generator=g,device=dev,dtype=torch.float16)*0.02+0.01
    wq=(torch.randn(N,K,generator=g,device=dev,dtype=torch.float16)*0.1).to(torch.float8_e4m3fn)
    A=torch.randn(M,K,generator=g,device=dev,dtype=torch.float16)*0.1
    wu8=wq.view(torch.uint8).reshape(-1).contiguous()
    out=ext.fp8_w8a16_gemm_wmma_poc(A, wu8, bs.reshape(-1), N, K, 128, 128).float()
    se=bs.repeat_interleave(128,0).repeat_interleave(128,1)[:N,:K]
    ref=(A.float() @ (wq.float()*se.float()).T)
    l2=((out-ref).norm()/ref.norm().clamp_min(1e-12)).item()
    ok=(l2<2e-2)
    print(f"[{'PASS' if ok else 'FAIL'}] BLOCK   M={M} N={N} K={K} | L2rel={l2:.4f}")
    return ok

print("-- channel-WMMA (block_h=1) vs FP32 ref, GLM dense shapes (N%64==0) --")
allok=True
allok&=chan_case(64,  4096,4096)    # o_proj-ish, decode-tail M
allok&=chan_case(256, 4096,4096)
allok&=chan_case(2048,4096,4096)    # prefill M
allok&=chan_case(512, 1792,4096)    # qkv-ish N (1792%64==0)
allok&=chan_case(64,  4096,4096, seed=7)
print("-- block_h=128 no-regression (Qwen block-FP8) --")
allok&=block_case(256,4096,4096)
allok&=block_case(2048,4096,4096)
print("\nRESULT:", "ALL PASS — channel-WMMA correct + block_h=128 no-regression"
      if allok else "FAIL (see rows)")
import sys; sys.exit(0 if allok else 1)
PY
rc=$?; echo "(exit $rc)"; exit $rc
