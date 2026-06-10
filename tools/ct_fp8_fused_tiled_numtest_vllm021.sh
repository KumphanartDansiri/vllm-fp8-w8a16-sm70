#!/usr/bin/env bash
# Phase 4 Stage 1.5 STANDALONE correctness: the FUSED grouped-tiled kernel
# (fp8_w8a16_grouped_tiled_gemm, ONE launch + GPU-side offsets) must match the
# per-expert a2 loop (the current tiled path) AND an FP32 reference, on GLM-Air w13
# shapes (E=128, N=2I=352 partial-N, K=H=4096), channel scale (block_h=1). This
# validates the kernel + the sync-free route-prep BEFORE wiring into the apply.
#
# Usage: ./tools/ct_fp8_fused_tiled_numtest_vllm021.sh ; Env: IMAGE GPU CACHE_TAG
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"; IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"; GPU="${GPU:-0}"; CACHE_TAG="${CACHE_TAG:-021}"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE missing"; exit 1; }

docker run --rm -i --name ct_fused_tiled_numtest --gpus "\"device=$GPU\"" \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python3 - <<'PY'
import torch, torch.nn.functional as F
from fp8_w8a16_sm70.ext_loader import load_kernel
ext = load_kernel(name="fp8_dequant_ext_vllm")
assert hasattr(ext, "fp8_w8a16_grouped_tiled_gemm"), "rebuild — fused kernel missing"
dev="cuda"; BM=8; torch.backends.cuda.matmul.allow_tf32=False

def route_prep(expert_ids, route_x, E):
    # ALL sync-free GPU ops (no .item / .tolist)
    order=torch.argsort(expert_ids)
    A_sorted=route_x.index_select(0,order).contiguous()
    counts=torch.zeros(E,dtype=torch.int32,device=dev)
    counts.scatter_add_(0,expert_ids,torch.ones_like(expert_ids,dtype=torch.int32))
    tiles=((counts+BM-1)//BM).to(torch.int32)
    e_tile_off=(torch.cumsum(tiles,0)-tiles).to(torch.int32)
    e_row_off=(torch.cumsum(counts,0)-counts).to(torch.int32)
    return order,A_sorted,e_tile_off,tiles,e_row_off,counts

def run_case(E,N,K,R,seed=0):
    g=torch.Generator(device=dev).manual_seed(seed)
    cs=torch.rand(E,N,1,generator=g,device=dev,dtype=torch.float16)*0.02+0.01
    wq=(torch.randn(E,N,K,generator=g,device=dev,dtype=torch.float16)*0.1).to(torch.float8_e4m3fn)
    wu8=wq.view(torch.uint8).contiguous()
    scl=cs.expand(E,N,K//128).contiguous()                       # [E,N,K/128] channel
    A=torch.randn(R,K,generator=g,device=dev,dtype=torch.float16)*0.1
    eids=torch.randint(0,E,(R,),generator=g,device=dev,dtype=torch.int64)

    # fused
    order,A_s,eto,tpe,ero,cnt=route_prep(eids,A,E)
    Cs=ext.fp8_w8a16_grouped_tiled_gemm(A_s,eto,tpe,ero,cnt,wu8,scl,N,K,1,128)
    fused=torch.empty(R,N,device=dev,dtype=torch.float16); fused.index_copy_(0,order,Cs)

    # reference 1: per-expert a2 (current tiled path)
    a2=torch.empty(R,N,device=dev,dtype=torch.float16)
    for e in range(E):
        m=(eids==e)
        if m.any(): a2[m]=ext.fp8_w8a16_gemm_a2(A[m].contiguous(),wu8[e].reshape(-1),scl[e].reshape(-1),N,K,1,128)
    # reference 2: FP32
    ref=torch.empty(R,N,device=dev,dtype=torch.float32)
    for r in range(min(R,4096)):
        e=int(eids[r]); ref[r]=A[r].float()@(wq[e].float()*cs[e].float()).T
    rr=min(R,4096)
    def l2(a,b,n=R): return ((a[:n].float()-b[:n]).norm()/b[:n].float().norm().clamp_min(1e-12)).item() if b.dtype==torch.float32 else ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)).item()
    fa=l2(fused,a2); fr=l2(fused[:rr],ref[:rr],rr)
    cos=F.cosine_similarity(fused.flatten().float(),a2.flatten().float(),dim=0).item()
    ok=(fa<5e-3 and fr<2e-2 and cos>0.9999)
    print(f"[{'PASS' if ok else 'FAIL'}] E={E} N={N} K={K} R={R} | fused-vs-a2 L2={fa:.4f} cos={cos:.5f} | fused-vs-FP32={fr:.4f}")
    return ok

print("-- fused grouped-tiled (one launch) vs per-expert a2 + FP32, GLM-Air w13 shapes --")
allok=True
allok&=run_case(E=128,N=352,K=4096,R=512)      # GLM-Air w13, small prefill
allok&=run_case(E=128,N=352,K=4096,R=8192)     # larger prefill
allok&=run_case(E=8,  N=352,K=4096,R=37)       # tiny + non-BM-aligned counts (tile masking)
allok&=run_case(E=128,N=352,K=4096,R=64,seed=3)# many experts, few rows (0-tile experts)
allok&=run_case(E=16, N=512,K=4096,R=1000)     # aligned-N sanity
print("\nRESULT:", "ALL PASS — fused grouped-tiled matches per-expert a2 + FP32"
      if allok else "FAIL (see rows)")
import sys; sys.exit(0 if allok else 1)
PY
rc=$?; echo "(exit $rc)"; exit $rc
