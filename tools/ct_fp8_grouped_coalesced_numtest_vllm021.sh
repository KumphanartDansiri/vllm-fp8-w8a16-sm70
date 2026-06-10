#!/usr/bin/env bash
# Stage G1 STANDALONE correctness: the GROUPED coalesced routed GEMV
# (fp8_w8a16_grouped_gemv_coalesced) must match the current grouped a3 kernel
# AND an FP32 reference on GLM-Air w13 decode shapes (E=128, N=352, K=4096,
# channel block_h=1), R=1/8/64 (decode) + invalid routes + block_h=128 sanity.
# Validates BEFORE wiring into the decode path.
#
# Usage: ./tools/ct_fp8_grouped_coalesced_numtest_vllm021.sh ; Env: IMAGE GPU
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"; GPU="${GPU:-0}"; CACHE_TAG="${CACHE_TAG:-021}"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
docker run --rm -i --gpus "\"device=$GPU\"" \
    -v "$(pwd)":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python3 - <<'PY'
import torch, torch.nn.functional as F
from fp8_w8a16_sm70.ext_loader import load_kernel
ext = load_kernel(name="fp8_dequant_ext_vllm")
assert hasattr(ext, "fp8_w8a16_grouped_gemv_coalesced"), "rebuild — grouped coalesced kernel missing"
dev="cuda"; torch.backends.cuda.matmul.allow_tf32=False

def run(E, N, K, R, bh=1, invalid=False, seed=0):
    g=torch.Generator(device=dev).manual_seed(seed)
    if bh==1:
        cs=torch.rand(E,N,1,generator=g,device=dev,dtype=torch.float16)*0.02+0.01
        scl=cs.expand(E,N,K//128).contiguous()
    else:
        cs=torch.rand(E,N//128,K//128,generator=g,device=dev,dtype=torch.float16)*0.02+0.01
        scl=cs.contiguous()
    wq=(torch.randn(E,N,K,generator=g,device=dev,dtype=torch.float16)*0.1).to(torch.float8_e4m3fn)
    wu8=wq.view(torch.uint8).contiguous()
    A=torch.randn(R,K,generator=g,device=dev,dtype=torch.float16)*0.1
    eids=torch.randint(0,E,(R,),generator=g,device=dev,dtype=torch.int64)
    if invalid and R>=2: eids[0]=-1
    coal=ext.fp8_w8a16_grouped_gemv_coalesced(A,eids,wu8,scl,N,K,bh,128).float()
    ks=8 if (K%1024==0) else 4 if (K%512==0) else 1
    a3=ext.fp8_w8a16_grouped_routed_gemm_a3(A,eids,wu8,scl,N,K,bh,128,ks).float()
    # FP32 ref
    ref=torch.zeros(R,N,device=dev,dtype=torch.float32)
    if bh==1:
        wdq=wq.float()*cs.float()                       # [E,N,K]
    else:
        se=cs.repeat_interleave(128,1).repeat_interleave(128,2)[:,:N,:K]
        wdq=wq.float()*se.float()
    for rr in range(R):
        e=int(eids[rr])
        if e<0: continue
        ref[rr]=A[rr].float()@wdq[e].T
    def l2(a,b): return ((a-b).norm()/b.norm().clamp_min(1e-12)).item()
    ca=l2(coal,a3); cr=l2(coal,ref); cos=F.cosine_similarity(coal.flatten(),a3.flatten(),dim=0).item()
    zok=True
    if invalid and R>=2: zok=bool(coal[0].abs().max()==0)
    ok=(ca<5e-3 and cr<2e-2 and cos>0.9999 and zok)
    tag="block" if bh==128 else "chan"
    print(f"[{'PASS' if ok else 'FAIL'}] {tag} E={E} N={N} K={K} R={R}{' inv0' if invalid else ''} | "
          f"coal-vs-a3 L2={ca:.4f} cos={cos:.5f} | coal-vs-FP32={cr:.4f} zero_ok={zok}")
    return ok

print("-- grouped coalesced routed GEMV vs grouped a3 + FP32, GLM-Air w13 shapes --")
allok=True
allok&=run(128,352,4096,1)        # decode single routed row
allok&=run(128,352,4096,8)        # decode topk=8
allok&=run(128,352,4096,64)       # small batch
allok&=run(128,352,4096,8,invalid=True)   # -1 route -> zero
allok&=run(8,  352,4096,8,seed=3) # few experts
allok&=run(8,  512,4096,8,bh=128) # block_h=128 sanity (N%128==0)
print("\nRESULT:", "ALL PASS — grouped coalesced GEMV matches a3 + FP32"
      if allok else "FAIL (see rows)")
import sys; sys.exit(0 if allok else 1)
PY
rc=$?; echo "(exit $rc)"; exit $rc
