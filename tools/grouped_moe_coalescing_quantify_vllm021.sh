#!/usr/bin/env bash
# QUANTIFY the grouped-MoE coalescing headroom (the "wire it correctly" question):
# the grouped routed decode kernels (fp8_w8a16_grouped_routed_gemm_a3 = w13,
# fp16_grouped_routed_gemm = w2) use the SAME N-strided uncoalesced read the
# coalesced GEMV fixed for dense Linears, but they're NOT wired to it. This bench
# measures (a) their per-layer decode cost at GLM-Air shapes (R=topk=8), (b) the
# coalesced LOWER BOUND (per-expert coalesced dense GEMV), and (c) their fraction
# of the 45.4 tok/s (=22 ms/token) decode budget — so we know the e2e upside
# BEFORE building a grouped coalesced kernel.
#
# Usage: ./tools/grouped_moe_coalescing_quantify_vllm021.sh ; Env: IMAGE GPU
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
import torch
from fp8_w8a16_sm70.ext_loader import load_kernel
ext = load_kernel(name="fp8_dequant_ext_vllm")
dev="cuda"; torch.backends.cuda.matmul.allow_tf32=False

def t(fn, it=200, wu=30):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it   # ms

# GLM-4.5-Air TP=8 decode: M=1 token, topk=8 -> R=8 routed rows. 45 MoE layers.
E=128; TOPK=8; R=TOPK; LAYERS=45; H=4096
N13=352; K13=H          # w13 (gate_up): [E, 2I/TP=352, H=4096], channel block_h=1
N2=H;    K2=176          # w2 (down):    [E, H=4096, I/TP=176]
g=torch.Generator(device=dev).manual_seed(0)

# ---- grouped w13 (FP8 a3) at R=8 ----
s13=(torch.rand(E,N13,1,generator=g,device=dev,dtype=torch.float16)*0.02+0.01).expand(E,N13,K13//128).contiguous()
w13=(torch.randn(E,N13,K13,generator=g,device=dev,dtype=torch.float16)*0.1).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
x13=torch.randn(R,K13,generator=g,device=dev,dtype=torch.float16)*0.1
eids=torch.randint(0,E,(R,),generator=g,device=dev,dtype=torch.int64)
ksplit=8
ms_w13=t(lambda: ext.fp8_w8a16_grouped_routed_gemm_a3(x13,eids,w13,s13,N13,K13,1,128,ksplit))

# ---- grouped w2 (FP16) at R=8 ----
w2=(torch.randn(E,N2,K2,generator=g,device=dev,dtype=torch.float16)*0.1).contiguous()
x2=torch.randn(R,K2,generator=g,device=dev,dtype=torch.float16)*0.1
ms_w2=t(lambda: ext.fp16_grouped_routed_gemm(x2,eids,w2,1))

# ---- coalesced LOWER BOUND for w13: per-expert coalesced dense GEMV x TOPK ----
# (one expert = [1,K13] x W[e][N13,K13]; a grouped coalesced kernel would parallelize
#  the TOPK experts, so TOPK x sequential is a CONSERVATIVE upper bound on its time.)
we=w13[0].reshape(-1).contiguous(); se=s13[0].reshape(-1).contiguous()
x1=torch.randn(1,K13,generator=g,device=dev,dtype=torch.float16)*0.1
ms_coal_1exp=t(lambda: ext.fp8_w8a16_gemv_coalesced(x1,we,se,N13,K13,1,128))
ms_coal_w13_lb=ms_coal_1exp*TOPK

# ---- reference: coalesced dense attn Linear (o_proj 4096x4096) = the path we DID coalesce ----
woN=woK=4096
so=(torch.rand(woN,1,generator=g,device=dev,dtype=torch.float16)*0.02+0.01).expand(woN,woK//128).contiguous()
wo=(torch.randn(woN,woK,generator=g,device=dev,dtype=torch.float16)*0.1).to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1).contiguous()
xo=torch.randn(1,woK,generator=g,device=dev,dtype=torch.float16)*0.1
ms_coal_oproj=t(lambda: ext.fp8_w8a16_gemv_coalesced(xo,wo,so,woN,woK,1,128))

budget=1000.0/45.4   # ms/token at 45.4 tok/s
moe_per_layer=ms_w13+ms_w2
moe_e2e=moe_per_layer*LAYERS
print("== GLM-4.5-Air TP=8 decode (M=1, topk=8 -> R=8), grouped MoE kernels ==")
print(f"  grouped w13 (FP8 a3, uncoalesced)   : {ms_w13*1000:8.1f} us/call")
print(f"  grouped w2  (FP16, uncoalesced)     : {ms_w2*1000:8.1f} us/call")
print(f"  -> MoE kernels per layer            : {moe_per_layer*1000:8.1f} us")
print(f"  -> x{LAYERS} layers (e2e/token)          : {moe_e2e:8.3f} ms  ({moe_e2e/budget*100:.0f}% of the {budget:.2f} ms/token decode budget)")
print()
print("== coalescing HEADROOM for w13 ==")
print(f"  coalesced dense GEMV, 1 expert      : {ms_coal_1exp*1000:8.1f} us")
print(f"  coalesced w13 LOWER BOUND (x{TOPK} seq)  : {ms_coal_w13_lb*1000:8.1f} us   (parallelized would be less)")
print(f"  grouped w13 a3 / coalesced-LB       : {ms_w13/ms_coal_w13_lb:8.2f}x  (headroom on w13)")
print()
print("== reference: the attn Linear we ALREADY coalesced (o_proj 4096x4096) ==")
print(f"  coalesced o_proj GEMV               : {ms_coal_oproj*1000:8.1f} us/call (this is the FAST path now)")
print()
# crude e2e projection: if w13 reaches its coalesced LB, MoE per-layer drops by (ms_w13 - ms_coal_w13_lb)
saved=(ms_w13-ms_coal_w13_lb)*LAYERS
print(f"== crude projection: coalescing w13 saves ~{saved:.2f} ms/token ({saved/budget*100:.0f}% of budget) "
      f"-> ~{45.4/(1-saved/budget) if saved<budget else float('inf'):.1f} tok/s if it's pure GPU time ==")
print("  (upper bound — real gain tempered by all-reduce + non-kernel decode; w2 K=176 needs coalesced K-tail separately)")
PY
echo "(done)"
