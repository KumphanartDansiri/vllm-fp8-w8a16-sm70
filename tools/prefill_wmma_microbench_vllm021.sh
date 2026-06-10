#!/usr/bin/env bash
# Phase 4 (prefill) SCOPING microbench — does a WMMA (tensor-core) MoE kernel
# justify the build? Decode is solved by cudagraph (~30.7 tok/s); the remaining
# long-context cost is PREFILL (169s TTFT @26k tokens), which runs eager CUDA-core
# kernels. This bench grounds the decision Codex asked for, with no model load:
#
#   (A) DENSE WMMA-vs-CUDA-core speedup on GLM Linear shapes (N=K=4096). The dense
#       CT Linears ALREADY use the WMMA PoC at prefill, so this is the realized
#       tensor-core speedup factor on V100 (HMMA.884, FP8->FP16-in-regs).
#   (B) GROUPED MoE kernel cost at prefill R (w13: FP8 grouped; w2: FP16 grouped).
#       These have NO WMMA path today (CUDA-core) — the suspected bottleneck.
#
# Decision rule: if (A) shows a big WMMA win AND (B) shows the grouped MoE dominates
# prefill FLOPs, then a WMMA grouped-MoE kernel is the lever. Else look elsewhere
# (attention prefill, etc.).
#
# Runs in the stock vllm021 image, seconds on ONE GPU. Usage:
#   ./tools/prefill_wmma_microbench_vllm021.sh
# Env: IMAGE GPU CACHE_TAG
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
GPU="${GPU:-0}"
CACHE_TAG="${CACHE_TAG:-021}"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE missing"; exit 1; }
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
[[ "${used:-9999}" -le 2000 ]] || echo "WARN: GPU $GPU has ${used} MiB used (shared box?). Set GPU= to a free one."

docker run --rm -i --name prefill_wmma_microbench --gpus "\"device=$GPU\"" \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python3 - <<'PY'
import torch, time
from fp8_w8a16_sm70.ext_loader import load_kernel
ext = load_kernel(name="fp8_dequant_ext_vllm")
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False

def tflops(flops, ms): return flops / (ms * 1e-3) / 1e12

def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters   # ms/iter

# ---- (A) DENSE: WMMA PoC vs A.2 (CUDA-core), GLM Linear shape N=K=4096 ----
print("== (A) DENSE FP8 W8A16 GEMM: WMMA(tensor-core) vs A.2(CUDA-core), N=4096 K=4096 ==")
N=K=4096; bh=bw=128
g=torch.Generator(device=dev).manual_seed(0)
# WMMA PoC requires block_h==block_w==128 (true block scale [N/128,K/128]). GLM CT
# weights are CHANNEL (block_h=1) — so this measures the RAW tensor-core speedup a
# channel-aware WMMA kernel could approach, not the current channel path (which
# falls to A.2 CUDA-core because wmma_layer_ok needs block_h==128).
scale=torch.rand(N//bh, K//bw, generator=g, device=dev, dtype=torch.float16)*0.02+0.01
wq=(torch.randn(N,K,generator=g,device=dev,dtype=torch.float16)*0.1).to(torch.float8_e4m3fn)
wu8=wq.view(torch.uint8).reshape(-1).contiguous(); sc=scale.reshape(-1).contiguous()
has_wmma=hasattr(ext,"fp8_w8a16_gemm_wmma_poc")
print(f"{'M':>6} {'A2_ms':>8} {'A2_TF':>7} {'WMMA_ms':>8} {'WMMA_TF':>8} {'speedup':>8}")
for M in (64,256,512,1024,2048):
    A=torch.randn(M,K,generator=g,device=dev,dtype=torch.float16)*0.1
    f=2*M*N*K
    a2=timeit(lambda: ext.fp8_w8a16_gemm_a2(A,wu8,sc,N,K,bh,bw))
    if has_wmma and M%64==0:
        wm=timeit(lambda: ext.fp8_w8a16_gemm_wmma_poc(A,wu8,sc,N,K,bh,bw))
        print(f"{M:>6} {a2:>8.3f} {tflops(f,a2):>7.2f} {wm:>8.3f} {tflops(f,wm):>8.2f} {a2/wm:>7.2f}x")
    else:
        print(f"{M:>6} {a2:>8.3f} {tflops(f,a2):>7.2f} {'-':>8} {'-':>8} {'-':>8}")

# ---- (B) GROUPED MoE kernels at prefill R (CUDA-core, no WMMA path today) ----
print("\n== (B) GROUPED MoE prefill cost (CUDA-core). GLM-Air TP8: E=128, w13 N=352 K=4096, w2 N=4096 K=176 ==")
E=128
# w13: FP8 grouped, channel scale block_h=1 block_w=128
N13,K13=352,4096
s13=(torch.rand(E,N13,1,generator=g,device=dev,dtype=torch.float16)*0.02+0.01).expand(E,N13,K13//128).contiguous()
w13=(torch.randn(E,N13,K13,generator=g,device=dev,dtype=torch.float16)*0.1).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
# w2: FP16 grouped
N2,K2=4096,176
w2=(torch.randn(E,N2,K2,generator=g,device=dev,dtype=torch.float16)*0.1).contiguous()
print(f"{'R(rows)':>8} {'w13_ms':>8} {'w13_TF':>7} {'w2_ms':>8} {'w2_TF':>7} {'sum_ms':>8}")
W13CHUNK=60000
for R in (512,2048,8192,32768):
    eids=torch.randint(0,E,(R,),generator=g,device=dev,dtype=torch.int64)
    x13=torch.randn(R,K13,generator=g,device=dev,dtype=torch.float16)*0.1
    x2=torch.randn(R,K2,generator=g,device=dev,dtype=torch.float16)*0.1
    def run13():
        if R<=W13CHUNK: ext.fp8_w8a16_grouped_routed_gemm_a3(x13,eids,w13,s13,N13,K13,1,128,8)
        else:
            for i in range(0,R,W13CHUNK):
                j=min(i+W13CHUNK,R); ext.fp8_w8a16_grouped_routed_gemm_a3(x13[i:j].contiguous(),eids[i:j].contiguous(),w13,s13,N13,K13,1,128,8)
    run2=lambda: ext.fp16_grouped_routed_gemm(x2,eids,w2,1) if hasattr(ext,"fp16_grouped_routed_gemm") else None
    m13=timeit(run13,iters=10,warmup=3)
    m2=timeit(run2,iters=10,warmup=3) if hasattr(ext,"fp16_grouped_routed_gemm") else float('nan')
    f13=2*R*N13*K13; f2=2*R*N2*K2
    print(f"{R:>8} {m13:>8.3f} {tflops(f13,m13):>7.2f} {m2:>8.3f} {tflops(f2,m2):>7.2f} {m13+m2:>8.3f}")

print("\n== EXTRAPOLATION to GLM-Air 26k-token prefill (R = 26000*topk8 = 208000 routed rows/layer, 45 MoE layers) ==")
print("   (linear scale the R=32768 grouped numbers; MoE-only, excludes attention + dense Linears)")
PY
rc=$?
echo "(exit $rc)"; exit $rc
