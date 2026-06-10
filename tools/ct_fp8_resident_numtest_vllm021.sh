#!/usr/bin/env bash
# Phase 1 numerical validation: CT channel-scale FP8-RESIDENT Linear kernel path.
#
# Proves the load-bearing claim WITHOUT loading a model: that running the V100
# W8A16 kernel on an FP8 weight + channel scale expanded to fake 128-wide blocks
# ([N, K/128], block_h=1, block_w=128) reproduces the dequant-to-FP16 reference
# (F.linear(x, fp8.to(fp16)*scale)) within FP16 tolerance — and that A.3's
# split-K decode path actually fires for K%1024/512==0 shapes (the whole reason
# we expand to block_w=128 instead of block_w=K).
#
# Runs in the stock vllm021 image with the package mounted (kernel JIT-cached).
# A few seconds on ONE GPU. Usage:  ./tools/ct_fp8_resident_numtest_vllm021.sh
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
[[ "${used:-9999}" -le 2000 ]] || { echo "WARN: GPU $GPU has ${used} MiB used (shared box?). Set GPU= to a free one."; }

docker run --rm -i --name ct_fp8_numtest --gpus "\"device=$GPU\"" \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python3 - <<'PY'
import torch, torch.nn.functional as F
from fp8_w8a16_sm70.vllm_serve import _v100_fp8_gemm

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False

def run_case(N, K, M, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    # positive per-row channel scale; random fp16 weight; quantize to e4m3.
    scale = (torch.rand(N, 1, generator=g, device=dev, dtype=torch.float16) * 0.02 + 0.01)
    w_ref = torch.randn(N, K, generator=g, device=dev, dtype=torch.float16) * 0.1
    w_q   = (w_ref / scale).to(torch.float8_e4m3fn)        # FP8 weight (resident)
    w_dq  = w_q.to(torch.float16) * scale                  # dequant-FP16 reference weight
    x     = torch.randn(M, K, generator=g, device=dev, dtype=torch.float16) * 0.1

    ref = F.linear(x, w_dq)                                # [M,N] reference

    # RESIDENT path: expand channel scale -> [N, K/128], block_w=128 (A.3-preserving)
    Kb = K // 128
    scale_blk = scale.expand(N, Kb).contiguous()
    out, variant = _v100_fp8_gemm(x, w_q, scale_blk, N, K, 1, 128)

    # cross-check: block_w=K (Kb=1) must give the SAME math (just no A.3)
    out_bwK, var_bwK = _v100_fp8_gemm(x, w_q, scale.expand(N, 1).contiguous(), N, K, 1, K)

    def err(a, b):
        a = a.float(); b = b.float()
        # L2 relative error bounds BOTH direction and magnitude; robust vs the
        # per-element near-zero-denominator blowup (output rows that ~cancel to 0).
        l2rel = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
        # max abs error normalized by the reference's max magnitude (not per-elem).
        maxn  = ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()
        cos   = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        return l2rel, maxn, cos

    l2, mn, cos     = err(out, ref)               # bw128 (A.3-preserving) vs reference
    l2K, mnK, cosK  = err(out_bwK, ref)            # bwK (forces A.1) vs reference
    l2X, mnX, cosX  = err(out, out_bwK)            # bw128 vs bwK: must agree (split-K loophole)
    ok_ref   = (cos  > 0.9999 and l2  < 2e-2)      # resident kernel ~ dequant-FP16
    ok_refK  = (cosK > 0.9999 and l2K < 2e-2)      # bwK path also correct
    ok_cross = (cosX > 0.9999 and l2X < 1e-2)      # the two block_w choices match
    ok = ok_ref and ok_refK and ok_cross
    print(f"[{'PASS' if ok else 'FAIL'}] N={N:>5} K={K:>5} M={M:>4} variant={variant:<8} | "
          f"bw128-vs-ref L2rel={l2:.4f} maxn={mn:.4f} cos={cos:.5f} | "
          f"bwK({var_bwK})-vs-ref L2rel={l2K:.4f} | "
          f"bw128-vs-bwK L2rel={l2X:.4f}"
          + ("" if ok else f"  <-- ref={ok_ref} refK={ok_refK} cross={ok_cross}"))
    return ok

# All cases GATE the suite. The non-128-aligned-N shapes (N=352/2736 = GLM-Air
# gate_up per TP8 shard) exercise the partial-N path that Phase 1.5 hardened in
# A.1/A.2/A.3 (all threads reach the cooperative load + __syncthreads; only the
# n-dependent work is masked). Before 1.5 these produced garbage; they must now
# PASS, which is the proof the hardening worked.
cases = [
    (4096, 4096,   1),   # decode, K%1024==0 -> A.3 k=8
    (4096, 4096,   8),
    (4096, 4096,  64),
    (4096, 4096, 512),   # prefill
    (12288, 4096,  1),   # o_proj-like N
    (1792, 4096,   1),   # qkv_proj-like N (14*128)
    (4096, 1408,   1),   # K%128==0 but not %512/1024 -> A.1
    (5120, 2048,   8),   # K%1024==0 -> A.3 k=4
    # --- partial-N (Phase 1.5): NOT 128-aligned; must PASS after hardening ---
    (352,  4096,   1),   # shared gate_up, M=1 -> A.3 split-K, partial last N-tile
    (352,  4096,   8),
    (352,  4096, 512),   # prefill, A.2 partial N-tile
    (2736, 4096,   1),   # dense gate_up, partial N-tile
    (2736, 4096,  64),
]
allok = True
for (N, K, M) in cases:
    allok &= run_case(N, K, M)

print("\nRESULT:", "ALL PASS (incl. partial-N) — channel FP8-resident kernel matches dequant-FP16 reference"
      if allok else "FAIL — see rows above")
PY
rc=$?
echo "(exit $rc)"; exit $rc
