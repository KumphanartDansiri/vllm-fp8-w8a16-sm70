#!/usr/bin/env python3
# ======================================================================================
# A/B counterpart to fa_v100_microbench.py: benchmark vLLM 0.21's TRITON_ATTN
# unified_attention (the kernel GLM-Air serving ACTUALLY runs today) at the identical
# prefill shape and paged layout. Run inside vllm-v100:vllm021-cu126.
#
# Completes the go/no-go: ai-bond 112.6 ms/layer vs THIS number decides how much of
# the ~42s TTFT residual is the Triton attention kernel itself.
# ======================================================================================
import argparse
import sys
import time
import torch

try:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
except Exception as e:  # pragma: no cover
    print(f"[TRITON-BENCH] cannot import unified_attention: {e}")
    sys.exit(2)

DT = torch.float16
DEV = "cuda"


def attn_tflops(seqlen, hq, d, causal=True):
    f = 2.0 * 2.0 * hq * (seqlen ** 2) * d
    if causal:
        f *= 0.5
    return f / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=26000)
    ap.add_argument("--hq", type=int, default=12)
    ap.add_argument("--hk", type=int, default=1)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--layers", type=int, default=46)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(0)
    nb = (args.seqlen + args.block - 1) // args.block
    q = torch.randn(args.seqlen, args.hq, args.d, device=DEV, dtype=DT)
    k_cache = torch.randn(nb, args.block, args.hk, args.d, device=DEV, dtype=DT)
    v_cache = torch.randn(nb, args.block, args.hk, args.d, device=DEV, dtype=DT)
    block_table = torch.arange(nb, dtype=torch.int32, device=DEV).view(1, nb)
    cu_seqlens_q = torch.tensor([0, args.seqlen], dtype=torch.int32, device=DEV)
    seqused_k = torch.tensor([args.seqlen], dtype=torch.int32, device=DEV)
    out = torch.empty_like(q)
    scale = args.d ** -0.5

    print(f"[TRITON-BENCH] shape: seqlen={args.seqlen} Hq={args.hq} Hk={args.hk} "
          f"D={args.d} block={args.block} causal=True fp16 paged (vLLM 0.21 unified_attention)")

    def tri():
        unified_attention(
            q=q, k=k_cache, v=v_cache, out=out,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=args.seqlen,
            seqused_k=seqused_k, max_seqlen_k=args.seqlen,
            softmax_scale=scale, causal=True,
            window_size=[-1, -1], block_table=block_table, softcap=0.0,
            q_descale=None, k_descale=None, v_descale=None,
        )

    for _ in range(args.warmup):
        tri()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        tri()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / args.iters * 1e3
    tf = attn_tflops(args.seqlen, args.hq, args.d) / (ms / 1e3)
    est = ms * args.layers / 1e3
    print(f"[TRITON-BENCH] triton unified : {ms:8.3f} ms/layer  ~{tf:6.2f} TFLOP/s  "
          f"-> x{args.layers} layers = {est:6.2f} s")
    print(f"[TRITON-BENCH] vs ai-bond 112.625 ms/layer: triton/aibond = {ms/112.625:5.2f}x")


if __name__ == "__main__":
    main()
