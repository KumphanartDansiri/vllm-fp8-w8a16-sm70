#!/usr/bin/env python3
# ======================================================================================
# Go/no-go microbench: ai-bond flash-attention-v100 varlen forward at a GLM-4.5-Air
# prefill shape (Route A: paged, block_size=256, fp16 KV, causal, no split-KV).
#
# Reports ai-bond absolute latency + approx TFLOP/s, and a torch-SDPA baseline
# (V100 falls back to mem-efficient/math = CUDA cores; a sanity anchor, NOT vLLM-Triton).
# Extrapolates to --layers and compares to the documented in-model attention residual
# (~42s of the 60s TTFT@26k for GLM-Air, per docs/GLM45_AIR_V100_CONFIG.md).
#
# Defaults are GLM-4.5-Air TP8 per-rank, CONFIRMED vs config (Codex):
#   num_attention_heads=96, num_key_value_heads=8, head_dim=128, num_hidden_layers=46
#   -> per-rank HQ=12, HK=1, D=128, LAYERS=46
#
# Benches the LOW-LEVEL flash_attn_v100_cuda.varlen_fwd (the exact adapter call path —
# the public python wrapper hardcodes seqused_k=None/out=None and adds a copy).
# ======================================================================================
import argparse
import sys
import time
import torch

try:
    import flash_attn_v100_cuda
except Exception as e:  # pragma: no cover
    print(f"[BENCH] cannot import flash_attn_v100_cuda: {e}")
    sys.exit(2)

DT = torch.float16
DEV = "cuda"


def attn_tflops(seqlen, hq, d, causal=True):
    # QK^T + PV ; causal halves the work
    f = 2.0 * 2.0 * hq * (seqlen ** 2) * d
    if causal:
        f *= 0.5
    return f / 1e12


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/call


def build_paged(seqlen, hq, hk, d, block):
    nb = (seqlen + block - 1) // block
    q = torch.randn(seqlen, hq, d, device=DEV, dtype=DT)
    k_paged = torch.randn(nb, block, hk, d, device=DEV, dtype=DT)
    v_paged = torch.randn(nb, block, hk, d, device=DEV, dtype=DT)
    block_table = torch.arange(nb, dtype=torch.int32, device=DEV).view(1, nb)
    cu_q = torch.tensor([0, seqlen], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, seqlen], dtype=torch.int32, device=DEV)  # diff >= seqused
    seqused_k = torch.tensor([seqlen], dtype=torch.int32, device=DEV)
    out = torch.empty_like(q)
    scale = d ** -0.5
    return q, k_paged, v_paged, block_table, cu_q, cu_k, seqused_k, out, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=26000)
    ap.add_argument("--hq", type=int, default=12, help="per-rank query heads (GLM-Air TP8: 96/8)")
    ap.add_argument("--hk", type=int, default=1, help="per-rank KV heads (GLM-Air TP8: 8/8)")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--layers", type=int, default=46,
                    help="TTFT extrapolation; GLM-Air num_hidden_layers=46")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ref-seqlen-cap", type=int, default=8192,
                    help="cap SDPA baseline seqlen (O(n^2) mem) ; 0 = skip SDPA")
    args = ap.parse_args()

    print(f"[BENCH] shape: seqlen={args.seqlen} Hq={args.hq} Hk={args.hk} D={args.d} "
          f"block={args.block} causal=True fp16 paged")

    q, kp, vp, bt, cu_q, cu_k, su_k, out, scale = build_paged(
        args.seqlen, args.hq, args.hk, args.d, args.block)

    def ab():
        # low-level varlen_fwd, same positional contract as the smoke / future adapter
        flash_attn_v100_cuda.varlen_fwd(
            q, kp, vp, out, cu_q, cu_k, su_k, None, bt, None,
            args.seqlen, args.seqlen, 0.0, scale,
            False, True, -1, -1, 0.0, False, None, 0)

    ms = timed(ab, args.iters, args.warmup)
    tf = attn_tflops(args.seqlen, args.hq, args.d) / (ms / 1e3)
    est = ms * args.layers / 1e3
    print(f"[BENCH] ai-bond varlen : {ms:8.3f} ms/layer  ~{tf:6.2f} TFLOP/s  "
          f"-> x{args.layers} layers = {est:6.2f} s")

    # SDPA baseline (CUDA-core fallback on V100) at a capped seqlen, then scale ^2.
    if args.ref_seqlen_cap > 0:
        cap = min(args.ref_seqlen_cap, args.seqlen)
        g = args.hq // args.hk
        qd = torch.randn(1, args.hq, cap, args.d, device=DEV, dtype=DT)
        kd = torch.randn(1, args.hq, cap, args.d, device=DEV, dtype=DT)  # expand KV to Hq
        vd = torch.randn(1, args.hq, cap, args.d, device=DEV, dtype=DT)

        def sdpa():
            torch.nn.functional.scaled_dot_product_attention(qd, kd, vd, is_causal=True)

        ms_cap = timed(sdpa, args.iters, args.warmup)
        ms_sdpa = ms_cap * (args.seqlen / cap) ** 2  # O(n^2) scale to full seqlen
        est_sdpa = ms_sdpa * args.layers / 1e3
        print(f"[BENCH] torch-SDPA*   : {ms_sdpa:8.3f} ms/layer (scaled from {cap}) "
              f"-> x{args.layers} = {est_sdpa:6.2f} s   "
              f"[*CUDA-core anchor, NOT vLLM-Triton]")
        if ms_sdpa > 0:
            print(f"[BENCH] ai-bond speedup vs SDPA anchor: {ms_sdpa / ms:4.2f}x")

    print("[BENCH] GO/NO-GO: compare ai-bond x-layers estimate against the documented "
          "~42s in-model attention residual (GLM45_AIR_V100_CONFIG.md). "
          "Faster + correctness PASS => integrate the adapter.")


if __name__ == "__main__":
    main()
