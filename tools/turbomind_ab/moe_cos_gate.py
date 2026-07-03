#!/usr/bin/env python3
"""
Stage-A grouped-MoE COS GATE — 1catai/1Cat-vLLM SM70 FP8 TurboMind `fp8_moe_gemm_sm70_out`.

PURPOSE (correctness FIRST — no timing here)
  Before any perf A/B, prove the grouped MoE GEMM produces correct math on V100.
  Drives `fp8_moe_gemm_sm70_out` directly (bypassing moe_permute) with a hand-built
  sorted input + expert_offsets so the reference is deterministic and kernel-neutral.
  Gate: cos(kernel_out, fp32 dequant ref) >= COS_PASS for EVERY expert range, across
  BOTH GEMM shapes in a real MoE block and BOTH routing regimes.

CONTRACT (mirrors vllm/.../fp8_sm70_moe.py exactly)
  per expert:  fp8_sm70_prepare(W_e[N,K] fp8_e4m3, scale_e[N/128,K/128] fp32, 128)
               -> tm_w[K,N] u8, tm_s[K/128,N] f16, meta=[k_ld,q_ld]
  ptrs = awq_moe_build_strided_ptrs(stack(tm_w)[E,K,N], stack(tm_s)[E,K/128,N],
                                    k_ld, q_ld, E)            # k_ld,q_ld from expert-0 meta
  fp8_moe_gemm_sm70_out(out[T,N], sorted_in[T,K], expert_offsets[E+1] i32,
                        ptrs_w, ptrs_s, E, k=K, n=N, 128, gated_silu=False)
  semantics: for expert e, rows [off[e],off[e+1]) : out = sorted_in @ dequant(W_e).T

Two GEMM shapes gated (a real MoE block):  w13 (N=2I, K=H)  and  w2 (N=H, K=I).
Two routing regimes:  spread (rows even across all E)  and  hot (all rows -> 1 expert).

Run inside the 1catai SM70 FP8 image. Self-contained; synthesizes its own weights.
"""
import argparse, sys
import torch
from vllm import _custom_ops as ops

COS_PASS = 0.99


def cossim(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def dequant_ref(W_fp8, scales, block):
    N, K = W_fp8.shape
    s = scales.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return W_fp8.float() * s


def make_expert_weights(E, N, K, block, seed, dev):
    """E experts, each W[N,K] fp8 + block scales; returns stacked tm_w, tm_s, ptrs, refs."""
    tm_ws, tm_ss, refs = [], [], []
    k_ld = q_ld = None
    for e in range(E):
        g = torch.Generator().manual_seed(seed + 100 * e)
        w = (torch.randn(N, K, generator=g) * 0.2).clamp(-6, 6)
        W_fp8 = w.to(torch.float8_e4m3fn).to(dev)
        scales = ((torch.rand(N // block, K // block, generator=g) * 0.5 + 0.5)
                  .float().to(dev))
        refs.append(dequant_ref(W_fp8, scales, block))            # fp32 [N,K]
        tm_w, tm_s, meta = ops.fp8_sm70_prepare(W_fp8, scales, block)
        tm_ws.append(tm_w); tm_ss.append(tm_s)
        if e == 0:
            k_ld, q_ld = int(meta[0].item()), int(meta[1].item())
    W_stack = torch.stack(tm_ws)                                  # [E,K,N]
    S_stack = torch.stack(tm_ss)                                  # [E,K/128,N]
    ptrs = ops.awq_moe_build_strided_ptrs(W_stack, S_stack, k_ld, q_ld, E)
    return W_stack, S_stack, ptrs, refs, k_ld, q_ld


def routing_counts(mode, E, tpe):
    """Return per-expert row counts. spread: tpe each. hot: all on expert 0."""
    if mode == "spread":
        return [tpe] * E
    if mode == "hot":
        c = [0] * E; c[0] = tpe * E; return c
    if mode == "skew":                      # expert 0 heavy, a couple light, rest empty
        c = [0] * E; c[0] = tpe * E
        if E > 1: c[1] = tpe
        if E > 2: c[2] = 1
        return c
    raise ValueError(mode)


def gate_one(name, E, N, K, block, mode, tpe, seed, dev):
    W_stack, S_stack, ptrs, refs, k_ld, q_ld = make_expert_weights(
        E, N, K, block, seed, dev)
    counts = routing_counts(mode, E, tpe)
    T = sum(counts)
    offsets = torch.zeros(E + 1, dtype=torch.int32)
    for e in range(E):
        offsets[e + 1] = offsets[e] + counts[e]
    offsets = offsets.to(dev)

    ga = torch.Generator().manual_seed(seed + 7)
    sorted_in = (torch.randn(T, K, generator=ga) * 0.1).to(torch.float16).to(dev)
    out = torch.empty(T, N, dtype=torch.float16, device=dev)
    ops.fp8_moe_gemm_sm70_out(out, sorted_in, offsets, ptrs[0], ptrs[1],
                              E, K, N, block, False)

    # per-expert reference + cos
    worst = 1.0; worst_e = -1; per = []
    for e in range(E):
        lo, hi = int(offsets[e].item()), int(offsets[e + 1].item())
        if hi == lo:
            per.append((e, 0, None)); continue
        ref = sorted_in[lo:hi].float() @ refs[e].T                # [rows,N]
        c = cossim(out[lo:hi], ref)
        per.append((e, hi - lo, c))
        if c < worst:
            worst = c; worst_e = e
    ok = worst >= COS_PASS
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name:<20} mode={mode:<6} E={E} N={N} K={K} tpe={tpe} T={T} "
          f"k_ld={k_ld} q_ld={q_ld}  worst_cos={worst:.4f}@e{worst_e}")
    detail = "  ".join(f"e{e}:{'--' if c is None else f'{c:.4f}'}({r})"
                       for e, r, c in per)
    print(f"       per-expert cos(rows): {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=8, help="num experts")
    ap.add_argument("--H", type=int, default=1024, help="hidden size (mult 128)")
    ap.add_argument("--I", type=int, default=768, help="intermediate size (mult 128)")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--tpe", type=int, nargs="+", default=[1, 2, 4],
                    help="tokens-per-expert sweep (effective decode M)")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    if not hasattr(torch.ops._C, "fp8_moe_gemm_sm70_out"):
        sys.exit("[FATAL] fp8_moe_gemm_sm70_out missing — not a 1catai SM70 FP8 MoE build.")
    dev = "cuda:0"
    E, H, I, b = args.E, args.H, args.I, args.block

    # Two GEMM shapes in one MoE block:  w13 = gate_up (N=2I,K=H); w2 = down (N=H,K=I)
    shapes = [("w13_gate_up", 2 * I, H), ("w2_down", H, I)]
    print(f"# grouped-MoE cos gate  E={E} H={H} I={I} block={b}  COS_PASS={COS_PASS}")
    print(f"# w13: N={2*I} K={H}   w2: N={H} K={I}\n")

    results = []
    for name, N, K in shapes:
        for mode in ("spread", "hot", "skew"):
            for tpe in args.tpe:
                results.append(gate_one(name, E, N, K, b, mode, tpe, args.seed, dev))
            print()

    n_pass = sum(results); n = len(results)
    print(f"=== GATE {'PASS' if n_pass == n else 'FAIL'}: {n_pass}/{n} configs cos>={COS_PASS} ===")
    sys.exit(0 if n_pass == n else 1)


if __name__ == "__main__":
    main()
