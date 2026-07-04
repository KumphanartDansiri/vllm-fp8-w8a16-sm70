#!/usr/bin/env python3
"""Run the EXACT vendor_moe_smoke logic but with 1catai's ops (in the 1catai image).
Decides: is the grouped nan a build/flag issue (passes here) or a test-logic issue (fails here)?"""
import sys
import torch
from vllm import _custom_ops as ops
BLOCK = 128
dev = "cuda:0"


def cossim(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def deq(w, s, block=BLOCK):
    N, K = w.shape
    e = s.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return w.float() * e


def run(name, E, N, K, counts):
    g = torch.Generator().manual_seed(1)
    tmw, tms, refs = [], [], []
    k_ld = q_ld = None
    for e in range(E):
        w = (torch.randn(N, K, generator=g) * 0.2).clamp(-6, 6).to(torch.float8_e4m3fn).to(dev)
        s = ((torch.rand(N // BLOCK, K // BLOCK, generator=g) * 0.5 + 0.5).float().to(dev))
        refs.append(deq(w, s))
        r = ops.fp8_sm70_prepare(w, s, BLOCK)
        tmw.append(r[0]); tms.append(r[1])
        if e == 0:
            k_ld, q_ld = int(r[2][0]), int(r[2][1])
    ptrs = ops.awq_moe_build_strided_ptrs(torch.stack(tmw), torch.stack(tms), k_ld, q_ld, E)
    T = sum(counts)
    off = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    for e in range(E):
        off[e + 1] = off[e] + counts[e]
    ga = torch.Generator().manual_seed(7)
    x = (torch.randn(T, K, generator=ga) * 0.1).half().to(dev)
    out = torch.empty(T, N, dtype=torch.float16, device=dev)
    ops.fp8_moe_gemm_sm70_out(out, x, off, ptrs[0], ptrs[1], E, K, N, BLOCK, False)
    per = []
    worst = 1.0
    for e in range(E):
        lo, hi = int(off[e]), int(off[e + 1])
        if hi > lo:
            c = cossim(out[lo:hi], x[lo:hi].float() @ refs[e].T); worst = min(worst, c)
            per.append(f"e{e}:{c:.2f}")
    print(f"[{name}] E={E} N={N} K={K} T={T} nan={bool(out.isnan().any())} worst={worst:.4f} | {' '.join(per)}")


run("8exp_M2_Nqwen", E=8, N=1024, K=2048, counts=[2] * 8)   # exact vendor-smoke failing case
run("8exp_M2_Ncos", E=8, N=1536, K=1024, counts=[2] * 8)    # moe_cos_gate shape
run("8exp_M1", E=8, N=1024, K=2048, counts=[1] * 8)
run("Nqwen_M4", E=8, N=1024, K=2048, counts=[4] * 8)
run("Nqwen_M8", E=8, N=1024, K=2048, counts=[8] * 8)
run("Nqwen_M1_2exp", E=2, N=1024, K=2048, counts=[2, 2])


def run_aligned(name, E, N, K, real_m, align):
    """Each expert gets `align` slots (real_m real rows + padding), expert starts tile-aligned."""
    g = torch.Generator().manual_seed(1)
    tmw, tms, refs = [], [], []
    k_ld = q_ld = None
    for e in range(E):
        w = (torch.randn(N, K, generator=g) * 0.2).clamp(-6, 6).to(torch.float8_e4m3fn).to(dev)
        s = ((torch.rand(N // BLOCK, K // BLOCK, generator=g) * 0.5 + 0.5).float().to(dev))
        refs.append(deq(w, s)); r = ops.fp8_sm70_prepare(w, s, BLOCK)
        tmw.append(r[0]); tms.append(r[1])
        if e == 0: k_ld, q_ld = int(r[2][0]), int(r[2][1])
    ptrs = ops.awq_moe_build_strided_ptrs(torch.stack(tmw), torch.stack(tms), k_ld, q_ld, E)
    T = E * align
    off = torch.arange(0, E * align + 1, align, dtype=torch.int32, device=dev)  # aligned boundaries
    ga = torch.Generator().manual_seed(7)
    x = (torch.randn(T, K, generator=ga) * 0.1).half().to(dev)
    out = torch.empty(T, N, dtype=torch.float16, device=dev)
    ops.fp8_moe_gemm_sm70_out(out, x, off, ptrs[0], ptrs[1], E, K, N, BLOCK, False)
    worst = 1.0
    for e in range(E):  # check only the real_m rows of each expert
        lo = e * align
        c = cossim(out[lo:lo + real_m], x[lo:lo + real_m].float() @ refs[e].T); worst = min(worst, c)
    print(f"[{name}] E={E} K={K} align={align} real_m={real_m} T={T} worst={worst:.4f}")


run_aligned("aligned16_m2", E=8, N=1024, K=2048, real_m=2, align=16)
run_aligned("aligned8_m2",  E=8, N=1024, K=2048, real_m=2, align=8)
run_aligned("aligned32_m2", E=8, N=1024, K=2048, real_m=2, align=32)
