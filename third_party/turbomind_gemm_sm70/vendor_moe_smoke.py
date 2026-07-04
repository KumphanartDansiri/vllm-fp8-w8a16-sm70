#!/usr/bin/env python3
"""Isolate the vendored grouped-MoE nan: synthetic experts, controllable empties.
Tests turbomind_fp8_sm70.fp8_moe_gemm_sm70_out (w13 only) vs fp32 dequant ref."""
import sys
import torch
from _ext_build import build_ops, BLOCK

ops = build_ops()
dev = "cuda:0"


def cossim(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def deq(w, s, block=BLOCK):
    N, K = w.shape
    e = s.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return w.float() * e


def run(name, E, N, K, counts):
    """counts[e] = rows for expert e (0 => empty)."""
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
    # ref per expert range
    worst = 1.0
    for e in range(E):
        lo, hi = int(off[e]), int(off[e + 1])
        if hi == lo:
            continue
        ref = x[lo:hi].float() @ refs[e].T
        worst = min(worst, cossim(out[lo:hi], ref))
    per = []
    for e in range(E):
        lo, hi = int(off[e]), int(off[e + 1])
        if hi > lo:
            per.append(f"e{e}:{cossim(out[lo:hi], x[lo:hi].float() @ refs[e].T):.2f}")
    print(f"[{name}] E={E} T={T} nan={bool(out.isnan().any())} worst={worst:.4f} | {' '.join(per)}")
    return worst


run("1exp_M2", E=8, N=1024, K=2048, counts=[0, 2, 0, 0, 0, 0, 0, 0])
run("2exp_M2", E=8, N=1024, K=2048, counts=[2, 2, 0, 0, 0, 0, 0, 0])
run("4exp_M1", E=8, N=1024, K=2048, counts=[1, 1, 1, 1, 0, 0, 0, 0])
run("8exp_M1", E=8, N=1024, K=2048, counts=[1] * 8)
run("8exp_M2", E=8, N=1024, K=2048, counts=[2] * 8)
