#!/usr/bin/env python3
"""Gemma-4-31B FP8 resident Linear shape microbench.

Parses a Gemma FP8 serve.log for resident CT-channel Linear shapes, times each
unique per-rank shape at decode M={1,2,4,8}, and sums the weighted per-step
Linear time. This is meant to explain the FP8-vs-FP16 concurrency gap without
loading the full model.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from fp8_w8a16_sm70.ext_loader import load_kernel


RESIDENT_RE = re.compile(r"resident N=(\d+),K=(\d+),Kb=\d+,bw=128")
MS = (1, 2, 4, 8)


@dataclass(frozen=True)
class Shape:
    n: int
    k: int
    count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--serve-log",
        default="results/gemma4_31b_revisit_fp8_021_20260618_042353/serve.log",
    )
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260618)
    return p.parse_args()


def parse_shapes(path: str) -> list[Shape]:
    counts: Counter[tuple[int, int]] = Counter()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RESIDENT_RE.search(line)
            if m:
                counts[(int(m.group(1)), int(m.group(2)))] += 1
    if not counts:
        raise SystemExit(f"no resident shapes found in {path}")
    return [Shape(n, k, c) for (n, k), c in sorted(counts.items())]


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        y = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        y = fn()
    end.record()
    torch.cuda.synchronize()
    if y.numel() == 0:
        raise RuntimeError("unexpected empty output")
    return start.elapsed_time(end) / iters


def make_case(n: int, k: int, m: int, seed: int):
    block_h = 1
    block_w = 128
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda", dtype=torch.float16) * 0.1
    w_ref = torch.randn(n, k, generator=g, device="cuda", dtype=torch.float16) * 0.1
    scales = (
        torch.rand(
            math.ceil(n / block_h),
            math.ceil(k / block_w),
            generator=g,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.02
        + 0.01
    )
    expanded = scales.repeat_interleave(block_h, 0)[:n]
    expanded = expanded.repeat_interleave(block_w, 1)[:, :k]
    w_q = (w_ref / expanded).to(torch.float8_e4m3fn).contiguous()
    w_dq = (w_q.to(torch.float16) * expanded).contiguous()
    return x.contiguous(), w_q, w_q.view(torch.uint8).reshape(-1).contiguous(), scales.reshape(-1).contiguous(), w_dq


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    shapes = parse_shapes(args.serve_log)
    ext = load_kernel(name="fp8_dequant_ext_gemma4_shape_microbench")

    from fp8_w8a16_sm70.vllm_serve import _v100_fp8_gemm

    print("Gemma-4-31B FP8 resident Linear shape microbench")
    print(f"serve_log={args.serve_log}")
    print(f"unique_shapes={len(shapes)} total_resident_linears={sum(s.count for s in shapes)}")
    print("block_h=1 block_w=128 (CT-channel resident path); ms/call")
    print()

    totals = {
        m: {"cublas": 0.0, "coal": 0.0, "wrapper": 0.0, "tmpdq_cublas": 0.0}
        for m in MS
    }
    print(
        f"{'count':>5s} {'M':>2s} {'N':>6s} {'K':>6s} "
        f"{'cuBLAS':>8s} {'coal_m':>8s} {'wrapper':>8s} {'tmpdq+blas':>10s} "
        f"{'variant':>16s} {'weighted_delta':>14s}"
    )
    print("-" * 98)

    for si, shape in enumerate(shapes):
        for m in MS:
            x, w_q, w_u8, scales, w_dq = make_case(
                shape.n, shape.k, m, args.seed + 1000 * si + m)
            cublas = time_cuda(lambda: F.linear(x, w_dq), args.warmup, args.iters)
            coal = time_cuda(
                lambda: ext.fp8_w8a16_gemv_coalesced_m(
                    x, w_u8, scales, shape.n, shape.k, 1, 128),
                args.warmup,
                args.iters,
            )
            holder = {"variant": ""}

            def wrapped():
                out, variant = _v100_fp8_gemm(
                    x, w_q, scales, shape.n, shape.k, 1, 128)
                holder["variant"] = variant
                return out

            def temp_dequant_cublas():
                w_tmp = ext.fp8_e4m3_to_fp16_block_scaled(
                    w_u8, scales, shape.n, shape.k, 1, 128).reshape(shape.n, shape.k)
                return F.linear(x, w_tmp)

            wrapper = time_cuda(wrapped, args.warmup, args.iters)
            tmpdq_cublas = time_cuda(temp_dequant_cublas, args.warmup, args.iters)
            delta = (wrapper - cublas) * shape.count
            totals[m]["cublas"] += cublas * shape.count
            totals[m]["coal"] += coal * shape.count
            totals[m]["wrapper"] += wrapper * shape.count
            totals[m]["tmpdq_cublas"] += tmpdq_cublas * shape.count
            print(
                f"{shape.count:5d} {m:2d} {shape.n:6d} {shape.k:6d} "
                f"{cublas:8.3f} {coal:8.3f} {wrapper:8.3f} {tmpdq_cublas:10.3f} "
                f"{holder['variant']:>16s} {delta:14.3f}"
            )

    print()
    print("Weighted per-rank resident-Linear sum per decode step (ms):")
    print(
        f"{'M':>2s} {'cuBLAS':>10s} {'coal_m':>10s} {'wrapper':>10s} "
        f"{'tmpdq+blas':>11s} {'delta':>10s} {'tmpdq_delta':>12s} {'ratio':>8s}"
    )
    print("-" * 84)
    for m in MS:
        c = totals[m]["cublas"]
        w = totals[m]["wrapper"]
        t = totals[m]["tmpdq_cublas"]
        ratio = w / c if c else float("nan")
        print(
            f"{m:2d} {c:10.3f} {totals[m]['coal']:10.3f} {w:10.3f} "
            f"{t:11.3f} {w-c:10.3f} {t-c:12.3f} {ratio:8.3f}"
        )


if __name__ == "__main__":
    main()
