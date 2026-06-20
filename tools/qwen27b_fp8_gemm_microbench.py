#!/usr/bin/env python3
"""Qwen3.6-27B dense FP8 GEMM timing microbench.

Measures the custom V100 FP8 W8A16 kernels on representative TP=4 decode
shapes, then compares them with FP16 F.linear and with the Python wrapper
dispatch used by the vLLM monkey-patch.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from fp8_w8a16_sm70.ext_loader import load_kernel


@dataclass(frozen=True)
class Shape:
    name: str
    m: int
    n: int
    k: int
    block_h: int = 128
    block_w: int = 128


SHAPES = [
    Shape("attn_q_or_o", 1, 5120, 5120),
    Shape("gdn_in_qkvz", 1, 10240, 5120),
    Shape("mlp_gate_up", 1, 8704, 5120),
    Shape("mlp_down", 1, 5120, 4352),
    Shape("decode_m2_attn", 2, 5120, 5120),
    Shape("decode_m4_attn", 4, 5120, 5120),
    Shape("decode_m8_attn", 8, 5120, 5120),
    Shape("prefill_m64_attn", 64, 5120, 5120),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--iters-decode", type=int, default=int(os.getenv("ITERS_DECODE", "400")))
    p.add_argument("--iters-prefill", type=int, default=int(os.getenv("ITERS_PREFILL", "80")))
    p.add_argument("--warmup", type=int, default=int(os.getenv("WARMUP", "20")))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


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
    # Keep y live until after synchronize.
    if y.numel() == 0:
        raise RuntimeError("unexpected empty output")
    return start.elapsed_time(end) / iters


def make_case(shape: Shape, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape.m, shape.k, generator=g, device="cuda", dtype=torch.float16) * 0.1
    w_ref = torch.randn(shape.n, shape.k, generator=g, device="cuda", dtype=torch.float16) * 0.1
    scales = (
        torch.rand(
            math.ceil(shape.n / shape.block_h),
            math.ceil(shape.k / shape.block_w),
            generator=g,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.02
        + 0.01
    )
    expanded = scales.repeat_interleave(shape.block_h, 0)[: shape.n]
    expanded = expanded.repeat_interleave(shape.block_w, 1)[:, : shape.k]
    w_q = (w_ref / expanded).to(torch.float8_e4m3fn).contiguous()
    w_dq = (w_q.to(torch.float16) * expanded).contiguous()
    w_u8 = w_q.view(torch.uint8).reshape(-1).contiguous()
    s_flat = scales.reshape(-1).contiguous()
    return x.contiguous(), w_q, w_u8, s_flat, w_dq


def maybe(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> str:
    try:
        return f"{time_cuda(fn, warmup, iters):8.3f}"
    except Exception as exc:
        return f"{type(exc).__name__[:8]:>8}"


def maxerr(fn_a: Callable[[], torch.Tensor], fn_b: Callable[[], torch.Tensor]) -> str:
    try:
        a = fn_a()
        b = fn_b()
        torch.cuda.synchronize()
        return f"{(a.float() - b.float()).abs().max().item():8.3g}"
    except Exception as exc:
        return f"{type(exc).__name__[:8]:>8}"


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    ext = load_kernel(name="fp8_dequant_ext_qwen27b_microbench")

    # Import after direct extension load so direct timings stay independent from
    # the wrapper's torch._dynamo.disable decoration.
    from fp8_w8a16_sm70.vllm_serve import _v100_fp8_gemm

    print("Qwen3.6-27B FP8 GEMM microbench on", torch.cuda.get_device_name(0))
    print(f"warmup={args.warmup} iters_decode={args.iters_decode} iters_prefill={args.iters_prefill}")
    print("times are ms/call; direct kernels allocate outputs like the real path\n")
    print(
        f"{'shape':18s} {'M':>3s} {'N':>6s} {'K':>6s} "
        f"{'cuBLAS':>8s} {'A1':>8s} {'A2':>8s} {'A3k8':>8s} {'A3k4':>8s} {'A3k2':>8s} "
        f"{'coal':>8s} {'coal_m':>8s} {'coal_h2':>8s} {'h2_err':>8s} "
        f"{'coal_vq':>8s} {'vq_err':>8s} "
        f"{'sk2':>8s} {'sk4':>8s} {'sk8':>8s} {'sk8_err':>8s} "
        f"{'wrapper':>8s} {'wrap_var':>14s}"
    )
    print("-" * 160)

    for idx, shape in enumerate(SHAPES):
        iters = args.iters_prefill if shape.m >= 64 else args.iters_decode
        x, w_q, w_u8, scales, w_dq = make_case(shape, args.seed + idx)
        n, k, bh, bw = shape.n, shape.k, shape.block_h, shape.block_w

        cublas = maybe(lambda: F.linear(x, w_dq), args.warmup, iters)
        a1 = maybe(lambda: ext.fp8_w8a16_gemm_a1(x, w_u8, scales, n, k, bh, bw), args.warmup, iters)
        a2 = maybe(lambda: ext.fp8_w8a16_gemm_a2(x, w_u8, scales, n, k, bh, bw), args.warmup, iters)
        a3k8 = maybe(lambda: ext.fp8_w8a16_gemm_a3(x, w_u8, scales, n, k, bh, bw, 8), args.warmup, iters)
        a3k4 = maybe(lambda: ext.fp8_w8a16_gemm_a3(x, w_u8, scales, n, k, bh, bw, 4), args.warmup, iters)
        a3k2 = maybe(lambda: ext.fp8_w8a16_gemm_a3(x, w_u8, scales, n, k, bh, bw, 2), args.warmup, iters)
        coal = maybe(lambda: ext.fp8_w8a16_gemv_coalesced(x, w_u8, scales, n, k, bh, bw), args.warmup, iters)
        coal_m = maybe(lambda: ext.fp8_w8a16_gemv_coalesced_m(x, w_u8, scales, n, k, bh, bw), args.warmup, iters)
        coal_h2 = maybe(
            lambda: ext.fp8_w8a16_gemv_coalesced_m_half2(x, w_u8, scales, n, k, bh, bw),
            args.warmup,
            iters,
        )
        h2_err = maxerr(
            lambda: ext.fp8_w8a16_gemv_coalesced_m(x, w_u8, scales, n, k, bh, bw),
            lambda: ext.fp8_w8a16_gemv_coalesced_m_half2(x, w_u8, scales, n, k, bh, bw),
        )
        coal_vq = maybe(
            lambda: ext.fp8_w8a16_gemv_coalesced_m_vecdq(x, w_u8, scales, n, k, bh, bw),
            args.warmup,
            iters,
        )
        vq_err = maxerr(
            lambda: ext.fp8_w8a16_gemv_coalesced_m(x, w_u8, scales, n, k, bh, bw),
            lambda: ext.fp8_w8a16_gemv_coalesced_m_vecdq(x, w_u8, scales, n, k, bh, bw),
        )
        coal_sk2 = maybe(
            lambda: ext.fp8_w8a16_gemv_coalesced_m_splitk(x, w_u8, scales, n, k, bh, bw, 2),
            args.warmup,
            iters,
        )
        coal_sk4 = maybe(
            lambda: ext.fp8_w8a16_gemv_coalesced_m_splitk(x, w_u8, scales, n, k, bh, bw, 4),
            args.warmup,
            iters,
        )
        coal_sk8 = maybe(
            lambda: ext.fp8_w8a16_gemv_coalesced_m_splitk(x, w_u8, scales, n, k, bh, bw, 8),
            args.warmup,
            iters,
        )
        sk8_err = maxerr(
            lambda: ext.fp8_w8a16_gemv_coalesced_m(x, w_u8, scales, n, k, bh, bw),
            lambda: ext.fp8_w8a16_gemv_coalesced_m_splitk(x, w_u8, scales, n, k, bh, bw, 8),
        )

        variant_holder = {"v": ""}

        def wrapped():
            out, variant = _v100_fp8_gemm(x, w_q, scales, n, k, bh, bw)
            variant_holder["v"] = variant
            return out

        wrapper = maybe(wrapped, args.warmup, iters)
        print(
            f"{shape.name:18s} {shape.m:3d} {n:6d} {k:6d} "
            f"{cublas:>8s} {a1:>8s} {a2:>8s} {a3k8:>8s} {a3k4:>8s} {a3k2:>8s} "
            f"{coal:>8s} {coal_m:>8s} {coal_h2:>8s} {h2_err:>8s} "
            f"{coal_vq:>8s} {vq_err:>8s} "
            f"{coal_sk2:>8s} {coal_sk4:>8s} {coal_sk8:>8s} {sk8_err:>8s} "
            f"{wrapper:>8s} {variant_holder['v']:>14s}"
        )


if __name__ == "__main__":
    main()
