#!/usr/bin/env python3
"""Measure standalone dequant/reconstruct cost for FP8 vs GPTQ int4.

This is a format-conversion microbench, not an end-to-end kernel comparison:
FP8 uses the extension's block-scale E4M3->FP16 path, while GPTQ uses a
probe-only int4 reconstruct kernel matching the vLLM GPTQ layout for
bits=4/group_size=128/desc_act=False.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable

import torch

from fp8_w8a16_sm70.ext_loader import load_kernel


@dataclass(frozen=True)
class Shape:
    name: str
    n: int
    k: int
    block_h: int = 128
    block_w: int = 128


SHAPES = [
    Shape("attn_5120", 5120, 5120),
    Shape("gdn_10240x5120", 10240, 5120),
    Shape("mlp_up_8704x5120", 8704, 5120),
    Shape("mlp_down_5120x4352", 5120, 4352),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=int(os.getenv("ITERS", "100")))
    p.add_argument("--warmup", type=int, default=int(os.getenv("WARMUP", "10")))
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
    if y.numel() == 0:
        raise RuntimeError("empty output")
    return start.elapsed_time(end) / iters


def make_fp8(shape: Shape, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randint(
        0, 127, (shape.n * shape.k,), generator=g, device="cuda", dtype=torch.uint8
    )
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
    ).contiguous()
    return weight.contiguous(), scales


def make_gptq(shape: Shape, seed: int, group_size: int = 128):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (shape.k // 8, shape.n),
        generator=g,
        device="cuda",
        dtype=torch.int32,
    ).contiguous()
    groups = shape.k // group_size
    qzeros = torch.randint(
        0,
        2**31 - 1,
        (groups, math.ceil(shape.n / 8)),
        generator=g,
        device="cuda",
        dtype=torch.int32,
    ).contiguous()
    scales = (
        torch.rand(
            groups, shape.n, generator=g, device="cuda", dtype=torch.float16
        )
        * 0.02
        + 0.01
    ).contiguous()
    return qweight, qzeros, scales


def main() -> None:
    args = parse_args()
    ext = load_kernel(name="fp8_dequant_ext_dequant_tax")
    group_size = 128
    print("Dequant tax microbench on", torch.cuda.get_device_name(0))
    print(f"warmup={args.warmup} iters={args.iters}")
    print("times are ms/call; GB/s counts compressed weight read + FP16 output write\n")
    print(
        f"{'shape':20s} {'N':>6s} {'K':>6s} "
        f"{'FP8 ms':>9s} {'FP8 GB/s':>10s} {'GPTQ ms':>9s} {'GPTQ GB/s':>10s} "
        f"{'GPTQ/FP8':>9s}"
    )
    print("-" * 86)
    for i, shape in enumerate(SHAPES):
        fp8_w, fp8_s = make_fp8(shape, args.seed + i)
        g_qw, g_qz, g_s = make_gptq(shape, args.seed + 100 + i, group_size)

        fp8_ms = time_cuda(
            lambda: ext.fp8_e4m3_to_fp16_block_scaled(
                fp8_w, fp8_s, shape.n, shape.k, shape.block_h, shape.block_w
            ),
            args.warmup,
            args.iters,
        )
        gptq_ms = time_cuda(
            lambda: ext.gptq4_to_fp16_dequant(
                g_qw, g_qz, g_s, shape.k, shape.n, group_size, False
            ),
            args.warmup,
            args.iters,
        )

        fp8_bytes = shape.n * shape.k * 1 + shape.n * shape.k * 2
        gptq_bytes = shape.n * shape.k * 0.5 + shape.n * shape.k * 2
        fp8_gbs = fp8_bytes / (fp8_ms * 1e-3) / 1e9
        gptq_gbs = gptq_bytes / (gptq_ms * 1e-3) / 1e9
        print(
            f"{shape.name:20s} {shape.n:6d} {shape.k:6d} "
            f"{fp8_ms:9.3f} {fp8_gbs:10.1f} {gptq_ms:9.3f} {gptq_gbs:10.1f} "
            f"{gptq_ms / fp8_ms:9.2f}"
        )


if __name__ == "__main__":
    main()
