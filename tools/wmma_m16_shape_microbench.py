#!/usr/bin/env python3
"""Microbench the decode-oriented WMMA-M16 FP8 W8A16 kernel.

Targets the real coal_m-retile shapes:
  - GLM-Air attention (block_h=1)
  - Qwen3.5-122B-A10B attention at TP=8 (block_h=128)
  - Gemma-4-31B diagnostic shape (block_h=1)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from fp8_w8a16_sm70.ext_loader import load_kernel


MS = (1, 2, 4, 8)


@dataclass(frozen=True)
class Shape:
    family: str
    name: str
    n: int
    k: int
    block_h: int


SHAPES = (
    Shape("glm-air", "qkv", 1792, 4096, 1),
    Shape("glm-air", "o_proj", 4096, 1536, 1),
    Shape("qwen122b", "qkv_tp8", 1152, 3072, 128),
    Shape("qwen122b", "o_proj_tp8", 3072, 1024, 128),
    Shape("gemma31b", "diagnostic", 5120, 5120, 1),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--family", choices=("all", "glm-air", "qwen122b", "gemma31b"), default="all")
    p.add_argument(
        "--splits",
        default="4,8,12",
        help="Comma-separated split-K factors for the WMMA-M16 split-K prototype.",
    )
    return p.parse_args()


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    y = None
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
    if y is None or y.numel() == 0:
        raise RuntimeError("unexpected empty output")
    return start.elapsed_time(end) / iters, y


def make_case(shape: Shape, m: int, seed: int):
    block_w = 128
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, shape.k, generator=g, device="cuda", dtype=torch.float16) * 0.1
    w_ref = torch.randn(shape.n, shape.k, generator=g, device="cuda", dtype=torch.float16) * 0.1
    scales = (
        torch.rand(
            math.ceil(shape.n / shape.block_h),
            math.ceil(shape.k / block_w),
            generator=g,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.02
        + 0.01
    )
    expanded = scales.repeat_interleave(shape.block_h, 0)[: shape.n]
    expanded = expanded.repeat_interleave(block_w, 1)[:, : shape.k]
    w_q = (w_ref / expanded).to(torch.float8_e4m3fn).contiguous()
    w_dq = (w_q.to(torch.float16) * expanded).contiguous()
    return (
        x.contiguous(),
        w_q.view(torch.uint8).reshape(-1).contiguous(),
        scales.reshape(-1).contiguous(),
        w_dq,
    )


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    ext = load_kernel(name="fp8_dequant_ext_wmma_m16_shape_microbench")
    shapes = [s for s in SHAPES if args.family == "all" or s.family == args.family]
    splits = tuple(int(s) for s in args.splits.split(",") if s)

    print("WMMA-M16 FP8 W8A16 decode shape microbench")
    print(f"iters={args.iters} warmup={args.warmup}")
    print(f"splits={splits}")
    print()
    print(
        f"{'family':>9s} {'shape':>10s} {'M':>2s} {'N':>5s} {'K':>5s} {'bh':>3s} "
        f"{'cuBLAS':>8s} {'coal_m':>8s} {'wmma16':>8s} {'best_sk':>8s} "
        f"{'sk':>3s} {'sk/cu':>8s} {'sk/coal':>8s} {'maxerr':>10s}"
    )
    print("-" * 104)

    for si, shape in enumerate(shapes):
        for m in MS:
            x, w_u8, scales, w_dq = make_case(shape, m, args.seed + si * 1000 + m)
            cublas_ms, y_ref = time_cuda(lambda: F.linear(x, w_dq), args.warmup, args.iters)
            coal_ms, _ = time_cuda(
                lambda: ext.fp8_w8a16_gemv_coalesced_m(
                    x, w_u8, scales, shape.n, shape.k, shape.block_h, 128),
                args.warmup,
                args.iters,
            )
            wmma_ms, y_wmma = time_cuda(
                lambda: ext.fp8_w8a16_gemm_wmma_m16(
                    x, w_u8, scales, shape.n, shape.k, shape.block_h, 128),
                args.warmup,
                args.iters,
            )
            best_split = None
            best_split_ms = float("inf")
            best_split_y = None
            for split in splits:
                split_ms, y_split = time_cuda(
                    lambda split=split: ext.fp8_w8a16_gemm_wmma_m16_splitk(
                        x, w_u8, scales, shape.n, shape.k, shape.block_h, 128, split),
                    args.warmup,
                    args.iters,
                )
                if split_ms < best_split_ms:
                    best_split = split
                    best_split_ms = split_ms
                    best_split_y = y_split
            maxerr = (best_split_y.float() - y_ref.float()).abs().max().item()
            print(
                f"{shape.family:>9s} {shape.name:>10s} {m:2d} {shape.n:5d} {shape.k:5d} {shape.block_h:3d} "
                f"{cublas_ms:8.3f} {coal_ms:8.3f} {wmma_ms:8.3f} {best_split_ms:8.3f} "
                f"{best_split:3d} {best_split_ms / cublas_ms:8.3f} "
                f"{best_split_ms / coal_ms:8.3f} {maxerr:10.3e}"
            )


if __name__ == "__main__":
    main()
