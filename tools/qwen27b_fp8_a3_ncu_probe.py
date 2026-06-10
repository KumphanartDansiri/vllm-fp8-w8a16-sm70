#!/usr/bin/env python3
"""Tiny Nsight Compute target for Qwen3.6 FP8 decode GEMV.

The script creates one representative Qwen3.6 dense-linear shape, warms it up,
then brackets only the selected operation with cudaProfilerStart/Stop. Run it
with ncu --profile-from-start off to keep profiling focused on the kernel under
test rather than extension import/JIT setup.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

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


SHAPES = {
    "attn": Shape("attn", 1, 5120, 5120),
    "gdn_in": Shape("gdn_in", 1, 10240, 5120),
    "mlp_gate": Shape("mlp_gate", 1, 8704, 5120),
    "mlp_down": Shape("mlp_down", 1, 5120, 4352),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=sorted(SHAPES), default="attn")
    p.add_argument("--op", choices=("a3", "a1", "coalesced", "cublas"), default="a3")
    p.add_argument("--k-split", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


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
    return (
        x.contiguous(),
        w_q.view(torch.uint8).reshape(-1).contiguous(),
        scales.reshape(-1).contiguous(),
        w_dq.contiguous(),
    )


def main() -> None:
    args = parse_args()
    shape = SHAPES[args.shape]
    torch.backends.cuda.matmul.allow_tf32 = False
    ext = load_kernel(name=os.getenv("EXT_NAME", "fp8_dequant_ext_qwen27b_microbench"))
    x, w_u8, scales, w_dq = make_case(shape, args.seed)

    if args.op == "a3" and shape.k % (args.k_split * shape.block_w) != 0:
        raise ValueError(
            f"{shape.name} K={shape.k} is not divisible by "
            f"k_split*block_w={args.k_split * shape.block_w}"
        )

    def run_once():
        if args.op == "a3":
            return ext.fp8_w8a16_gemm_a3(
                x, w_u8, scales, shape.n, shape.k, shape.block_h, shape.block_w,
                args.k_split)
        if args.op == "a1":
            return ext.fp8_w8a16_gemm_a1(
                x, w_u8, scales, shape.n, shape.k, shape.block_h, shape.block_w)
        if args.op == "coalesced":
            return ext.fp8_w8a16_gemv_coalesced(
                x, w_u8, scales, shape.n, shape.k, shape.block_h, shape.block_w)
        return F.linear(x, w_dq)

    for _ in range(args.warmup):
        y = run_once()
    torch.cuda.synchronize()

    print(
        f"NCU probe op={args.op} shape={shape.name} "
        f"M={shape.m} N={shape.n} K={shape.k} k_split={args.k_split} "
        f"iters={args.iters}",
        flush=True,
    )

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.iters):
        y = run_once()
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()
    print(f"done output_mean={float(y.float().mean()):.6e}", flush=True)


if __name__ == "__main__":
    main()
