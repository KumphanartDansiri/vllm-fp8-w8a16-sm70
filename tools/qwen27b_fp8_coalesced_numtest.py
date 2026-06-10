#!/usr/bin/env python3
"""Correctness check for the prototype coalesced FP8 GEMV kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fp8_w8a16_sm70.ext_loader import load_kernel


@dataclass(frozen=True)
class Shape:
    name: str
    n: int
    k: int
    block_h: int = 128
    block_w: int = 128


SHAPES = [
    Shape("attn_q_or_o", 5120, 5120),
    Shape("gdn_in_qkvz", 10240, 5120),
    Shape("mlp_gate_up", 8704, 5120),
    Shape("mlp_down", 5120, 4352),
    Shape("attn_channel", 5120, 5120, block_h=1),
    Shape("gdn_in_channel", 10240, 5120, block_h=1),
    Shape("mlp_gate_channel", 8704, 5120, block_h=1),
    Shape("mlp_down_channel", 5120, 4352, block_h=1),
]


def make_case(shape: Shape, seed: int, m: int = 1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, shape.k, generator=g, device="cuda", dtype=torch.float16) * 0.1
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
    return x.contiguous(), w_q.view(torch.uint8).reshape(-1).contiguous(), scales.reshape(-1).contiguous(), w_dq


def metrics(a: torch.Tensor, b: torch.Tensor):
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    return {
        "cos": float(F.cosine_similarity(af, bf, dim=0)),
        "max_abs": float((af - bf).abs().max()),
        "mean_abs": float((af - bf).abs().mean()),
    }


def main() -> None:
    ext = load_kernel(name="fp8_dequant_ext_qwen27b_coalesced_numtest")
    torch.backends.cuda.matmul.allow_tf32 = False
    ok = True

    print("Qwen3.6 FP8 coalesced GEMV numtest")
    for idx, shape in enumerate(SHAPES):
        for m in (1, 2, 8):
            x, w_u8, scales, w_dq = make_case(shape, idx + 100 * m, m)
            ref = F.linear(x, w_dq)
            if shape.k % (8 * shape.block_w) == 0:
                old = ext.fp8_w8a16_gemm_a3(
                    x, w_u8, scales, shape.n, shape.k, shape.block_h, shape.block_w, 8)
            elif shape.k % (2 * shape.block_w) == 0:
                old = ext.fp8_w8a16_gemm_a3(
                    x, w_u8, scales, shape.n, shape.k, shape.block_h, shape.block_w, 2)
            else:
                old = ext.fp8_w8a16_gemm_a1(
                    x, w_u8, scales, shape.n, shape.k, shape.block_h, shape.block_w)

            coal = ext.fp8_w8a16_gemv_coalesced_m(
                x, w_u8, scales, shape.n, shape.k, shape.block_h, shape.block_w)
            if m == 1:
                coal_1 = ext.fp8_w8a16_gemv_coalesced(
                    x, w_u8, scales, shape.n, shape.k, shape.block_h, shape.block_w)
                m_pair = metrics(coal, coal_1)
                ok = ok and m_pair["cos"] > 0.9999

            m_ref = metrics(coal, ref)
            m_old = metrics(coal, old)
            print(
                f"{shape.name:18s} M={m:<2d} coal_m-vs-ref cos={m_ref['cos']:.8f} "
                f"max_abs={m_ref['max_abs']:.4e} mean_abs={m_ref['mean_abs']:.4e} | "
                f"coal_m-vs-old cos={m_old['cos']:.8f} max_abs={m_old['max_abs']:.4e}"
            )
            ok = ok and m_ref["cos"] > 0.9999 and m_old["cos"] > 0.9999

    if not ok:
        raise SystemExit("coalesced GEMV numtest failed")


if __name__ == "__main__":
    main()
