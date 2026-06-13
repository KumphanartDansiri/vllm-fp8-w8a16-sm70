#!/usr/bin/env python3
"""ViT attention microbench: Torch SDPA D=72 vs FA-V100 padded D=128.

This isolates the Qwen-VL-style MMEncoderAttention question:
can ai-bond flash-attention-v100 overcome the 72->128 head-dim padding tax?

The benchmark uses non-causal self attention with varlen layout (T, H, D), which
matches vLLM's ViT flash-attention wrapper after it flattens batch/sequence.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

try:
    import flash_attn_v100_cuda
except Exception as e:  # pragma: no cover
    print(f"[vit-fa] cannot import flash_attn_v100_cuda: {e}")
    sys.exit(2)


DTYPE = torch.float16
DEVICE = "cuda"


def cuda_time_ms(fn: Callable[[], None], iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def wall_time_ms(fn: Callable[[], None], iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def pad_head_dim(x: torch.Tensor, d_pad: int) -> torch.Tensor:
    pad = d_pad - x.shape[-1]
    if pad < 0:
        raise ValueError(f"d_pad={d_pad} is smaller than D={x.shape[-1]}")
    if pad == 0:
        return x.contiguous()
    return F.pad(x, (0, pad)).contiguous()


def fa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu: torch.Tensor,
    max_s: int,
    scale: float,
) -> None:
    flash_attn_v100_cuda.varlen_fwd(
        q,
        k,
        v,
        out,
        cu,
        cu,
        None,  # seqused_k
        None,  # leftpad_k
        None,  # block_table: dense varlen, no paged cache
        None,  # alibi
        max_s,
        max_s,
        0.0,  # dropout
        scale,
        False,  # zero_tensors
        False,  # causal
        -1,
        -1,
        0.0,  # softcap
        False,  # return_softmax
        None,  # generator
        0,  # num_splits
    )


def make_cu(batch: int, seq: int) -> torch.Tensor:
    return torch.arange(
        0,
        (batch + 1) * seq,
        step=seq,
        dtype=torch.int32,
        device=DEVICE,
    )


def sdpa_out(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    # q/k/v: [B, S, H, D] -> SDPA wants [B, H, S, D].
    qb = q.permute(0, 2, 1, 3)
    kb = k.permute(0, 2, 1, 3)
    vb = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(qb, kb, vb, dropout_p=0.0, scale=scale)
    return out.permute(0, 2, 1, 3)


def tflops(batch: int, seq: int, heads: int, d: int, ms: float) -> float:
    # QK^T and PV: each matmul is 2*B*H*S*S*D FLOPs.
    flops = 4.0 * batch * heads * seq * seq * d
    return flops / (ms / 1e3) / 1e12


def bench_one(args: argparse.Namespace, seq: int) -> None:
    torch.manual_seed(args.seed + seq)
    d = args.d
    d_pad = args.d_pad
    b = args.batch
    h = args.heads
    scale = d ** -0.5

    q = torch.randn(b, seq, h, d, device=DEVICE, dtype=DTYPE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ref = sdpa_out(q, k, v, scale)

    qf = pad_head_dim(q.reshape(b * seq, h, d), d_pad)
    kf = pad_head_dim(k.reshape(b * seq, h, d), d_pad)
    vf = pad_head_dim(v.reshape(b * seq, h, d), d_pad)
    of = torch.empty_like(qf)
    cu = make_cu(b, seq)

    def fa_call() -> None:
        fa_varlen(qf, kf, vf, of, cu, seq, scale)

    ob = torch.empty(b * seq, h, d_pad, device=DEVICE, dtype=DTYPE)

    def fa_bridge_call() -> None:
        # What a minimal MMEncoderAttention bridge pays if projections still
        # produce D=72: pad Q/K/V every call, run D=128 FA, then expose [:72].
        qb = pad_head_dim(q.reshape(b * seq, h, d), d_pad)
        kb = pad_head_dim(k.reshape(b * seq, h, d), d_pad)
        vb = pad_head_dim(v.reshape(b * seq, h, d), d_pad)
        fa_varlen(qb, kb, vb, ob, cu, seq, scale)
        _ = ob[..., :d]

    fa_call()
    got = of[..., :d].reshape(b, seq, h, d)
    torch.cuda.synchronize()

    max_abs = (got - ref).abs().max().item()
    denom = ref.float().norm().clamp_min(1e-12)
    l2_rel = ((got.float() - ref.float()).norm() / denom).item()
    cos = F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()

    def sdpa_call() -> None:
        sdpa_out(q, k, v, scale)

    fa_ms = cuda_time_ms(fa_call, args.iters, args.warmup)
    fa_bridge_ms = cuda_time_ms(fa_bridge_call, args.iters, args.warmup)
    sdpa_ms = cuda_time_ms(sdpa_call, args.iters, args.warmup)
    fa_wall = wall_time_ms(fa_call, args.wall_iters, args.warmup) if args.wall_iters else 0.0
    fa_bridge_wall = (
        wall_time_ms(fa_bridge_call, args.wall_iters, args.warmup)
        if args.wall_iters
        else 0.0
    )
    sdpa_wall = (
        wall_time_ms(sdpa_call, args.wall_iters, args.warmup) if args.wall_iters else 0.0
    )

    speedup = sdpa_ms / fa_ms if fa_ms > 0 else float("nan")
    bridge_speedup = sdpa_ms / fa_bridge_ms if fa_bridge_ms > 0 else float("nan")
    fa_tf = tflops(b, seq, h, d_pad, fa_ms)
    bridge_tf = tflops(b, seq, h, d_pad, fa_bridge_ms)
    sdpa_tf = tflops(b, seq, h, d, sdpa_ms)
    print(
        "[vit-fa] "
        f"B={b:2d} S={seq:5d} H={h:2d} D={d:3d}->{d_pad:3d} "
        f"sdpa={sdpa_ms:8.3f}ms ({sdpa_tf:5.2f}TF) "
        f"fa_pad={fa_ms:8.3f}ms ({fa_tf:5.2f}TF padded) "
        f"speedup={speedup:5.2f}x"
    )
    print(
        "[vit-fa] "
        f"          bridge_pad={fa_bridge_ms:8.3f}ms ({bridge_tf:5.2f}TF padded) "
        f"speedup={bridge_speedup:5.2f}x "
        f"cos={cos:.6f} l2={l2_rel:.3e} max={max_abs:.3e}"
    )
    if args.wall_iters:
        print(
            "[vit-fa] "
            f"          wall sdpa={sdpa_wall:8.3f}ms "
            f"fa_pad={fa_wall:8.3f}ms "
            f"bridge={fa_bridge_wall:8.3f}ms "
            f"bridge_speedup={sdpa_wall / fa_bridge_wall:5.2f}x"
        )


def unsupported_probe(args: argparse.Namespace) -> None:
    q = torch.randn(1 * 64, args.heads, args.d, device=DEVICE, dtype=DTYPE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)
    cu = make_cu(1, 64)
    try:
        fa_varlen(q, k, v, out, cu, 64, args.d ** -0.5)
    except Exception as e:
        print(f"[vit-fa] raw D={args.d} probe: expected failure: {type(e).__name__}: {e}")
        return
    print(f"[vit-fa] raw D={args.d} probe: unexpectedly succeeded")


def parse_seq_sweep(raw: str) -> list[int]:
    seqs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seqs.append(int(part))
    if not seqs:
        raise ValueError("--seqs cannot be empty")
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="256,512,1024,2048,4096")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--d", type=int, default=72)
    ap.add_argument("--d-pad", type=int, default=128)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--wall-iters", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-raw-probe", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[vit-fa] CUDA is not available")
        sys.exit(2)
    if args.d_pad not in (16, 32, 64, 128, 256):
        raise ValueError("--d-pad must be one of FA-V100's compiled head dims")
    if args.d_pad < args.d:
        raise ValueError("--d-pad must be >= --d")

    print(
        "[vit-fa] device="
        f"{torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"shape B={args.batch} H={args.heads} D={args.d}->{args.d_pad} "
        f"scale=1/sqrt({args.d}) padding_tax={args.d_pad / args.d:.3f}x"
    )
    if not args.skip_raw_probe:
        unsupported_probe(args)
    for seq in parse_seq_sweep(args.seqs):
        bench_one(args, seq)


if __name__ == "__main__":
    main()
