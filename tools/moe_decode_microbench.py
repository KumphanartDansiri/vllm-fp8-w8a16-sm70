#!/usr/bin/env python3
"""Kernel-level microbench of vLLM's Triton fused_experts at V100 DECODE shapes.

Context (2026-06-12): stock FP16 MoE decode on V100 is ~15.6 tok/s for Qwen3.6-35B-A3B
(TP4) vs 37-41 for the smaller dense 27B. A serve-level A/B (tools/moe_stages_ab_vllm021.sh)
REFUTED num_stages {4,3,2} as the cause — all identical. This bench answers: where do the
~50 ms/token of FP16-MoE GPU time actually go?  (FP8 grouped path on the same model =
70 tok/s = 14 ms/token, so the MoE blocks burn ~1.25 ms/layer x 40 layers extra.)

Measures, on ONE GPU, shapes = one TP4 rank of Qwen3.6-35B-A3B:
  E=256 experts, topk=8, K=hidden=2048, shard intermediate=128 -> w13 [256,256,2048],
  w2 [256,2048,128], fp16. M = decode batch in {1,2,4,8}.
  1) fused_experts() steady-state latency (CUDA events, 200 iters)
  2) torch.profiler per-kernel CUDA-time breakdown
  3) memory floor reference: bytes of active expert weights / measured HBM bandwidth
  4) dense-GEMM reference: same active params as ONE dense [2048 x 2048] GEMM slice
Run inside the vllm021 container:
  docker run --rm --gpus '"device=4"' -v /mnt/models:/mnt/models:ro -v $PWD:/work -w /work \
    -e PYTHONPATH=/work/src vllm-v100:vllm021-cu126 python3 tools/moe_decode_microbench.py
"""

import os
import sys
import time

import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/results/moe_decode_microbench.txt"


def log(msg, f=[None]):
    if f[0] is None:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        f[0] = open(OUT, "w")
    print(msg, flush=True)
    f[0].write(msg + "\n")
    f[0].flush()


def cuda_time(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def main():
    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    torch.cuda.init()
    log(f"device = {torch.cuda.get_device_name(0)}")

    # measured HBM bandwidth (large copy)
    big = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(big)
    t = cuda_time(lambda: dst.copy_(big), iters=20, warmup=5)
    bw = 2 * big.numel() / (t / 1e3) / 1e9
    log(f"copy bandwidth ~ {bw:.0f} GB/s")
    del big, dst

    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    # Qwen3.6-35B-A3B @ TP4, one rank
    E, TOPK, K, NSHARD = 256, 8, 2048, 128
    N13 = 2 * NSHARD  # gate+up
    torch.manual_seed(0)
    w13 = torch.randn(E, N13, K, dtype=torch.float16, device=dev) * 0.02
    w2 = torch.randn(E, K, NSHARD, dtype=torch.float16, device=dev) * 0.02
    log(f"shapes: E={E} topk={TOPK} K={K} Nshard={NSHARD} "
        f"w13={tuple(w13.shape)} w2={tuple(w2.shape)} fp16")

    for M in (1, 2, 4, 8):
        x = torch.randn(M, K, dtype=torch.float16, device=dev)
        # unique experts per token like the router produces
        ids = torch.stack(
            [torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]
        ).to(torch.int32)
        w = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)

        def call():
            fused_experts(x, w13, w2, w, ids, inplace=False, global_num_experts=E)

        ms = cuda_time(call)
        act_bytes = M * TOPK * (N13 * K + K * NSHARD) * 2  # touched expert tiles, fp16
        floor_ms = act_bytes / (bw * 1e9) * 1e3
        log(f"M={M}: fused_experts = {ms * 1e3:8.1f} us | active-weight bytes "
            f"{act_bytes / 1e6:6.1f} MB -> floor {floor_ms * 1e3:6.1f} us | "
            f"slowdown vs floor = {ms / floor_ms:5.1f}x")

    # per-kernel breakdown at M=1 (the serve decode case)
    M = 1
    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    ids = torch.stack(
        [torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]
    ).to(torch.int32)
    w = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)
    for _ in range(20):
        fused_experts(x, w13, w2, w, ids, inplace=False, global_num_experts=E)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(50):
            fused_experts(x, w13, w2, w, ids, inplace=False, global_num_experts=E)
        torch.cuda.synchronize()
    log("\n== top kernels by CUDA time, M=1, 50 calls ==")
    log(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # dense reference: one [1,2048]x[2048,2048] fp16 GEMM (cuBLAS) — what a dense FFN
    # slice of similar size costs on this GPU
    a = torch.randn(1, K, dtype=torch.float16, device=dev)
    b = torch.randn(K, K, dtype=torch.float16, device=dev)
    ms = cuda_time(lambda: a @ b)
    log(f"\ndense ref: [1,{K}]x[{K},{K}] fp16 cuBLAS = {ms * 1e3:.1f} us "
        f"({K * K * 2 / 1e6:.1f} MB weights)")

    log("\nDONE")


if __name__ == "__main__":
    main()
