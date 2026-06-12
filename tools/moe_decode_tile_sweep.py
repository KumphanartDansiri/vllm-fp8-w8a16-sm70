#!/usr/bin/env python3
"""Round 2: can ANY fused_moe tile config rescue V100 FP16 MoE decode?

Round 1 (moe_decode_microbench.py): fused_moe_kernel = 98.9% of fused_experts time,
645us/launch at M=1 — 90x off memory floor. num_stages e2e A/B was null. This sweeps
tile shapes in-process via VLLM_TUNED_CONFIG_FOLDER + lru_cache clear to settle whether
config tuning has ANY headroom on sm_70, or whether the kernel is structurally slow
(Triton FMA fallback on Volta). Same shapes as round 1 (Qwen3.6-35B-A3B TP4 rank).
Run: docker run --rm --gpus '"device=4"' -v $PWD:/work -w /work \
  -e PYTHONPATH=/work/src vllm-v100:vllm021-cu126 python3 tools/moe_decode_tile_sweep.py
"""

import json
import os
import sys
import tempfile

import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/results/moe_decode_tile_sweep_q35b.txt"
of = open(OUT, "w")


def log(msg):
    print(msg, flush=True)
    of.write(msg + "\n")
    of.flush()


def cuda_time(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    dev = torch.device("cuda")
    log(f"device = {torch.cuda.get_device_name(0)}")
    from vllm.model_executor.layers.fused_moe import fused_moe as fm

    # default = Qwen3.6-35B-A3B TP4 rank; override via env for other models
    # (gemma-4-26B-A4B TP4: E=128 TOPK=8 K=2816 NSHARD=176)
    E = int(os.environ.get("SWEEP_E", 256))
    TOPK = int(os.environ.get("SWEEP_TOPK", 8))
    K = int(os.environ.get("SWEEP_K", 2048))
    NSHARD = int(os.environ.get("SWEEP_NSHARD", 128))
    log(f"shapes: E={E} topk={TOPK} K={K} Nshard={NSHARD}")
    N13 = 2 * NSHARD
    torch.manual_seed(0)
    w13 = torch.randn(E, N13, K, dtype=torch.float16, device=dev) * 0.02
    w2 = torch.randn(E, K, NSHARD, dtype=torch.float16, device=dev) * 0.02
    M = 1
    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    ids = torch.stack(
        [torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]
    ).to(torch.int32)
    w = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)

    cfgdir = tempfile.mkdtemp()
    fname = f"E={E},N={NSHARD},device_name=Tesla_V100-SXM2-32GB.json"

    def run_variant(label, cfg):
        if cfg is None:
            os.environ.pop("VLLM_TUNED_CONFIG_FOLDER", None)
        else:
            json.dump({"1": cfg}, open(os.path.join(cfgdir, fname), "w"))
            os.environ["VLLM_TUNED_CONFIG_FOLDER"] = cfgdir
        fm.get_moe_configs.cache_clear()
        try:
            ms = cuda_time(lambda: fm.fused_experts(
                x, w13, w2, w, ids, inplace=False, global_num_experts=E))
            log(f"{label:36s}: {ms * 1e3:8.1f} us")
            return ms
        except Exception as ex:
            log(f"{label:36s}: FAIL {type(ex).__name__}: {str(ex)[:90]}")
            return None

    base = run_variant("default (16/64/128 w4 s4)", None)

    variants = [
        ("16/64/128 w4 s2", dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=2)),
        ("16/32/64  w4 s2 (more blocks)", dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=4, num_stages=2)),
        ("16/32/32  w4 s2", dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, GROUP_SIZE_M=1, num_warps=4, num_stages=2)),
        ("16/16/64  w2 s2 (max blocks)", dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=2, num_stages=2)),
        ("16/64/64  w8 s2 (more warps)", dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=8, num_stages=2)),
        ("16/128/64 w8 s2 (fat N)", dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=8, num_stages=2)),
        ("16/64/256 w4 s2 (deep K)", dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=4, num_stages=2)),
        ("16/32/128 w4 s3", dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=3)),
    ]
    best = (base, "default") if base else (None, None)
    for label, cfg in variants:
        ms = run_variant(label, cfg)
        if ms is not None and (best[0] is None or ms < best[0]):
            best = (ms, label)

    floor_us = 12.6e6 / (816e9) * 1e6
    if base and best[0]:
        log(f"\nbest = {best[1]} at {best[0] * 1e3:.1f} us "
            f"({base / best[0]:.2f}x vs default; floor ~{floor_us:.0f} us, "
            f"still {best[0] * 1e3 / floor_us:.0f}x off floor)")
    log("DONE")


if __name__ == "__main__":
    main()
