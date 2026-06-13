#!/usr/bin/env python3
"""Volta-aware fused-MoE tuner: feasibility-pruned, single-GPU, emits vLLM JSON.

Why this exists (2026-06-13): stock benchmarks/kernels/benchmark_moe.py --tune
brute-forces a 1920-config grid (BLOCK_M..256, stages..5) PER batch size, across
8 GPUs. On V100 most of that grid is shared-memory-infeasible: the tuner discovers
this the expensive way (compile -> catch triton OutOfResources -> skip), and the
big-tile compiles cost 100-240 s EACH, so a full run is multi-day and monopolizes
the box. The manual sweep (tools/moe_decode_tile_sweep.py) already proved the
winners are tiny tiles (16/32/64 at M<=4, 16/128/64 at M>=8) and that cost is
monotone along BLOCK_M / BLOCK_K / num_stages.

This tuner does the rigorous version of "where's the wall":
  1. FEASIBILITY PREFILTER (closed-form, no compile): Triton pipelined matmul needs
     ~ num_stages*(BM*BK + BK*BN)*2 bytes of smem (fp16). Volta = 96 KB/SM. Drop
     every config over the budget a-priori -> ~1920 shrinks to a few hundred, and
     the 240 s monster-compiles never happen.
  2. Reduced, monotone-aware ranges (no BLOCK_M>64, no BLOCK_K>128, stages {2,3}) —
     the proven-useful corner; still enumerates the interacting small axes (N,
     warps, GROUP_M) rather than assuming them.
  3. Times the survivors on ONE gpu, picks the min per batch size, writes the
     canonical vLLM file E={E},N={N},device_name=...json (same loader path as the
     317 in-tree configs).

Oracle: the manual sweep got 632 us @ M=1 (q35b shape); this must meet or beat it.

Run (needs a FREE gpu — do NOT co-run with the 8-gpu benchmark_moe autotune):
  docker run --rm --gpus '"device=4"' -v $PWD:/work -w /work -e PYTHONPATH=/work/src \
    -e TUNE_E=256 -e TUNE_TOPK=8 -e TUNE_K=2048 -e TUNE_NSHARD=128 \
    vllm-v100:vllm021-cu126 python3 tools/moe_volta_tune.py /work/results/moe_volta_tune
  # gemma 26B-A4B TP4: TUNE_E=128 TUNE_TOPK=8 TUNE_K=2816 TUNE_NSHARD=176
"""

import itertools
import json
import os
import sys

import torch

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/work/results/moe_volta_tune"
os.makedirs(OUTDIR, exist_ok=True)

E = int(os.environ.get("TUNE_E", 256))
TOPK = int(os.environ.get("TUNE_TOPK", 8))
K = int(os.environ.get("TUNE_K", 2048))
NSHARD = int(os.environ.get("TUNE_NSHARD", 128))
# decode-through-light-prefill batch sizes; extend via TUNE_M="1,2,...".
MS = [int(m) for m in os.environ.get("TUNE_M", "1,2,4,8,16,24,32,48,64").split(",")]
SMEM_BUDGET = int(os.environ.get("TUNE_SMEM_KB", "96")) * 1024
SMEM_FILL = float(os.environ.get("TUNE_SMEM_FILL", "0.95"))  # headroom vs hard 96KB

LOG = open(os.path.join(OUTDIR, "tune.log"), "w")


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


# ── monotone-aware reduced ranges (the proven-useful corner) ──────────────────
BLOCK_M_R = [16, 32, 64]
BLOCK_N_R = [32, 64, 128, 256]
BLOCK_K_R = [64, 128]
GROUP_M_R = [1, 16, 32]
WARPS_R = [4, 8]
STAGES_R = [2, 3]


def smem_bytes(bm, bn, bk, stages):
    # fp16 pipelined matmul A[BM,BK] + B[BK,BN] per stage, 2 bytes/elem
    return stages * (bm * bk + bk * bn) * 2


_AXES = [("BLOCK_SIZE_M", BLOCK_M_R), ("BLOCK_SIZE_N", BLOCK_N_R),
         ("BLOCK_SIZE_K", BLOCK_K_R), ("GROUP_SIZE_M", GROUP_M_R),
         ("num_warps", WARPS_R), ("num_stages", STAGES_R)]


def shell_radius(cfg):
    """L1 distance from the inner/low corner = sum of per-axis range-indices.
    Pinning a proven axis to a singleton range makes it contribute 0 — so the
    radius is measured only over the still-free (soft-slope) axes."""
    return sum(rng.index(cfg[name]) for name, rng in _AXES)


def feasible_configs():
    """Closed-form smem prefilter — the a-priori 'hard wall'. No compilation.
    Returns configs sorted by shell radius (inner corner outward)."""
    out, dropped = [], 0
    for bm, bn, bk, gm, w, st in itertools.product(
        BLOCK_M_R, BLOCK_N_R, BLOCK_K_R, GROUP_M_R, WARPS_R, STAGES_R
    ):
        if smem_bytes(bm, bn, bk, st) <= SMEM_BUDGET * SMEM_FILL:
            out.append(dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                            GROUP_SIZE_M=gm, num_warps=w, num_stages=st, SPLIT_K=1))
        else:
            dropped += 1
    out.sort(key=shell_radius)
    return out, dropped


def cuda_time(fn, iters=30, warmup=8):
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
    log(f"shape: E={E} topk={TOPK} K={K} Nshard={NSHARD}  Ms={MS}")

    import tempfile
    from vllm.model_executor.layers.fused_moe import fused_moe as fm

    grid, dropped = feasible_configs()
    total = dropped + len(grid)
    log(f"search space: {total} configs -> {len(grid)} smem-feasible "
        f"(<= {int(SMEM_BUDGET*SMEM_FILL/1024)}KB), {dropped} pruned a-priori "
        f"(no compile)")

    N13 = 2 * NSHARD
    torch.manual_seed(0)
    w13 = torch.randn(E, N13, K, dtype=torch.float16, device=dev) * 0.02
    w2 = torch.randn(E, K, NSHARD, dtype=torch.float16, device=dev) * 0.02

    cfgdir = tempfile.mkdtemp()
    fname = f"E={E},N={NSHARD},device_name=Tesla_V100-SXM2-32GB.json"
    best_per_m = {}

    for M in MS:
        x = torch.randn(M, K, dtype=torch.float16, device=dev)
        ids = torch.stack(
            [torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]
        ).to(torch.int32)
        w = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)

        # Walk feasible configs in shell order (grid is pre-sorted by radius).
        # Time ALL of them = the oracle. Simultaneously simulate the monotone
        # early-stop: track per-shell best; once STOP_GRACE+1 consecutive shells
        # all regress past the running best by >EPS, the shell-walk WOULD halt.
        # Record that trigger to prove (offline) the early-stop is lossless here.
        EPS = float(os.environ.get("TUNE_STOP_EPS", "0.02"))
        GRACE = int(os.environ.get("TUNE_STOP_GRACE", "1"))
        best_ms, best_cfg, ran, oor = float("inf"), None, 0, 0
        shell_best, regress = {}, 0
        stop_radius, stop_best, stop_ran = None, None, None
        cur_r = -1
        for cfg in grid:
            r = shell_radius(cfg)
            if r != cur_r and cur_r >= 0 and stop_radius is None:
                # close out the shell we just finished
                sb = shell_best.get(cur_r, float("inf"))
                regress = regress + 1 if sb > best_ms * (1 + EPS) else 0
                if regress > GRACE:
                    stop_radius, stop_best, stop_ran = cur_r, best_ms, ran
            cur_r = r
            json.dump({str(M): cfg}, open(os.path.join(cfgdir, fname), "w"))
            os.environ["VLLM_TUNED_CONFIG_FOLDER"] = cfgdir
            fm.get_moe_configs.cache_clear()
            try:
                ms = cuda_time(lambda: fm.fused_experts(
                    x, w13, w2, w, ids, inplace=False, global_num_experts=E))
                ran += 1
                shell_best[r] = min(shell_best.get(r, float("inf")), ms)
                if ms < best_ms:
                    best_ms, best_cfg = ms, cfg
            except Exception as ex:
                # smem prefilter should prevent OutOfResources; count any leakage
                oor += 1
                if oor <= 3:
                    log(f"  M={M} skip {cfg}: {type(ex).__name__}")
        bc = {k: best_cfg[k] for k in
              ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
               "GROUP_SIZE_M", "num_warps", "num_stages")}
        best_per_m[str(M)] = bc
        log(f"M={M:<3d} best {best_ms*1e3:7.1f} us  {bc}  (timed {ran}, skip {oor})")
        if stop_radius is not None:
            verdict = "LOSSLESS" if abs(stop_best - best_ms) <= best_ms * EPS else \
                      f"MISSED (early {stop_best*1e3:.1f} vs true {best_ms*1e3:.1f})"
            log(f"     shell early-stop: would halt after radius {stop_radius} "
                f"= {stop_ran}/{ran} configs ({100*stop_ran/ran:.0f}%); {verdict}")
        else:
            log(f"     shell early-stop: never triggered (best lives in outer shell)")

    payload = {"triton_version": __import__("triton").__version__, **best_per_m}
    outjson = os.path.join(OUTDIR, fname)
    json.dump(payload, open(outjson, "w"), indent=4)
    log(f"\nwrote {outjson}")
    log("compare M=1 vs manual-sweep oracle (632 us q35b / 620 us g26b): "
        f"{best_per_m['1']}")
    log("DONE")


if __name__ == "__main__":
    main()
