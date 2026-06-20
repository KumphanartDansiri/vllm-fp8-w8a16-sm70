#!/usr/bin/env python3
"""NCU target: GLM-Air grouped MoE w13 decode kernels.

Brackets ONE kernel call with cudaProfilerStart/Stop. Run under
`ncu --profile-from-start off` to measure sectors/request + DRAM% — the
coalescing falsifiable bar (A.3 dense was 25.3 sectors / 17.8% DRAM)."""
import argparse

import torch
from fp8_w8a16_sm70.ext_loader import load_kernel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=("coalesced", "a3", "tiled", "mtile"), default="coalesced")
    p.add_argument("--r", type=int, default=64)
    p.add_argument("--e", type=int, default=128)
    p.add_argument("--n", type=int, default=352)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--block-h", type=int, default=1)
    p.add_argument(
        "--eids-mode",
        choices=("random", "unique", "clustered", "hot", "roundrobin"),
        default="random",
        help=(
            "Expert-id distribution. random approximates worst-case L2 reuse; "
            "clustered/hot approximate router concentration."
        ),
    )
    p.add_argument("--hot-experts", type=int, default=8)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

args = parse_args()
ext = load_kernel(name="fp8_dequant_ext_vllm")
dev = "cuda"
E, R, N13, K13, BH = args.e, args.r, args.n, args.k, args.block_h
g = torch.Generator(device=dev).manual_seed(args.seed)
if BH == 1:
    s13 = (torch.rand(E, N13, 1, generator=g, device=dev, dtype=torch.float16) * 0.02 + 0.01
           ).expand(E, N13, K13 // 128).contiguous()
else:
    s13 = (torch.rand(E, (N13 + BH - 1) // BH, K13 // 128,
                      generator=g, device=dev, dtype=torch.float16) * 0.02 + 0.01
           ).contiguous()
w13 = (torch.randn(E, N13, K13, generator=g, device=dev, dtype=torch.float16) * 0.1
       ).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
x13 = torch.randn(R, K13, generator=g, device=dev, dtype=torch.float16) * 0.1

def make_eids():
    if args.eids_mode == "random":
        return torch.randint(0, E, (R,), generator=g, device=dev, dtype=torch.int64)
    if args.eids_mode == "unique":
        return (torch.arange(R, device=dev, dtype=torch.int64) % E).contiguous()
    if args.eids_mode == "roundrobin":
        hot = max(1, min(args.hot_experts, E))
        return (torch.arange(R, device=dev, dtype=torch.int64) % hot).contiguous()
    if args.eids_mode == "clustered":
        hot = max(1, min(args.hot_experts, E))
        base = torch.arange(R, device=dev, dtype=torch.int64) * hot // max(1, R)
        return base.clamp_max(hot - 1).contiguous()
    hot = max(1, min(args.hot_experts, E))
    return torch.randint(0, hot, (R,), generator=g, device=dev, dtype=torch.int64)

eids = make_eids()
unique_eids = int(torch.unique(eids).numel())

def make_tiled_layout():
    order = torch.argsort(eids)
    sorted_eids = eids[order]
    sorted_x = x13.index_select(0, order).contiguous()
    counts = torch.bincount(sorted_eids, minlength=E).to(torch.int32).contiguous()
    expert_row_off = torch.empty(E, device=dev, dtype=torch.int32)
    expert_row_off[0] = 0
    if E > 1:
        expert_row_off[1:] = torch.cumsum(counts[:-1], 0)
    tiles_per_e = ((counts + 7) // 8).to(torch.int32).contiguous()
    expert_tile_off = torch.empty(E, device=dev, dtype=torch.int32)
    expert_tile_off[0] = 0
    if E > 1:
        expert_tile_off[1:] = torch.cumsum(tiles_per_e[:-1], 0)
    return sorted_x, expert_tile_off.contiguous(), tiles_per_e, expert_row_off.contiguous(), counts

def make_mtile_layout():
    order = torch.argsort(eids)
    sorted_eids = eids[order]
    sorted_x = x13.index_select(0, order).contiguous()
    counts = torch.bincount(sorted_eids, minlength=E).to(torch.int32).contiguous()
    expert_row_off = torch.empty(E, device=dev, dtype=torch.int32)
    expert_row_off[0] = 0
    if E > 1:
        expert_row_off[1:] = torch.cumsum(counts[:-1], 0)
    tile_expert = []
    tile_row_off = []
    tile_rows = []
    counts_cpu = counts.cpu().tolist()
    offs_cpu = expert_row_off.cpu().tolist()
    for e, count in enumerate(counts_cpu):
        for local in range(0, count, 8):
            tile_expert.append(e)
            tile_row_off.append(offs_cpu[e] + local)
            tile_rows.append(min(8, count - local))
    return (
        sorted_x,
        torch.tensor(tile_expert, device=dev, dtype=torch.int32),
        torch.tensor(tile_row_off, device=dev, dtype=torch.int32),
        torch.tensor(tile_rows, device=dev, dtype=torch.int32),
    )

if args.op == "tiled":
    tiled_args = make_tiled_layout()
elif args.op == "mtile":
    mtile_args = make_mtile_layout()

def run_once():
    if args.op == "coalesced":
        return ext.fp8_w8a16_grouped_gemv_coalesced(
            x13, eids, w13, s13, N13, K13, BH, 128)
    if args.op == "tiled":
        sorted_x, expert_tile_off, tiles_per_e, expert_row_off, counts = tiled_args
        return ext.fp8_w8a16_grouped_tiled_gemm(
            sorted_x, expert_tile_off, tiles_per_e, expert_row_off, counts,
            w13, s13, N13, K13, BH, 128)
    if args.op == "mtile":
        sorted_x, tile_expert, tile_row_off, tile_rows = mtile_args
        return ext.fp8_w8a16_grouped_gemv_coalesced_mtile(
            sorted_x, tile_expert, tile_row_off, tile_rows,
            w13, s13, N13, K13, BH, 128)
    return ext.fp8_w8a16_grouped_routed_gemm_a3(
        x13, eids, w13, s13, N13, K13, BH, 128, 8)

for _ in range(args.warmup):
    y = run_once()
torch.cuda.synchronize()
print(f"grouped_moe_ncu op={args.op} E={E} R={R} N={N13} K={K13} "
      f"block_h={BH} eids_mode={args.eids_mode} hot={args.hot_experts} "
      f"unique_eids={unique_eids}",
      flush=True)
torch.cuda.cudart().cudaProfilerStart()
y = run_once()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print(f"done mean={float(y.float().mean()):.6e}", flush=True)
