#!/usr/bin/env python3
"""
Stage-C grouped-MoE A/B :: OURS side (coalesced FP8 W8A16 grouped GEMV, CUDA cores).

Our production path consumes per-row expert ids directly on natural [E,N,K] FP8 weights
-- NO pack, NO permute, NO unpermute. So the component map differs from 1catai:
  [1] prepare/repack  : ~0 (weights resident as [E,N,K] uint8) -- not timed
  [2] route materialize: coalesced = 0 (kernel takes eids). tiled = argsort+bincount+offsets
  [3] kernel-only     : grouped GEMV(w13) + silu_and_mul + grouped GEMV(w2)
  [4] scatter/combine : 0 for coalesced top_k=1 (output already per-row in token order)
  [5] end-to-end      : == kernel (coalesced) since route/combine ~ 0

Also times our sorted `tiled` GEMM kernel-only for an apples-to-apples kernel-algorithm
comparison vs 1catai's sorted-grouped kernel (route/unsort attributed separately).

cos-gated FIRST vs the shared fp32 reference. Run in vllm-v100:vllm021-cu126 with
PYTHONPATH=/work/src (JIT builds the ext). Writes results_moe_ours_<model>.csv.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import torch
from fp8_w8a16_sm70.ext_loader import load_kernel


def timed(fn, n_warmup=10, n_iter=30):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / n_iter


def cossim(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def silu_mul(inter, gate_up):
    if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "silu_and_mul"):
        torch.ops._C.silu_and_mul(inter, gate_up)
    else:
        I = inter.shape[1]
        inter.copy_(torch.nn.functional.silu(gate_up[:, :I]) * gate_up[:, I:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda:0"
    inp = Path(args.inp)
    meta = json.loads((inp / "meta.json").read_text())
    model = meta["model"]
    E, H, I = meta["E"], meta["H"], meta["I"]
    N13, K13, N2, K2, block = meta["N13"], meta["K13"], meta["N2"], meta["K2"], meta["block"]
    W = torch.load(inp / "weights.pt")
    # resident FP8 weights as uint8 [E,N,K] + fp16 block scales [E,N/128,K/128]
    w13 = W["w13"].view(torch.uint8).to(dev).contiguous()
    w2 = W["w2"].view(torch.uint8).to(dev).contiguous()
    s13 = W["w13_scale"].to(torch.float16).to(dev).contiguous()
    s2 = W["w2_scale"].to(torch.float16).to(dev).contiguous()
    BH = block  # block-128 in N as well

    ext = load_kernel(name="fp8_dequant_ext_vllm")
    if not hasattr(ext, "fp8_w8a16_grouped_gemv_coalesced"):
        sys.exit("[FATAL] fp8_w8a16_grouped_gemv_coalesced missing from ext.")
    has_tiled = hasattr(ext, "fp8_w8a16_grouped_tiled_gemm")

    rows = []
    for cfg in sorted(inp.glob("route_*.pt")):
        R = torch.load(cfg)
        eids = R["eids"].to(dev).to(torch.int64)
        x13 = R["x13"].to(dev)
        y_ref = R["y_ref"].to(dev); tpe = R["tpe"]; regime = R["regime"]; Rn = R["R"]
        inter = torch.empty(Rn, I, dtype=torch.float16, device=dev)

        # ---- coalesced (production): no permute, no unpermute ----
        def coalesced_chain():
            gate_up = ext.fp8_w8a16_grouped_gemv_coalesced(x13, eids, w13, s13,
                                                           N13, K13, BH, block)
            silu_mul(inter, gate_up)
            return ext.fp8_w8a16_grouped_gemv_coalesced(inter, eids, w2, s2,
                                                        N2, K2, BH, block)

        out = coalesced_chain()
        cos = cossim(out, y_ref)
        gate = "ok" if cos >= 0.99 else "FAIL"
        t_kern = timed(coalesced_chain)   # == e2e (route=unperm=0)

        # ---- tiled (sorted-grouped) kernel-only, apples-to-apples vs TurboMind ----
        t_tiled = float("nan"); cos_t = float("nan"); t_sort = float("nan")
        if has_tiled:
            def build_tiled():
                order = torch.argsort(eids)
                sx = x13.index_select(0, order).contiguous()
                se = eids[order]
                counts = torch.bincount(se, minlength=E).to(torch.int32)
                row_off = torch.zeros(E, device=dev, dtype=torch.int32)
                if E > 1:
                    row_off[1:] = torch.cumsum(counts[:-1], 0)
                tiles_pe = ((counts + 7) // 8).to(torch.int32)
                tile_off = torch.zeros(E, device=dev, dtype=torch.int32)
                if E > 1:
                    tile_off[1:] = torch.cumsum(tiles_pe[:-1], 0)
                return order, sx, tile_off.contiguous(), tiles_pe.contiguous(), \
                    row_off.contiguous(), counts.contiguous()
            t_sort = timed(build_tiled)
            order, sx, tile_off, tiles_pe, row_off, counts = build_tiled()
            inter_s = torch.empty(Rn, I, dtype=torch.float16, device=dev)

            def tiled_chain():
                gu = ext.fp8_w8a16_grouped_tiled_gemm(sx, tile_off, tiles_pe, row_off,
                                                      counts, w13, s13, N13, K13, BH, block)
                silu_mul(inter_s, gu)
                return ext.fp8_w8a16_grouped_tiled_gemm(inter_s, tile_off, tiles_pe, row_off,
                                                        counts, w2, s2, N2, K2, BH, block)
            out_s = tiled_chain()
            # unsort to row order for cos
            out_u = torch.empty_like(out_s)
            out_u[order] = out_s
            cos_t = cossim(out_u, y_ref)
            t_tiled = timed(tiled_chain)

        rows.append([model, regime, tpe, Rn, f"{cos:.4f}", gate,
                     f"{t_kern:.4f}", f"{t_kern:.4f}",   # kernel==e2e for coalesced
                     f"{t_tiled:.4f}", f"{cos_t:.4f}", f"{t_sort:.4f}"])
        print(f"  {regime:7s} tpe={tpe} R={Rn:5d} cos={cos:.4f}[{gate}]  "
              f"coalesced kern=e2e={t_kern:.3f}  |  tiled kern={t_tiled:.3f} "
              f"(cos={cos_t:.4f}, sort={t_sort:.3f}) ms")

    outp = args.out or f"results_moe_ours_{model}.csv"
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "regime", "tpe", "R", "cos", "gate",
                    "coalesced_kernel_ms", "coalesced_e2e_ms",
                    "tiled_kernel_ms", "tiled_cos", "tiled_sort_ms"])
        w.writerows(rows)
    print(f"[ok] wrote {outp} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
