#!/usr/bin/env python3
"""
Minimal repro — 1catai/1Cat-vLLM SM70 FP8 GEMM: `fp8_gemm_sm70_out_auto` returns garbage.

WHAT THIS SHOWS
  The TurboMind s884 FP8 W8A16 SM70 GEMM has three entry points:
    * fp8_gemm_sm70_out(out, x, tm_w, tm_s, group_size, k_ld, q_ld)  -- explicit ld
    * fp8_gemm_sm70_out_meta(out, x, tm_w, tm_s, meta)               -- ld read from meta
    * fp8_gemm_sm70_out_auto(out, x, tm_w, tm_s)                     -- ld reconstructed
  Against the SAME prepared weights, the first two produce cos ~= 1.0 vs an FP32
  dequant reference; `_auto` produces cos ~= 0 (garbage) at every M.

ROOT CAUSE (source read — csrc/quantization/awq/awq_sm70_gemm.cu)
  `fp8_sm70_prepare()` packs the weight/scales via TurboMind `LayoutConverter::Convert`,
  whose destination descriptor is passed by NON-CONST reference (convert.h:15-19).
  Convert OVERWRITES Ddesc.ld with the *packed/swizzled* leading dimension
  (convert_v3.cu:48: `Ddesc.ld = mk2cs<order>(Packing_v2<pack,order>::apply({rows,cols})).x`).
  prepare stores that packed ld in meta = {k_desc.ld, q_desc.ld} (awq_sm70_gemm.cu:887-889).
  The fixed path and the _meta path feed those packed ld values back into the GEMM
  descriptors (desc_B.ld=k_ld @1149, desc_V.ld=q_ld @1173).
  `fp8_gemm_sm70_out_auto` (awq_sm70_gemm.cu:1215-1279) rebuilds desc_B/desc_V from
  geometry ONLY -- it never runs Convert -- so desc_B.ld / desc_V.ld keep their nominal
  (unpacked) values from the MatrixLayout constructor, and it passes THOSE as k_ld/q_ld
  (line 1277-1278). The GEMM then strides through the packed buffer with the wrong
  leading dimension -> garbage. The packed ld is NOT recoverable from (n,k,order,pack)
  without actually running the converter, which _auto does not do.

Run inside the 1catai SM70 FP8 image (torch.ops._C.fp8_sm70_prepare must exist).
Self-contained: synthesizes its own FP8 block-128 weight + activations. No inputs needed.
"""
import argparse, sys
import torch
from vllm import _custom_ops as ops


def cossim(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def dequant_ref(W_fp8, scales, block):
    """Neutral FP8->fp32 * per-block scale (uses neither kernel)."""
    N, K = W_fp8.shape
    s = scales.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return W_fp8.float() * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256, help="out features (mult of 128)")
    ap.add_argument("--K", type=int, default=256, help="in features (mult of 128)")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--Ms", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not hasattr(torch.ops._C, "fp8_sm70_prepare"):
        sys.exit("[FATAL] torch.ops._C.fp8_sm70_prepare missing — not a 1catai SM70 FP8 build.")
    dev = "cuda:0"
    N, K, b = args.N, args.K, args.block
    assert N % b == 0 and K % b == 0, "N,K must be multiples of block"

    g = torch.Generator().manual_seed(args.seed)
    w = (torch.randn(N, K, generator=g) * 0.2).clamp(-6, 6)
    W_fp8 = w.to(torch.float8_e4m3fn).to(dev)                       # [N,K]
    scales = ((torch.rand(N // b, K // b, generator=g) * 0.5 + 0.5)
              .float().to(dev))                                     # [N/b, K/b]
    W_dq = dequant_ref(W_fp8, scales, b)                            # fp32 [N,K]

    prep = ops.fp8_sm70_prepare(W_fp8, scales, b)
    tm_w, tm_s, meta = prep[0], prep[1], prep[2]
    k_ld, q_ld = int(meta[0].item()), int(meta[1].item())

    print(f"# shape  N={N} K={K} block={b}")
    print(f"# meta (PACKED ld from prepare/Convert):  k_ld={k_ld}  q_ld={q_ld}")
    print(f"# nominal (unpacked) geometry the _auto path reconstructs from: k={K} n={N}")
    print(f"# -> packed k_ld {'!=' if k_ld != K else '=='} nominal k({K}); "
          f"packed q_ld {'!=' if q_ld != N else '=='} nominal n({N})")
    has_auto = hasattr(ops, "fp8_gemm_sm70_out_auto")
    has_meta = hasattr(ops, "fp8_gemm_sm70_out_meta")
    print(f"# entry points present: fixed=1 meta={int(has_meta)} auto={int(has_auto)}")
    print()
    print(f"{'M':>4} {'path':<14} {'cos':>10} {'max_abs':>12}")

    fail = False
    for M in args.Ms:
        gg = torch.Generator().manual_seed(args.seed + M)
        A = (torch.randn(M, K, generator=gg) * 0.1).to(torch.float16).to(dev)
        ref = (A.float() @ W_dq.T)                                  # [M,N] fp32
        out = torch.empty(M, N, dtype=torch.float16, device=dev)

        def report(tag):
            c = cossim(out, ref)
            e = (out.float() - ref).abs().max().item()
            print(f"{M:>4} {tag:<14} {c:>10.4f} {e:>12.3e}")
            return c

        ops.fp8_gemm_sm70_out(out, A, tm_w, tm_s, b, k_ld, q_ld)
        c_fixed = report("fixed")
        if has_meta:
            ops.fp8_gemm_sm70_out_meta(out, A, tm_w, tm_s, meta)
            report("meta")
        if has_auto:
            ops.fp8_gemm_sm70_out_auto(out, A, tm_w, tm_s)
            c_auto = report("auto")
            if c_fixed > 0.99 and c_auto < 0.5:
                fail = True
        print()

    if fail:
        print("[REPRO CONFIRMED] fixed/meta path cos~=1.0 but _auto path cos<0.5 (garbage).")
    else:
        print("[NO REPRO] _auto did not diverge — check build / shapes.")


if __name__ == "__main__":
    main()
