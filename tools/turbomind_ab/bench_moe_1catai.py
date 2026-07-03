#!/usr/bin/env python3
"""
Stage-C grouped-MoE A/B :: 1CATAI side (TurboMind s884 grouped FP8 MoE).

Runs the real serving op sequence from vllm/.../fp8_sm70_moe.py::apply, component-timed:
  [1] prepare/repack  : fp8_sm70_prepare per expert + awq_moe_build_strided_ptrs (ONE-TIME)
  [2] route materialize: moe_permute (sort/scatter -> permuted_input + expert_offsets)
  [3] kernel-only     : fp8_moe_gemm_sm70_out(w13) + silu_and_mul + fp8_moe_gemm_sm70_out(w2)
  [4] scatter/combine : moe_unpermute (gather back to token order)
  [5] end-to-end      : [2]+[3]+[4]

Routing model: per-slot (each frozen row = one token routed to one expert, top_k=1 for the
permute), so per-expert M = tpe is the timing driver. Combine-reduction arity is a
second-order effect and is symmetric across both engines (not modeled).

cos-gated FIRST vs the shared fp32 reference. Run in 1catai-vllm-v100:cu128-fp8sm70
(--entrypoint /bin/bash). Writes results_moe_1catai_<model>.csv.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import torch
from vllm import _custom_ops as ops


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="moe_inputs_<model> dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not hasattr(torch.ops._C, "fp8_moe_gemm_sm70_out"):
        sys.exit("[FATAL] fp8_moe_gemm_sm70_out missing — not a 1catai SM70 FP8 MoE build.")
    dev = "cuda:0"
    inp = Path(args.inp)
    meta = json.loads((inp / "meta.json").read_text())
    model = meta["model"]
    E, H, I = meta["E"], meta["H"], meta["I"]
    N13, K13, N2, K2, block = meta["N13"], meta["K13"], meta["N2"], meta["K2"], meta["block"]
    W = torch.load(inp / "weights.pt")
    w13 = W["w13"].to(dev); w13_s = W["w13_scale"].to(dev)
    w2 = W["w2"].to(dev); w2_s = W["w2_scale"].to(dev)

    # [1] prepare/repack (one-time): pack each expert + strided ptrs
    def do_prepare():
        w13_tw, w13_ts, w2_tw, w2_ts = [], [], [], []
        k13 = q13 = k2 = q2 = None
        for e in range(E):
            r13 = ops.fp8_sm70_prepare(w13[e], w13_s[e], block)
            r2 = ops.fp8_sm70_prepare(w2[e], w2_s[e], block)
            w13_tw.append(r13[0]); w13_ts.append(r13[1])
            w2_tw.append(r2[0]); w2_ts.append(r2[1])
            if e == 0:
                k13, q13 = int(r13[2][0]), int(r13[2][1])
                k2, q2 = int(r2[2][0]), int(r2[2][1])
        W13 = torch.stack(w13_tw); S13 = torch.stack(w13_ts)
        W2 = torch.stack(w2_tw); S2 = torch.stack(w2_ts)
        p13 = ops.awq_moe_build_strided_ptrs(W13, S13, k13, q13, E)
        p2 = ops.awq_moe_build_strided_ptrs(W2, S2, k2, q2, E)
        return (W13, S13, W2, S2, p13, p2)

    t_prep = timed(do_prepare, n_warmup=1, n_iter=3)
    W13, S13, W2, S2, p13, p2 = do_prepare()
    print(f"[1catai/{model}] prepare/repack (one-time, all {E} experts): {t_prep:.3f} ms")

    rows = []
    for cfg in sorted(inp.glob("route_*.pt")):
        R = torch.load(cfg)
        eids = R["eids"].to(dev); x13 = R["x13"].to(dev)
        y_ref = R["y_ref"].to(dev); tpe = R["tpe"]; regime = R["regime"]; Rn = R["R"]
        top_k = 1                      # per-slot routing model (see header)
        topk_ids_i32 = eids.reshape(Rn, top_k).to(torch.int32)
        topk_w = torch.ones(Rn, top_k, dtype=torch.float32, device=dev)

        # persistent buffers (mirror Fp8SM70MoEMethod.apply — raw ops, no per-call alloc)
        tok_expert_idx = torch.arange(Rn * top_k, dtype=torch.int32,
                                      device=dev).view(Rn, top_k)
        permuted_input = torch.empty(Rn, H, dtype=torch.float16, device=dev)
        off64 = torch.empty(E + 1, dtype=torch.int64, device=dev)
        off32 = torch.empty(E + 1, dtype=torch.int32, device=dev)
        inv_idx = torch.empty(Rn, top_k, dtype=torch.int32, device=dev)
        permuted_idx = torch.empty(Rn * top_k, dtype=torch.int32, device=dev)
        m_indices = torch.empty(Rn * top_k, dtype=torch.int32, device=dev)
        gate_up = torch.empty(Rn, N13, dtype=torch.float16, device=dev)
        inter = torch.empty(Rn, I, dtype=torch.float16, device=dev)
        sorted_out = torch.empty(Rn, N2, dtype=torch.float16, device=dev)
        out = torch.empty(Rn, H, dtype=torch.float16, device=dev)

        def route():
            torch.ops._moe_C.moe_permute(
                x13, topk_ids_i32, tok_expert_idx, None, E, E, top_k, None,
                permuted_input, off64, inv_idx, permuted_idx, m_indices)
            off32.copy_(off64, non_blocking=True)

        def kernels():
            ops.fp8_moe_gemm_sm70_out(gate_up, permuted_input, off32, p13[0], p13[1],
                                      E, K13, N13, block, False)
            torch.ops._C.silu_and_mul(inter, gate_up)
            ops.fp8_moe_gemm_sm70_out(sorted_out, inter, off32, p2[0], p2[1],
                                      E, K2, N2, block, False)

        def unpermute():
            torch.ops._moe_C.moe_unpermute(sorted_out, topk_w, inv_idx, off64, top_k, out)

        def e2e():
            route(); kernels(); unpermute()

        # correctness gate FIRST (full e2e once, compare to ref in original row order)
        e2e()
        cos = cossim(out, y_ref)
        gate = "ok" if cos >= 0.99 else "FAIL"

        t_route = timed(route)
        t_kern = timed(kernels)
        t_unperm = timed(unpermute)
        t_e2e = timed(e2e)
        rows.append([model, regime, tpe, Rn, f"{cos:.4f}", gate,
                     f"{t_route:.4f}", f"{t_kern:.4f}", f"{t_unperm:.4f}", f"{t_e2e:.4f}"])
        print(f"  {regime:7s} tpe={tpe} R={Rn:5d} cos={cos:.4f}[{gate}]  "
              f"route={t_route:.3f} kern={t_kern:.3f} unperm={t_unperm:.3f} e2e={t_e2e:.3f} ms")

    outp = args.out or f"results_moe_1catai_{model}.csv"
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "regime", "tpe", "R", "cos", "gate",
                    "route_ms", "kernel_ms", "unperm_ms", "e2e_ms"])
        w.writerow(["#prepare_once_ms", f"{t_prep:.3f}"])
        w.writerows(rows)
    print(f"[ok] wrote {outp} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
