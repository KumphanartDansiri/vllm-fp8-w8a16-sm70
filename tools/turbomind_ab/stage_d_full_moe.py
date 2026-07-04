#!/usr/bin/env python3
"""
Stage-D full MoE layer gate :: real checkpoint end-to-end through TurboMind.

Loads ALL experts of one real Qwen3.5-35B-A3B-FP8 MoE layer (gate/up/down block-FP8),
prepares them, and runs the COMPLETE serving path with real top_k=8 routing:
  moe_permute -> fp8_moe_gemm(w13) -> silu_and_mul -> fp8_moe_gemm(w2) -> moe_unpermute
against an independent fp32 reference (per-token sum over its 8 experts). Also:
  * w2/down single-expert round-trip cos (w13 was proven in stage_d_format_gate.py)
  * TP scale-sharding shape check: where does block-128 break as intermediate shards?

Run in 1catai-vllm-v100:cu128-fp8sm70 with /mnt/models mounted ro.
"""
import argparse, glob, re, sys
import torch
from safetensors import safe_open
from vllm import _custom_ops as ops

BLOCK = 128


def cossim(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def deq(w_fp8, scale, block=BLOCK):
    N, K = w_fp8.shape
    s = scale.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return w_fp8.float() * s


def build_index(model_dir):
    key2file = {}
    for f in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                key2file[k] = f
    return key2file


def load(key2file, k):
    with safe_open(key2file[k], framework="pt") as fh:
        return fh.get_tensor(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--n-token", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dev = "cuda:0"
    if not hasattr(torch.ops._C, "fp8_moe_gemm_sm70_out"):
        sys.exit("[FATAL] not a 1catai SM70 FP8 MoE build.")

    k2f = build_index(args.model_dir)
    # choose the lowest layer that has experts, count E
    layers = sorted({int(m.group(1)) for k in k2f
                     if (m := re.search(r"layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight$", k))})
    L = layers[0]
    pref = None
    for k in k2f:
        if f"layers.{L}.mlp.experts.0.gate_proj.weight" in k:
            pref = k.split(f".experts.0.")[0]; break
    E = 1 + max(int(m.group(1)) for k in k2f
                if (m := re.search(rf"layers\.{L}\.mlp\.experts\.(\d+)\.gate_proj\.weight$", k)))
    print(f"[real ckpt] layer {L}, E={E} experts, top_k={args.top_k}")

    # load + fuse all experts
    w13_fp8, w13_scl, w2_fp8, w2_scl = [], [], [], []
    for e in range(E):
        b = f"{pref}.experts.{e}"
        gw = load(k2f, f"{b}.gate_proj.weight"); gs = load(k2f, f"{b}.gate_proj.weight_scale_inv")
        uw = load(k2f, f"{b}.up_proj.weight");   us = load(k2f, f"{b}.up_proj.weight_scale_inv")
        dw = load(k2f, f"{b}.down_proj.weight");  ds = load(k2f, f"{b}.down_proj.weight_scale_inv")
        w13_fp8.append(torch.cat([gw, uw], 0)); w13_scl.append(torch.cat([gs, us], 0).float())
        w2_fp8.append(dw); w2_scl.append(ds.float())
    w13 = torch.stack(w13_fp8).to(dev); w13_s = torch.stack(w13_scl).to(dev)
    w2 = torch.stack(w2_fp8).to(dev); w2_s = torch.stack(w2_scl).to(dev)
    N13, K13 = w13.shape[1], w13.shape[2]      # [2I, H]
    N2, K2 = w2.shape[1], w2.shape[2]          # [H, I]
    I, H = K2, K13
    print(f"  w13={tuple(w13.shape)} w2={tuple(w2.shape)}  (H={H} I={I})")

    # ---- w2/down single-expert round-trip (w13 already proven) ----
    W2dq0 = deq(w2[0], w2_s[0])
    tw, ts, meta = ops.fp8_sm70_prepare(w2[0], w2_s[0], BLOCK)
    A = (torch.randn(8, K2, generator=torch.Generator().manual_seed(1)) * 0.1).half().to(dev)
    o = torch.empty(8, N2, dtype=torch.float16, device=dev)
    ops.fp8_gemm_sm70_out(o, A, tw, ts, BLOCK, int(meta[0]), int(meta[1]))
    print(f"[w2 round-trip] cos={cossim(o, A.float() @ W2dq0.T):.4f}")

    # ---- prepare all experts + strided ptrs ----
    W13t, S13t, W2t, S2t = [], [], [], []
    k13 = q13 = k2_ = q2_ = None
    for e in range(E):
        r13 = ops.fp8_sm70_prepare(w13[e], w13_s[e], BLOCK)
        r2 = ops.fp8_sm70_prepare(w2[e], w2_s[e], BLOCK)
        W13t.append(r13[0]); S13t.append(r13[1]); W2t.append(r2[0]); S2t.append(r2[1])
        if e == 0:
            k13, q13 = int(r13[2][0]), int(r13[2][1]); k2_, q2_ = int(r2[2][0]), int(r2[2][1])
    W13s, S13s = torch.stack(W13t), torch.stack(S13t)
    W2s, S2s = torch.stack(W2t), torch.stack(S2t)
    p13 = ops.awq_moe_build_strided_ptrs(W13s, S13s, k13, q13, E)
    p2 = ops.awq_moe_build_strided_ptrs(W2s, S2s, k2_, q2_, E)

    # ---- real top_k routing + fp32 reference ----
    nt, tk = args.n_token, args.top_k
    g = torch.Generator().manual_seed(args.seed)
    x = (torch.randn(nt, H, generator=g) * 0.1).half().to(dev)
    ids = torch.stack([torch.randperm(E, generator=g)[:tk] for _ in range(nt)]).to(dev)  # distinct
    w = torch.softmax(torch.randn(nt, tk, generator=g), dim=1).float().to(dev)
    uniq = torch.unique(ids).tolist()
    dq13 = {e: deq(w13[e], w13_s[e]) for e in uniq}
    dq2 = {e: deq(w2[e], w2_s[e]) for e in uniq}
    y_ref = torch.zeros(nt, H, dtype=torch.float32, device=dev)
    for t in range(nt):
        for k in range(tk):
            e = int(ids[t, k])
            gu = x[t].float() @ dq13[e].T
            inter = torch.nn.functional.silu(gu[:I]) * gu[I:]
            y_ref[t] += w[t, k] * (inter @ dq2[e].T)

    # ---- TurboMind full apply path (raw ops + buffers, top_k=tk) ----
    T = nt * tk
    ids_i32 = ids.to(torch.int32)
    tok_ei = torch.arange(T, dtype=torch.int32, device=dev).view(nt, tk)
    perm = torch.empty(T, H, dtype=torch.float16, device=dev)
    off64 = torch.empty(E + 1, dtype=torch.int64, device=dev)
    off32 = torch.empty(E + 1, dtype=torch.int32, device=dev)
    inv = torch.empty(nt, tk, dtype=torch.int32, device=dev)
    pidx = torch.empty(T, dtype=torch.int32, device=dev)
    midx = torch.empty(T, dtype=torch.int32, device=dev)
    gate_up = torch.empty(T, N13, dtype=torch.float16, device=dev)
    inter = torch.empty(T, I, dtype=torch.float16, device=dev)
    sorted_out = torch.empty(T, N2, dtype=torch.float16, device=dev)
    out = torch.empty(nt, H, dtype=torch.float16, device=dev)

    torch.ops._moe_C.moe_permute(x, ids_i32, tok_ei, None, E, E, tk, None,
                                 perm, off64, inv, pidx, midx)
    off32.copy_(off64)
    ops.fp8_moe_gemm_sm70_out(gate_up, perm, off32, p13[0], p13[1], E, K13, N13, BLOCK, False)

    # --- RECONCILE (2026-07-04): does the moe_permute layout make w13 (K=2048) correct per-expert? ---
    o64 = off64.tolist()
    counts_ = [o64[e + 1] - o64[e] for e in range(E)]
    active = [(e, c) for e, c in enumerate(counts_) if c > 0]
    mdist = {}
    for _, c in active:
        mdist[c] = mdist.get(c, 0) + 1
    bad = []
    for e, c in active:
        lo, hi = o64[e], o64[e + 1]
        ce = cossim(gate_up[lo:hi], perm[lo:hi].float() @ dq13[e].T)
        if ce < 0.99:
            bad.append((e, c, round(ce, 3)))
    aligned32 = all(o % 32 == 0 for o in o64)
    print(f"[reconcile] active_experts={len(active)} M-dist(count:num_experts)={mdist} "
          f"offsets_aligned32={aligned32}")
    print(f"[reconcile] w13 per-expert cos<0.99: {len(bad)} experts -> {bad[:12]}")

    torch.ops._C.silu_and_mul(inter, gate_up)
    ops.fp8_moe_gemm_sm70_out(sorted_out, inter, off32, p2[0], p2[1], E, K2, N2, BLOCK, False)
    torch.ops._moe_C.moe_unpermute(sorted_out, w, inv, off64, tk, out)  # w is float32

    cos = cossim(out, y_ref)
    print(f"[full MoE layer e2e]  n_token={nt} top_k={tk} T={T} uniq_experts={len(uniq)}  "
          f"cos={cos:.4f}")

    # ---- TP scale-sharding shape check (block-128 constraint) ----
    print("\n[TP scale sharding] intermediate I sharded across TP ranks (w13 col-par, w2 row-par):")
    for tp in (1, 2, 4, 8):
        Ish = I // tp
        ok = (Ish % BLOCK == 0) and (Ish > 0) and (I % tp == 0)
        note = "OK" if ok else f"BREAKS (I/tp={Ish} not mult of {BLOCK})"
        print(f"  tp={tp}: I/tp={Ish:5d}  block-128 {note}")

    passed = cos >= 0.99
    print(f"\n=== FULL-LAYER GATE {'PASS' if passed else 'FAIL'}: real-ckpt MoE layer cos={cos:.4f} ===")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
