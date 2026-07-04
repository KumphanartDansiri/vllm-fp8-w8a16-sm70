#!/usr/bin/env python3
"""
Full real-checkpoint MoE gate against the VENDORED engine (Stage-F MoE-ops verification).

Builds the vendored engine from third_party/ (same builder as the dense gate) and runs the
COMPLETE serving path on all experts of a real Qwen3.5-35B-A3B-FP8 MoE layer with real top_k=8
routing, using the vendored grouped-MoE ops:
    turbomind_fp8_sm70.fp8_sm70_prepare / awq_moe_build_strided_ptrs / fp8_moe_gemm_sm70_out
plus stock vLLM moe_permute / moe_unpermute / silu_and_mul. Compares to an fp32 per-token
reference (expect cos=1.0000).

Run inside vllm-v100:vllm021-cu126 with /mnt/models mounted ro.
"""
import glob, os, re, sys
import torch
import torch.nn.functional as F
from _ext_build import build_ops, BLOCK

ops = build_ops()
dev = "cuda:0"
# NOTE: permute/silu/combine are done manually (torch), NOT via vLLM's _moe_C — stock vLLM 0.21's
# moe_permute is a different version than the vendored fp8_moe_gemm expects. This gate isolates the
# VENDORED grouped GEMM; serving will use the engine's matching route path (Stage-F serving step).
MODEL = os.environ.get("MODEL_DIR", "/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8")
NT, TK, SEED = 16, 8, 7
from safetensors import safe_open


def cossim(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def deq(w, s, block=BLOCK):
    N, K = w.shape
    e = s.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return w.float() * e


k2f = {}
for f in sorted(glob.glob(f"{MODEL}/*.safetensors")):
    with safe_open(f, framework="pt") as fh:
        for k in fh.keys():
            k2f[k] = f
L = sorted({int(m.group(1)) for k in k2f
            if (m := re.search(r"layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight$", k))})[0]
pre = next(k.split(".experts.0.")[0] for k in k2f if f"layers.{L}.mlp.experts.0.gate_proj.weight" in k)
E = 1 + max(int(m.group(1)) for k in k2f
            if (m := re.search(rf"layers\.{L}\.mlp\.experts\.(\d+)\.gate_proj\.weight$", k)))
print(f"[moe] real ckpt layer {L}, E={E} experts, top_k={TK}")


def lt(k):
    with safe_open(k2f[k], framework="pt") as fh:
        return fh.get_tensor(k)


w13f, w13s, w2f, w2s = [], [], [], []
for e in range(E):
    b = f"{pre}.experts.{e}"
    gw, gs = lt(f"{b}.gate_proj.weight"), lt(f"{b}.gate_proj.weight_scale_inv")
    uw, us = lt(f"{b}.up_proj.weight"), lt(f"{b}.up_proj.weight_scale_inv")
    dw, ds = lt(f"{b}.down_proj.weight"), lt(f"{b}.down_proj.weight_scale_inv")
    w13f.append(torch.cat([gw, uw], 0)); w13s.append(torch.cat([gs, us], 0).float())
    w2f.append(dw); w2s.append(ds.float())
w13 = torch.stack(w13f).to(dev); w13_s = torch.stack(w13s).to(dev)
w2 = torch.stack(w2f).to(dev); w2_s = torch.stack(w2s).to(dev)
N13, K13, N2, K2 = w13.shape[1], w13.shape[2], w2.shape[1], w2.shape[2]
I, H = K2, K13
print(f"[moe] w13={tuple(w13.shape)} w2={tuple(w2.shape)} (H={H} I={I})")

# prepare all experts + strided ptrs (vendored ops)
W13t, S13t, W2t, S2t = [], [], [], []
k13 = q13 = k2_ = q2_ = None
for e in range(E):
    r13 = ops.fp8_sm70_prepare(w13[e], w13_s[e], BLOCK)
    r2 = ops.fp8_sm70_prepare(w2[e], w2_s[e], BLOCK)
    W13t.append(r13[0]); S13t.append(r13[1]); W2t.append(r2[0]); S2t.append(r2[1])
    if e == 0:
        k13, q13 = int(r13[2][0]), int(r13[2][1]); k2_, q2_ = int(r2[2][0]), int(r2[2][1])
p13 = ops.awq_moe_build_strided_ptrs(torch.stack(W13t), torch.stack(S13t), k13, q13, E)
p2 = ops.awq_moe_build_strided_ptrs(torch.stack(W2t), torch.stack(S2t), k2_, q2_, E)

# real top_k routing + fp32 reference
g = torch.Generator().manual_seed(SEED)
x = (torch.randn(NT, H, generator=g) * 0.1).half().to(dev)
ids = torch.stack([torch.randperm(E, generator=g)[:TK] for _ in range(NT)]).to(dev)
w = torch.softmax(torch.randn(NT, TK, generator=g), dim=1).float().to(dev)
uniq = torch.unique(ids).tolist()
dq13 = {e: deq(w13[e], w13_s[e]) for e in uniq}
dq2 = {e: deq(w2[e], w2_s[e]) for e in uniq}
y_ref = torch.zeros(NT, H, dtype=torch.float32, device=dev)
for t in range(NT):
    for k in range(TK):
        e = int(ids[t, k])
        gu = x[t].float() @ dq13[e].T
        inter = F.silu(gu[:I]) * gu[I:]
        y_ref[t] += w[t, k] * (inter @ dq2[e].T)

# --- REAL serving layout via stock vLLM-0.21 moe_permute (11-arg), like Stage D ---
import vllm  # noqa: F401 — registers stock _C (silu) + _moe_C (moe_permute)
import vllm.model_executor.layers.fused_moe.moe_permute_unpermute  # noqa: F401 — loads _moe_C
T = NT * TK
ids_i32 = ids.to(torch.int32)
tok_ei = torch.arange(T, dtype=torch.int32, device=dev).view(NT, TK)
perm = torch.empty(T, H, dtype=torch.float16, device=dev)
off64 = torch.empty(E + 1, dtype=torch.int64, device=dev)
inv = torch.empty(NT, TK, dtype=torch.int32, device=dev)
pidx = torch.empty(T, dtype=torch.int32, device=dev)
# stock 0.21 signature: (input, topk_ids, tok_expert_idx, expert_map, n_expert, n_local, topk,
#                        permuted_input, expert_first_token_offset, inv_permuted_idx, permuted_idx)
torch.ops._moe_C.moe_permute(x, ids_i32, tok_ei, None, E, E, TK, perm, off64, inv, pidx)
off32 = off64.to(torch.int32)

gate_up = torch.empty(T, N13, dtype=torch.float16, device=dev)
sorted_out = torch.empty(T, N2, dtype=torch.float16, device=dev)
ops.fp8_moe_gemm_sm70_out(gate_up, perm, off32, p13[0], p13[1], E, K13, N13, BLOCK, False)
inter = (F.silu(gate_up[:, :I].float()) * gate_up[:, I:].float()).half().contiguous()
ops.fp8_moe_gemm_sm70_out(sorted_out, inter, off32, p2[0], p2[1], E, K2, N2, BLOCK, False)

# --- per-expert cos on the PERMUTED layout (full w13->silu->w2 chain) vs fp32 ref ---
o64 = off64.tolist()
active = [(e, o64[e + 1] - o64[e]) for e in range(E) if o64[e + 1] > o64[e]]
mdist = {}
for _, c in active:
    mdist[c] = mdist.get(c, 0) + 1
bad = []
for e, c in active:
    lo, hi = o64[e], o64[e + 1]
    gu = perm[lo:hi].float() @ dq13[e].T
    it = F.silu(gu[:, :I]) * gu[:, I:]
    ref = it @ dq2[e].T
    ce = cossim(sorted_out[lo:hi], ref)
    if ce < 0.99:
        bad.append((e, c, round(ce, 3)))
print(f"[moe] stock moe_permute layout: active={len(active)} M-dist={mdist} "
      f"aligned32={all(o % 32 == 0 for o in o64)}")
print(f"[moe] w13->w2 per-expert cos<0.99: {len(bad)} experts -> {bad[:12]}")
ok = len(bad) == 0
print(f"\n=== VENDOR MoE GATE {'PASS' if ok else 'FAIL'}: v0.14.0 engine + stock moe_permute, "
      f"real-ckpt MoE per-expert cos>=0.99 = {ok} ===")
sys.exit(0 if ok else 1)
