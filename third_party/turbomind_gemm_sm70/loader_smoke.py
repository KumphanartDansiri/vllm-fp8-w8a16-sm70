#!/usr/bin/env python3
"""Stage F loader-wiring smoke — dense + grouped MoE, against the WIRED native-Fp8 helpers.

Unlike build_and_gate*.py (which drive the raw ops), this imports the ACTUAL functions wired
into the loader and exercises them exactly as serving will:
    vllm_serve._tm_dense_prepare / _tm_dense_apply        (dense block-FP8 Linear)
    vllm_serve._tm_moe_prepare  / _our_moe_apply_turbomind (grouped block-FP8 MoE)
on real Qwen3.5-35B-A3B-FP8 weights, with the vendored engine JIT-built.

It confirms, in order:
  1. turbomind_fp8_backend.select_backend() routes these (un-sharded ≈ TP1) shapes to turbomind;
  2. prepare-at-load stores packed weights + meta(k_ld,q_ld) on the layer;
  3. _tm_dense_apply reproduces an fp16 dequant reference (cos >= 0.99, M = 1/4/16);
  4. the MoE permute -> w13 -> SwiGLU -> w2 -> unpermute COMBINE reproduces an fp32 per-token
     top_k reference (cos >= 0.99) — the one step build_and_gate_moe.py deliberately skipped.

This is the correctness gate that must pass BEFORE any TP serving (validation order, Stage F note).

Run inside vllm-v100:vllm021-cu126 with /mnt/models mounted ro:
  docker run --rm --gpus all --entrypoint /bin/bash \
    -v /mnt/models:/mnt/models:ro -v "$PWD":/work -w /work \
    vllm-v100:vllm021-cu126 -lc \
    "VLLM_V100_FP8_ENGINE_JIT=1 python3 third_party/turbomind_gemm_sm70/loader_smoke.py"
"""
import os
import sys
import glob
import re

# Force the vendored engine to JIT-build (dev), and force the turbomind backend so any
# ineligible shape HARD-RAISES instead of silently using ours. Keep raw weights (exercise
# the _TM_FREE_RAW=0 branch: replace_parameter with the fp8 weight, like the ours path).
os.environ.setdefault("VLLM_V100_FP8_ENGINE_JIT", "1")
os.environ.setdefault("VLLM_V100_FP8_BACKEND", "turbomind")
os.environ.setdefault("VLLM_V100_FP8_TM_FREE_RAW", "0")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# Importing the serve module compiles the W8A16 kernel, applies the V100 FP8 patches, and
# (with the JIT flag) builds the vendored engine — exactly the startup path serving takes.
from fp8_w8a16_sm70 import vllm_serve as vs
from fp8_w8a16_sm70 import turbomind_fp8_backend as tb

DEV = "cuda:0"
BLOCK = 128
MODEL = os.environ.get("MODEL_DIR", "/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8")


def cossim(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def deq(w, s, block=BLOCK):
    N, K = w.shape
    e = s.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return w.float() * e


# ── locate a real MoE layer's experts on disk ───────────────────────────────────
k2f = {}
for f in sorted(glob.glob(f"{MODEL}/*.safetensors")):
    with safe_open(f, framework="pt") as fh:
        for k in fh.keys():
            k2f[k] = f


def lt(k):
    with safe_open(k2f[k], framework="pt") as fh:
        return fh.get_tensor(k)


L = sorted({int(m.group(1)) for k in k2f
            if (m := re.search(r"layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight$", k))})[0]
pre = next(k.split(".experts.0.")[0]
           for k in k2f if f"layers.{L}.mlp.experts.0.gate_proj.weight" in k)
E = 1 + max(int(m.group(1)) for k in k2f
            if (m := re.search(rf"layers\.{L}\.mlp\.experts\.(\d+)\.gate_proj\.weight$", k)))
print(f"[smoke] engine ops_available(moe)={tb.ops_available(need_moe=True)} "
      f"backend_mode={tb.backend_mode()}")
print(f"[smoke] model={MODEL}  layer={L}  E={E}")

fails = 0

# ── DENSE: expert-0 gate_proj is a real block-FP8 [N,K] weight ───────────────────
w = lt(f"{pre}.experts.0.gate_proj.weight").to(DEV).contiguous()            # fp8 [N,K]
s = lt(f"{pre}.experts.0.gate_proj.weight_scale_inv").to(DEV).contiguous()  # [N/128,K/128]
N, K = int(w.shape[0]), int(w.shape[1])
be, why = tb.select_backend(strategy="BLOCK", weight_block_size=(128, 128),
                            local_n=N, local_k=K, need_moe=False, quiet=True)
print(f"[smoke][dense] N={N} K={K} select_backend={be}  ({why})")
assert be == "turbomind", f"dense [{N},{K}] should route to turbomind, got {be}"

self_d = type("FakeDenseMethod", (), {"weight_block_size": [128, 128]})()
layer_d = nn.Module()
vs._tm_dense_prepare(self_d, layer_d, w, s, [128, 128], why)
assert getattr(layer_d, "_v100_fp8_backend", None) == "turbomind"
for M in (1, 4, 16):
    g = torch.Generator().manual_seed(100 + M)
    x = (torch.randn(M, K, generator=g) * 0.1).half().to(DEV)
    ref = x.float() @ deq(w, s).T
    out = vs._tm_dense_apply(layer_d, x, None)
    c = cossim(out, ref)
    ok = c >= 0.99
    fails += (not ok)
    print(f"[smoke][dense] M={M:>3}  cos={c:.4f}  {'OK' if ok else 'FAIL'}")

# ── MoE: all experts of a real layer, real top_k=8 routing ───────────────────────
w13f, w13s, w2f, w2s = [], [], [], []
for e in range(E):
    b = f"{pre}.experts.{e}"
    gw, gs = lt(f"{b}.gate_proj.weight"), lt(f"{b}.gate_proj.weight_scale_inv")
    uw, us = lt(f"{b}.up_proj.weight"), lt(f"{b}.up_proj.weight_scale_inv")
    dw, ds = lt(f"{b}.down_proj.weight"), lt(f"{b}.down_proj.weight_scale_inv")
    w13f.append(torch.cat([gw, uw], 0)); w13s.append(torch.cat([gs, us], 0))
    w2f.append(dw); w2s.append(ds)
w13 = torch.stack(w13f).to(DEV); w13_s = torch.stack(w13s).to(DEV)
w2 = torch.stack(w2f).to(DEV);  w2_s = torch.stack(w2s).to(DEV)
N13, K13 = int(w13.shape[1]), int(w13.shape[2])
N2, K2 = int(w2.shape[1]), int(w2.shape[2])
I, H = K2, K13
print(f"[smoke][moe] w13={tuple(w13.shape)} w2={tuple(w2.shape)} H={H} I={I}")

self_m = type("FakeMoEMethod", (), {
    "weight_block_size": [128, 128], "weight_scale_name": "weight_scale_inv",
    "moe_quant_config": None, "moe_kernel": None})()
layer_m = nn.Module()
layer_m.w13_weight = w13
layer_m.w2_weight = w2
layer_m.w13_weight_scale_inv = w13_s
layer_m.w2_weight_scale_inv = w2_s
layer_m.expert_map = None
vs._tm_moe_prepare(self_m, layer_m)
assert getattr(layer_m, "_v100_fp8_backend", None) == "turbomind"

NT, TK, SEED = 16, 8, 7
g = torch.Generator().manual_seed(SEED)
x = (torch.randn(NT, H, generator=g) * 0.1).half().to(DEV)
ids = torch.stack([torch.randperm(E, generator=g)[:TK] for _ in range(NT)]).to(DEV)
wts = torch.softmax(torch.randn(NT, TK, generator=g), dim=1).to(DEV)
uniq = torch.unique(ids).tolist()
dq13 = {e: deq(w13[e], w13_s[e]) for e in uniq}
dq2 = {e: deq(w2[e], w2_s[e]) for e in uniq}
y_ref = torch.zeros(NT, H, dtype=torch.float32, device=DEV)
for t in range(NT):
    for k in range(TK):
        e = int(ids[t, k])
        gu = x[t].float() @ dq13[e].T
        inter = F.silu(gu[:I]) * gu[I:]
        y_ref[t] += wts[t, k].float() * (inter @ dq2[e].T)

out = vs._our_moe_apply_turbomind(self_m, layer_m, x, wts, ids)
c = cossim(out, y_ref)
ok = c >= 0.99
fails += (not ok)
print(f"[smoke][moe] full-layer permute->w13->silu->w2->unpermute cos={c:.4f}  "
      f"{'OK' if ok else 'FAIL'}")

print(f"\n=== STAGE-F LOADER SMOKE {'PASS' if fails == 0 else f'FAIL ({fails})'} "
      f"(dense M=1/4/16 + grouped MoE, real Qwen block-FP8, wired helpers) ===")
sys.exit(1 if fails else 0)
