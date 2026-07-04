#!/usr/bin/env python3
"""
Reproducible build of the vendored TurboMind SM70 FP8 engine — FROM OUR TREE (no external
checkout) — plus the Stage-D real-Qwen dense gate.

Pin: lmdeploy v0.14.0 + carried V100 patch (see PROVENANCE.md). Compiles the sm70-only source
list via torch cpp_extension in our cu126 image; registers dense ops under `turbomind_fp8_sm70`;
round-trips a real Qwen3.5-35B-A3B-FP8 expert vs an fp32 dequant reference (expect cos=1.0000).

Run inside vllm-v100:vllm021-cu126 with /mnt/models mounted ro.
"""
import glob, os, re, sys
import torch
from _ext_build import build_ops, BLOCK

ops = build_ops()

# ---- Stage-D real-Qwen dense gate ----
dev = "cuda:0"
MODEL = os.environ.get("MODEL_DIR", "/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8")
from safetensors import safe_open


def cossim(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def dequant(w, s, block=BLOCK):
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


def lt(k):
    with safe_open(k2f[k], framework="pt") as fh:
        return fh.get_tensor(k)


gw, gs = lt(f"{pre}.experts.0.gate_proj.weight"), lt(f"{pre}.experts.0.gate_proj.weight_scale_inv")
uw, us = lt(f"{pre}.experts.0.up_proj.weight"), lt(f"{pre}.experts.0.up_proj.weight_scale_inv")
w13 = torch.cat([gw, uw], 0).to(dev).contiguous()
w13_s = torch.cat([gs, us], 0).float().to(dev).contiguous()
N13, K13 = w13.shape
W_dq = dequant(w13, w13_s)
tm_w, tm_s, meta = ops.fp8_sm70_prepare(w13, w13_s, BLOCK)
k_ld, q_ld = int(meta[0]), int(meta[1])
print(f"[vendor] real Qwen w13={tuple(w13.shape)}  prepare OK k_ld={k_ld} q_ld={q_ld}")

ok = True
print(f"{'M':>4} {'cos':>10} {'max_abs':>12}")
for M in (1, 4, 16):
    g = torch.Generator().manual_seed(100 + M)
    A = (torch.randn(M, K13, generator=g) * 0.1).half().to(dev)
    ref = A.float() @ W_dq.T
    out = torch.empty(M, N13, dtype=torch.float16, device=dev)
    ops.fp8_gemm_sm70_out(out, A, tm_w, tm_s, BLOCK, k_ld, q_ld)
    c = cossim(out, ref); e = (out.float() - ref).abs().max().item()
    print(f"{M:>4} {c:>10.4f} {e:>12.3e}")
    ok = ok and c >= 0.99

print(f"\n=== VENDOR GATE {'PASS' if ok else 'FAIL'}: v0.14.0+V100patch builds from our tree "
      f"+ real-Qwen dense cos>=0.99 = {ok} ===")
sys.exit(0 if ok else 1)
