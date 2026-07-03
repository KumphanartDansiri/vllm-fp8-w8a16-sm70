#!/usr/bin/env python3
"""
Stage-D format gate :: can TurboMind's fp8_sm70_prepare consume OUR REAL checkpoints?

Loads a REAL block-FP8 compressed-tensors MoE expert (Qwen3.5-35B-A3B-FP8) straight off
disk -- fp8_e4m3fn weight + BF16 weight_scale_inv[N/128,K/128] -- fuses gate/up like the
loader does, feeds it through fp8_sm70_prepare -> fp8_gemm_sm70_out, and checks cos vs an
independent fp32 dequant (weight_fp8 * scale). Proves (or refutes) that our real block-FP8
layout + scale semantics round-trip through the TurboMind engine with only a bf16->fp32 cast.

Also feeds a CHANNEL-scale weight ([N,1]) to fp8_sm70_prepare to confirm it is REJECTED
loudly (TORCH_CHECK), i.e. channel-scale (GLM-4.5-Air W8A8) must fall back to our path.

Run in 1catai-vllm-v100:cu128-fp8sm70 with /mnt/models mounted ro.
"""
import argparse, glob, sys
from pathlib import Path
import torch
from safetensors import safe_open
from vllm import _custom_ops as ops

BLOCK = 128


def cossim(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def dequant_block(w_fp8, scale, block):
    N, K = w_fp8.shape
    s = scale.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return w_fp8.float() * s


def find_expert_tensors(model_dir):
    """Return (gate_w, gate_s, up_w, up_s) for experts.0 of the first layer that has them."""
    files = sorted(glob.glob(f"{model_dir}/*.safetensors"))
    # index which shard holds which key
    key2file = {}
    for f in files:
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                key2file[k] = f
    # pick the lowest layer index with experts.0 gate+up weight+scale present
    import re
    layers = set()
    for k in key2file:
        m = re.search(r"layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight$", k)
        if m:
            layers.add(int(m.group(1)))
    for L in sorted(layers):
        base = None
        for k in key2file:
            if f"layers.{L}.mlp.experts.0.gate_proj.weight" in k:
                base = k.replace(".gate_proj.weight", "")
                break
        need = {f"{base}.gate_proj.weight", f"{base}.gate_proj.weight_scale_inv",
                f"{base}.up_proj.weight", f"{base}.up_proj.weight_scale_inv"}
        if need <= set(key2file):
            def get(k):
                with safe_open(key2file[k], framework="pt") as fh:
                    return fh.get_tensor(k)
            return (L, get(f"{base}.gate_proj.weight"),
                    get(f"{base}.gate_proj.weight_scale_inv"),
                    get(f"{base}.up_proj.weight"),
                    get(f"{base}.up_proj.weight_scale_inv"))
    sys.exit("No experts.0 gate/up block-FP8 tensors found.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--M", type=int, nargs="+", default=[1, 4, 16])
    args = ap.parse_args()
    dev = "cuda:0"
    if not hasattr(torch.ops._C, "fp8_sm70_prepare"):
        sys.exit("[FATAL] not a 1catai SM70 FP8 build.")

    L, gw, gs, uw, us = find_expert_tensors(args.model_dir)
    print(f"[real ckpt] layer {L} experts.0  gate={tuple(gw.shape)}/{gw.dtype} "
          f"scale={tuple(gs.shape)}/{gs.dtype}")

    # fuse gate/up -> w13 [2I,H]; scales -> [2I/128, H/128]; cast scale bf16->fp32
    w13 = torch.cat([gw, uw], dim=0).to(dev).contiguous()          # fp8_e4m3fn [2I,H]
    w13_s = torch.cat([gs, us], dim=0).to(torch.float32).to(dev).contiguous()
    N13, K13 = w13.shape
    print(f"[fused]   w13={tuple(w13.shape)}/{w13.dtype}  scale={tuple(w13_s.shape)}/"
          f"{w13_s.dtype}  (expect scale=[{N13//BLOCK},{K13//BLOCK}])")
    assert w13_s.shape == (N13 // BLOCK, K13 // BLOCK), "block scale orientation mismatch!"

    # reference dequant + prepare/gemm
    W_dq = dequant_block(w13, w13_s, BLOCK)                         # fp32 [2I,H]
    tm_w, tm_s, meta = ops.fp8_sm70_prepare(w13, w13_s, BLOCK)
    k_ld, q_ld = int(meta[0]), int(meta[1])
    print(f"[prepare] OK  meta k_ld={k_ld} q_ld={q_ld} (packed)")

    print(f"\n{'M':>4} {'cos':>10} {'max_abs':>12}   (real block-FP8 expert -> TurboMind)")
    ok = True
    for M in args.M:
        g = torch.Generator().manual_seed(100 + M)
        A = (torch.randn(M, K13, generator=g) * 0.1).to(torch.float16).to(dev)
        ref = A.float() @ W_dq.T
        out = torch.empty(M, N13, dtype=torch.float16, device=dev)
        ops.fp8_gemm_sm70_out(out, A, tm_w, tm_s, BLOCK, k_ld, q_ld)
        c = cossim(out, ref); e = (out.float() - ref).abs().max().item()
        print(f"{M:>4} {c:>10.4f} {e:>12.3e}")
        ok = ok and c >= 0.99

    # channel-scale must be REJECTED loudly
    print("\n[channel-scale gate] feeding [N,1] channel scale to fp8_sm70_prepare ...")
    chan_s = torch.ones(N13, 1, dtype=torch.float32, device=dev)
    try:
        ops.fp8_sm70_prepare(w13, chan_s, BLOCK)
        print("  UNEXPECTED: channel scale accepted (should reject).")
        chan_ok = False
    except Exception as ex:
        msg = str(ex).splitlines()[0][:120]
        print(f"  REJECTED loudly (expected): {msg}")
        chan_ok = True

    print(f"\n=== FORMAT GATE {'PASS' if (ok and chan_ok) else 'FAIL'}: "
          f"block-FP8 round-trip cos>=0.99={ok}; channel-scale rejected={chan_ok} ===")
    sys.exit(0 if (ok and chan_ok) else 1)


if __name__ == "__main__":
    main()
