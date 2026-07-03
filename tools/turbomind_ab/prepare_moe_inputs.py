#!/usr/bin/env python3
"""
Stage-C grouped-MoE A/B :: canonical input freezer (byte-identical for both engines).

Freezes, for one model's MoE geometry:
  - E experts of block-128 FP8 weights: w13 [E,2I,H], w2 [E,H,I] (+ fp32 block scales),
  - per (regime, tpe) routing eids[R] + activations x13[R,H] fp16,
  - a kernel-neutral fp32 reference y_ref[R,H] = full MoE chain
        gate_up = x13 @ dequant(w13[e]).T ; inter = silu(gate_up[:I])*gate_up[I:] ;
        y = inter @ dequant(w2[e]).T
so that bench_moe_ours.py (coalesced GEMV) and bench_moe_1catai.py (TurboMind grouped)
consume identical bytes and are judged against the SAME reference.

Weights are frozen to disk (not seed-regenerated) to guarantee byte-identity across the
two docker images / torch versions. Real GLM-4.5-Air-FP8 is channel-scale W8A8; here we use
synthetic block-128 weights at its true dims — a valid *kernel* comparison (shape-driven),
flagged as a Stage-D format gap (1catai grouped MoE requires block-128).

Run once in any torch env (only randn + one matmul). Outputs to moe_inputs_<model>/.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import torch
import torch.nn.functional as F

MODELS = {
    "qwen35_a3b": dict(E=256, top_k=8, H=2048, I=512),   # native block-128 FP8
    "glm45_air":  dict(E=128, top_k=8, H=4096, I=1408),  # real ckpt = channel W8A8 (see note)
}
REGIMES = ("spread", "hot1", "hot8")
BLOCK = 128


def clean_gpu_guard(force):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        return
    mypid = str(os.getpid())
    others = [l for l in out.splitlines() if l.strip()
              and l.split(",")[0].strip() != mypid]
    if others and not force:
        print("[GUARD] Other GPU compute processes present:")
        for l in others:
            print("   ", l)
        sys.exit("[GUARD] Refusing to run on shared GPU. Re-run with --force.")


def synth_expert_weights(E, N, K, block, seed, dev):
    """[E,N,K] fp8 + [E,N/block,K/block] fp32 scales, deterministic per (seed,E,N,K)."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(E, N, K, generator=g) * 0.2).clamp(-6, 6)
    W_fp8 = w.to(torch.float8_e4m3fn)
    scales = (torch.rand(E, N // block, K // block, generator=g) * 0.5 + 0.5).float()
    return W_fp8.to(dev), scales.to(dev)


def dequant_e(W_fp8_e, scales_e, block):
    """One expert: [N,K] fp8 * block scale -> fp32 [N,K]."""
    N, K = W_fp8_e.shape
    s = scales_e.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return W_fp8_e.float() * s


def routing(regime, E, tpe):
    """Per-row expert ids eids[R] for a routing regime at tokens-per-expert=tpe."""
    if regime == "spread":
        active = E
    elif regime == "hot8":
        active = min(8, E)
    elif regime == "hot1":
        active = 1
    else:
        raise ValueError(regime)
    R = active * tpe
    eids = (torch.arange(R, dtype=torch.int64) % active)
    return eids, R, active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tpe", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    dev = "cuda:0"
    clean_gpu_guard(args.force)

    d = MODELS[args.model]
    E, H, I, top_k = d["E"], d["H"], d["I"], d["top_k"]
    N13, K13 = 2 * I, H          # w13: [2I, H]
    N2,  K2  = H, I              # w2:  [H, I]
    for x, nm in [(N13, "2I"), (K13, "H"), (N2, "H"), (K2, "I")]:
        assert x % BLOCK == 0, f"{nm}={x} not mult of {BLOCK}"

    out = Path(args.out or f"moe_inputs_{args.model}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] {args.model}: E={E} top_k={top_k} H={H} I={I} "
          f"w13=[{N13},{K13}] w2=[{N2},{K2}]")
    w13, w13_s = synth_expert_weights(E, N13, K13, BLOCK, args.seed + 1, dev)
    w2,  w2_s  = synth_expert_weights(E, N2,  K2,  BLOCK, args.seed + 2, dev)
    torch.save({"w13": w13.cpu(), "w13_scale": w13_s.cpu(),
                "w2": w2.cpu(), "w2_scale": w2_s.cpu(),
                "E": E, "H": H, "I": I, "top_k": top_k,
                "N13": N13, "K13": K13, "N2": N2, "K2": K2, "block": BLOCK},
               out / "weights.pt")

    # pre-dequant experts once (fp32) for the reference
    w13_dq = [dequant_e(w13[e], w13_s[e], BLOCK) for e in range(E)]
    w2_dq  = [dequant_e(w2[e],  w2_s[e],  BLOCK) for e in range(E)]

    manifest = []
    for regime in REGIMES:
        for tpe in args.tpe:
            eids, R, active = routing(regime, E, tpe)
            g = torch.Generator().manual_seed(args.seed + 1000 * tpe + hash(regime) % 997)
            x13 = (torch.randn(R, K13, generator=g) * 0.1).to(torch.float16).to(dev)
            y_ref = torch.empty(R, N2, dtype=torch.float32, device=dev)
            for r in range(R):
                e = int(eids[r])
                gate_up = x13[r].float() @ w13_dq[e].T            # [2I]
                inter = F.silu(gate_up[:I]) * gate_up[I:]         # [I]
                y_ref[r] = inter @ w2_dq[e].T                     # [H]
            tag = f"{regime}_tpe{tpe}"
            torch.save({"eids": eids, "x13": x13.cpu(), "y_ref": y_ref.cpu(),
                        "regime": regime, "tpe": tpe, "R": R, "active": active},
                       out / f"route_{tag}.pt")
            manifest.append(dict(tag=tag, regime=regime, tpe=tpe, R=R, active=active))
            print(f"  froze {tag:14s} R={R:5d} active={active}")

    (out / "meta.json").write_text(json.dumps(
        {"model": args.model, **d, "N13": N13, "K13": K13, "N2": N2, "K2": K2,
         "block": BLOCK, "seed": args.seed, "configs": manifest,
         "note": "synthetic block-128 weights at real MoE dims; "
                 "GLM-4.5-Air real ckpt is channel W8A8 (Stage-D format gap)"}, indent=2))
    print(f"[ok] -> {out}/  ({len(manifest)} configs)")


if __name__ == "__main__":
    main()
