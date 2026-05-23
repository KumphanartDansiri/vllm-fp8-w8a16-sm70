"""
test_attention_shapes.py
────────────────────────
Offline kernel-vs-reference test for the exact attention/linear_attn shapes
the 27B-FP8 model exercises at TP=4. Skips the 14-min serve cycle.

Strategy:
  1. Load real weights+scales from /mnt/models/Qwen3.6-27B-FP8 for each
     suspect layer family (self_attn.{q,k,v,o}_proj,
     linear_attn.{in_proj_qkv, in_proj_z, out_proj}) plus an mlp.down_proj
     baseline that's known-working.
  2. Replicate vLLM's TP=4 sharding offline — column-parallel for
     Q/K/V/in_proj_qkvz; row-parallel for o_proj/out_proj/down_proj. For
     merged layers (qkv_proj, in_proj_qkvz) concatenate per-rank partitions
     just as MergedColumnParallelLinear does.
  3. Run our kernel (A.1/A.2/A.3 dispatch by M) on each shard at several
     M values, compare against PyTorch FP16 dequant-matmul reference.
  4. Report per-(layer, rank, M, variant): max_abs_diff vs ref, NaN/Inf
     counts, and scale-finite stats.

Any layer-family-shape where every rank+M passes → kernel is correct on
that shape. Any failure isolates the bug to that shape combination.

Run:
    ./run_docker.sh attn-test
"""
import os
import subprocess
import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "safetensors"]
    )
    from safetensors import safe_open

import json
import torch
from torch.utils.cpp_extension import load


HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/mnt/models/Qwen3.6-27B-FP8"))
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
BLOCK_N = 128
BLOCK_K = 128
TOL_ABS = float(os.environ.get("TOL_ABS", "5e-2"))   # FP16 + FP8-dequant noise


# ─── kernel ────────────────────────────────────────────────────────────────
def load_kernel():
    print("Compiling kernel for sm_70 ...", flush=True)
    ext = load(
        name="fp8_dequant_ext_attn_test",
        sources=[str(HERE / "fp8_dequant.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )
    print("Compiled OK.\n", flush=True)
    return ext


# ─── safetensors helpers ───────────────────────────────────────────────────
def load_index(model_dir: Path) -> dict[str, str]:
    """name -> safetensors-file containing it."""
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    return idx["weight_map"]


def fetch(model_dir: Path, wmap: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(model_dir / wmap[name], framework="pt") as f:
        return f.get_tensor(name)


# ─── vLLM-equivalent sharding ──────────────────────────────────────────────
def shard_col_parallel(W: torch.Tensor, S: torch.Tensor,
                       tp_size: int, tp_rank: int):
    """For ColumnParallel: split along output dim 0. W=[N,K], S=[Nb,Kb]."""
    N = W.shape[0]
    assert N % tp_size == 0, f"N={N} not divisible by tp_size={tp_size}"
    n_per = N // tp_size
    Nb_per = S.shape[0] // tp_size
    return (W[tp_rank * n_per:(tp_rank + 1) * n_per, :].contiguous(),
            S[tp_rank * Nb_per:(tp_rank + 1) * Nb_per, :].contiguous())


def shard_row_parallel(W: torch.Tensor, S: torch.Tensor,
                       tp_size: int, tp_rank: int):
    """For RowParallel: split along input dim 1. W=[N,K], S=[Nb,Kb]."""
    K = W.shape[1]
    assert K % tp_size == 0, f"K={K} not divisible by tp_size={tp_size}"
    k_per = K // tp_size
    Kb_per = S.shape[1] // tp_size
    return (W[:, tp_rank * k_per:(tp_rank + 1) * k_per].contiguous(),
            S[:, tp_rank * Kb_per:(tp_rank + 1) * Kb_per].contiguous())


def fuse_qkv_merged(parts_w: list[torch.Tensor], parts_s: list[torch.Tensor]):
    """Concatenate per-partition shards along output dim 0, matching
    MergedColumnParallelLinear's in-memory layout for the fused param."""
    return torch.cat(parts_w, dim=0).contiguous(), torch.cat(parts_s, dim=0).contiguous()


# ─── reference dequant-matmul ──────────────────────────────────────────────
def reference_matmul(x_fp16: torch.Tensor, w_fp8: torch.Tensor,
                     s_fp16: torch.Tensor, block_h: int, block_w: int):
    """Reference: dequant w*s elementwise to FP16, then x @ w_fp16.T.
    All math in FP32 for fairness; cast result to FP16 to match kernel output dtype."""
    N, K = w_fp8.shape
    Nb, Kb = s_fp16.shape
    # Convert FP8-E4M3 bytes -> FP16 using torch's native FP8 dtype.
    w_view = w_fp8.view(torch.float8_e4m3fn)
    w_fp16 = w_view.to(torch.float16)
    # Per-(i, j) scale = s[i // block_h, j // block_w]
    i_idx = (torch.arange(N, device=w_fp16.device) // block_h).clamp_max(Nb - 1)
    j_idx = (torch.arange(K, device=w_fp16.device) // block_w).clamp_max(Kb - 1)
    scale_grid = s_fp16[i_idx][:, j_idx]   # [N, K]
    w_dequant_f32 = w_fp16.float() * scale_grid.float()
    out_f32 = x_fp16.float() @ w_dequant_f32.T
    return out_f32.to(torch.float16), w_dequant_f32


def kernel_dispatch(ext, x, w_flat_u8, s_flat, N, K, block_h, block_w):
    M = x.size(0)
    def ok(k_split):
        return (K % (k_split * block_w)) == 0
    if M <= 4 and ok(8):
        return ext.fp8_w8a16_gemm_a3(x, w_flat_u8, s_flat, N, K, block_h, block_w, 8), "A.3 k=8"
    if M <= 8 and ok(4):
        return ext.fp8_w8a16_gemm_a3(x, w_flat_u8, s_flat, N, K, block_h, block_w, 4), "A.3 k=4"
    if M >= 64:
        return ext.fp8_w8a16_gemm_a2(x, w_flat_u8, s_flat, N, K, block_h, block_w), "A.2"
    return ext.fp8_w8a16_gemm_a1(x, w_flat_u8, s_flat, N, K, block_h, block_w), "A.1"


def stat(tag, t):
    f = torch.isfinite(t)
    finite = int(f.sum())
    total = t.numel()
    return (f"{tag}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"finite={finite}/{total} "
            f"min={t[f].min().item() if finite else float('nan'):.3e} "
            f"max={t[f].max().item() if finite else float('nan'):.3e}")


# ─── per-layer test runner ────────────────────────────────────────────────
def test_layer(ext, dev, label, w_fp8_shard, s_bf16_shard, M_values,
               max_print_rows=6):
    """w_fp8_shard, s_bf16_shard: PER-RANK shards (CPU or GPU).
    Casts scale BF16 → FP16 exactly as serve_fp8_v100.py does.
    Returns dict[M] -> (max_abs_diff, variant, n_nonfinite)."""
    N, K = w_fp8_shard.shape
    Nb, Kb = s_bf16_shard.shape

    # Scale finite-check pre-cast and post-cast
    s_pre = s_bf16_shard.float()
    finite_pre = int(torch.isfinite(s_pre).sum())
    s_fp16 = s_bf16_shard.to(torch.float16)
    finite_post = int(torch.isfinite(s_fp16).sum())
    cast_err = (s_pre - s_fp16.float()).abs().max().item()

    print(f"  [{label}] weight=({N},{K}) scale=({Nb},{Kb}) "
          f"eff_block=[{N//Nb},{K//Kb}] finite_bf16={finite_pre}/{Nb*Kb} "
          f"finite_fp16={finite_post}/{Nb*Kb} "
          f"max_bf16_to_fp16_err={cast_err:.3e}")

    if finite_post < Nb * Kb:
        # Any non-finite scale post-cast is the bug — short-circuit.
        nf_mask = ~torch.isfinite(s_fp16)
        bad = torch.nonzero(nf_mask)[:max_print_rows]
        print(f"    !! NON-FINITE SCALES POST-CAST. Sample positions: {bad.tolist()}")
        print(f"    !! Pre-cast values at those positions: "
              f"{s_pre[nf_mask][:max_print_rows].tolist()}")
        return {"scale_nonfinite": True}

    w_flat_u8 = w_fp8_shard.view(torch.uint8).reshape(-1).to(dev).contiguous()
    s_flat = s_fp16.reshape(-1).to(dev).contiguous()
    w_full = w_fp8_shard.to(dev)

    results = {}
    for M in M_values:
        torch.manual_seed(M)
        x = (torch.randn(M, K, device=dev) * 0.1).to(torch.float16)
        out_ours, variant = kernel_dispatch(ext, x, w_flat_u8, s_flat,
                                            N, K, BLOCK_N, BLOCK_K)
        out_ref, _ = reference_matmul(x, w_full, s_fp16.to(dev), BLOCK_N, BLOCK_K)
        diff = (out_ours.float() - out_ref.float()).abs()
        finite = torch.isfinite(out_ours).all().item()
        max_abs = diff.max().item() if finite else float("inf")
        nf_count = int((~torch.isfinite(out_ours)).sum())
        verdict = "PASS" if (finite and max_abs < TOL_ABS) else "FAIL"
        print(f"    M={M:>3d} variant={variant:<8s} "
              f"max_abs_diff={max_abs:.3e} nonfinite_in_out={nf_count} "
              f"{verdict}")
        results[M] = (max_abs, variant, nf_count, finite)
    return results


# ─── test plan ─────────────────────────────────────────────────────────────
def main():
    assert torch.cuda.is_available(), "Need a V100 (sm_70) to run the kernel."
    dev = torch.device("cuda:0")
    cap = torch.cuda.get_device_capability(0)
    print(f"Device: {torch.cuda.get_device_name(0)} (cap {cap}) — TP_SIZE={TP_SIZE}")
    print(f"Model:  {MODEL_DIR}")
    if cap != (7, 0):
        print(f"  warning: this kernel targets sm_70; running on {cap}")
    print()

    wmap = load_index(MODEL_DIR)
    ext = load_kernel()
    M_values = [1, 4, 8, 32, 128]

    # ---------- pick representative layers ----------
    # Layer 3 has self_attn; layer 0 has linear_attn; both have mlp.
    SA_L = "model.language_model.layers.3"
    LA_L = "model.language_model.layers.0"

    sa_q_w  = fetch(MODEL_DIR, wmap, f"{SA_L}.self_attn.q_proj.weight")
    sa_q_s  = fetch(MODEL_DIR, wmap, f"{SA_L}.self_attn.q_proj.weight_scale_inv")
    sa_k_w  = fetch(MODEL_DIR, wmap, f"{SA_L}.self_attn.k_proj.weight")
    sa_k_s  = fetch(MODEL_DIR, wmap, f"{SA_L}.self_attn.k_proj.weight_scale_inv")
    sa_v_w  = fetch(MODEL_DIR, wmap, f"{SA_L}.self_attn.v_proj.weight")
    sa_v_s  = fetch(MODEL_DIR, wmap, f"{SA_L}.self_attn.v_proj.weight_scale_inv")
    sa_o_w  = fetch(MODEL_DIR, wmap, f"{SA_L}.self_attn.o_proj.weight")
    sa_o_s  = fetch(MODEL_DIR, wmap, f"{SA_L}.self_attn.o_proj.weight_scale_inv")

    la_qkv_w  = fetch(MODEL_DIR, wmap, f"{LA_L}.linear_attn.in_proj_qkv.weight")
    la_qkv_s  = fetch(MODEL_DIR, wmap, f"{LA_L}.linear_attn.in_proj_qkv.weight_scale_inv")
    la_z_w    = fetch(MODEL_DIR, wmap, f"{LA_L}.linear_attn.in_proj_z.weight")
    la_z_s    = fetch(MODEL_DIR, wmap, f"{LA_L}.linear_attn.in_proj_z.weight_scale_inv")
    la_out_w  = fetch(MODEL_DIR, wmap, f"{LA_L}.linear_attn.out_proj.weight")
    la_out_s  = fetch(MODEL_DIR, wmap, f"{LA_L}.linear_attn.out_proj.weight_scale_inv")

    mlp_dn_w = fetch(MODEL_DIR, wmap, f"{LA_L}.mlp.down_proj.weight")
    mlp_dn_s = fetch(MODEL_DIR, wmap, f"{LA_L}.mlp.down_proj.weight_scale_inv")

    print(f"Shapes:\n"
          f"  q_proj      {sa_q_w.shape}  scale {sa_q_s.shape}\n"
          f"  k_proj      {sa_k_w.shape}  scale {sa_k_s.shape}\n"
          f"  v_proj      {sa_v_w.shape}  scale {sa_v_s.shape}\n"
          f"  o_proj      {sa_o_w.shape}  scale {sa_o_s.shape}\n"
          f"  in_proj_qkv {la_qkv_w.shape}  scale {la_qkv_s.shape}\n"
          f"  in_proj_z   {la_z_w.shape}  scale {la_z_s.shape}\n"
          f"  out_proj    {la_out_w.shape}  scale {la_out_s.shape}\n"
          f"  mlp.down    {mlp_dn_w.shape}  scale {mlp_dn_s.shape}\n")

    # ---------- run TP=4 ----------
    all_results = {}
    for rank in range(TP_SIZE):
        print(f"\n══════════════ rank={rank} of TP={TP_SIZE} ══════════════")

        # mlp.down_proj — row-parallel, baseline known-working.
        w, s = shard_row_parallel(mlp_dn_w, mlp_dn_s, TP_SIZE, rank)
        all_results[(rank, "mlp.down_proj")] = test_layer(
            ext, dev, "mlp.down_proj", w, s, M_values)

        # self_attn.qkv_proj — merged column-parallel from q/k/v.
        q_w, q_s = shard_col_parallel(sa_q_w, sa_q_s, TP_SIZE, rank)
        k_w, k_s = shard_col_parallel(sa_k_w, sa_k_s, TP_SIZE, rank)
        v_w, v_s = shard_col_parallel(sa_v_w, sa_v_s, TP_SIZE, rank)
        qkv_w, qkv_s = fuse_qkv_merged([q_w, k_w, v_w], [q_s, k_s, v_s])
        all_results[(rank, "self_attn.qkv_proj(fused)")] = test_layer(
            ext, dev, "self_attn.qkv_proj(fused)", qkv_w, qkv_s, M_values)

        # self_attn.o_proj — row-parallel.
        w, s = shard_row_parallel(sa_o_w, sa_o_s, TP_SIZE, rank)
        all_results[(rank, "self_attn.o_proj")] = test_layer(
            ext, dev, "self_attn.o_proj", w, s, M_values)

        # linear_attn.in_proj_qkvz — 4-partition merged column-parallel
        # (Q, K, V from in_proj_qkv as 3 partitions at offsets [0, key_dim,
        # 2*key_dim] within the 10240-row tensor; Z from in_proj_z as the
        # 4th partition).
        # output_sizes = [key_dim, key_dim, value_dim, value_dim]
        #              = [2048, 2048, 6144, 6144] for this model.
        key_dim = 2048
        value_dim = 6144
        # Split in_proj_qkv along output dim into [Q=key_dim, K=key_dim, V=value_dim]
        qp1 = la_qkv_w[0:key_dim, :]
        qp2 = la_qkv_w[key_dim:2*key_dim, :]
        qp3 = la_qkv_w[2*key_dim:2*key_dim + value_dim, :]
        sp1 = la_qkv_s[0:key_dim // BLOCK_N, :]
        sp2 = la_qkv_s[key_dim // BLOCK_N:2*key_dim // BLOCK_N, :]
        sp3 = la_qkv_s[2*key_dim // BLOCK_N:(2*key_dim + value_dim) // BLOCK_N, :]
        # Then shard each partition (column-parallel)
        qp1_w, qp1_s = shard_col_parallel(qp1, sp1, TP_SIZE, rank)
        qp2_w, qp2_s = shard_col_parallel(qp2, sp2, TP_SIZE, rank)
        qp3_w, qp3_s = shard_col_parallel(qp3, sp3, TP_SIZE, rank)
        # Z is its own partition
        z_w, z_s = shard_col_parallel(la_z_w, la_z_s, TP_SIZE, rank)
        # Fuse all 4 as MergedColumnParallelLinear does in-memory
        qkvz_w, qkvz_s = fuse_qkv_merged(
            [qp1_w, qp2_w, qp3_w, z_w], [qp1_s, qp2_s, qp3_s, z_s])
        all_results[(rank, "linear_attn.in_proj_qkvz(fused)")] = test_layer(
            ext, dev, "linear_attn.in_proj_qkvz(fused)", qkvz_w, qkvz_s, M_values)

        # linear_attn.out_proj — row-parallel.
        w, s = shard_row_parallel(la_out_w, la_out_s, TP_SIZE, rank)
        all_results[(rank, "linear_attn.out_proj")] = test_layer(
            ext, dev, "linear_attn.out_proj", w, s, M_values)

    # ---------- summary ----------
    print("\n══════════════ SUMMARY ══════════════")
    print(f"{'Layer':<35} {'rank':<5} {'M':<4} {'variant':<10} {'max_abs_diff':>14} {'verdict':<6}")
    for (rank, label), res in all_results.items():
        if isinstance(res, dict) and res.get("scale_nonfinite"):
            print(f"{label:<35} {rank:<5} {'-':<4} {'-':<10} {'-':>14} SCALE-NONFINITE")
            continue
        for M, (max_abs, variant, nf, finite) in res.items():
            verdict = "PASS" if (finite and max_abs < TOL_ABS) else "FAIL"
            print(f"{label:<35} {rank:<5} {M:<4} {variant:<10} {max_abs:14.3e} {verdict}")


if __name__ == "__main__":
    main()
