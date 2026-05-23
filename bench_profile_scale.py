"""
bench_profile_scale.py
──────────────────────
Time our FP8 W8A16 kernel vs torch.matmul (cuBLAS HMMA.884) at the exact
Linear shapes vLLM's profile_run exercises during init engine.

This isolates "GEMM time" from the rest of vLLM (Triton autotune, FLA Mamba,
NCCL coordination), so we can answer:

  Q: How much of the 14-min init engine is our slow CUDA-core kernel
     vs everything else?

Method:
  For each layer family in 27B-FP8 at TP=4, load the real per-rank weight
  shard and scale. Pre-dequant once into FP16 for the cuBLAS reference.
  Run a forward at M=max_num_tokens=4096 with both methods. Time it with
  CUDA events. Sum across all layer families × layer counts to estimate
  the total Linear-GEMM portion of one profile_run forward.

Run:
    ./run_docker.sh dev-test bench_profile_scale.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "safetensors"]
    )
    from safetensors import safe_open

import torch
from torch.utils.cpp_extension import load


HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/mnt/models/Qwen3.6-27B-FP8"))
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
# Profile_run uses max_num_tokens; for our typical config (max_model_len=4096,
# max_num_seqs=1, no chunked prefill) this is 4096.
M = int(os.environ.get("BENCH_M", "4096"))
N_WARMUP = 2
N_ITER = 5
BLOCK = 128

# Sweep mode: set BENCH_SWEEP=1 to run M-sweep across representative layers
# instead of the default full-model-at-fixed-M layout. Used to find the
# crossover M where cuBLAS (tensor cores) starts pulling away from our
# CUDA-core kernel — the regime where WMMA would actually help.
BENCH_SWEEP = os.environ.get("BENCH_SWEEP", "0") == "1"
SWEEP_MS = [1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# Dispatch thresholds matching FP8W8A16Linear in fp8_w8a16_module.py
DISPATCH_M_A3_K8 = 4
DISPATCH_M_A3_K4 = 8
DISPATCH_M_A2 = 64


def load_kernel():
    print("Compiling kernel ...", flush=True)
    ext = load(
        name="fp8_dequant_ext_bench",
        sources=[str(HERE / "fp8_dequant.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )
    print("Compiled.\n", flush=True)
    return ext


def fetch(wmap, name):
    with safe_open(MODEL_DIR / wmap[name], framework="pt") as f:
        return f.get_tensor(name)


def shard_col(t, tp_size, tp_rank):
    n = t.shape[0]
    per = n // tp_size
    return t[tp_rank * per:(tp_rank + 1) * per, :].contiguous()


def shard_row(t, tp_size, tp_rank):
    k = t.shape[1]
    per = k // tp_size
    return t[:, tp_rank * per:(tp_rank + 1) * per].contiguous()


def time_kernel(fn, n_iter=N_ITER, n_warmup=N_WARMUP):
    """Return mean kernel ms over n_iter runs after n_warmup warmups."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def dispatch_ours(ext, x, w_u8_flat, s_flat, N, K, M, block_h=BLOCK, block_w=BLOCK):
    """Pick the right kernel variant for this M, matching production module."""
    k_split_8_ok = (K % (8 * block_w)) == 0
    k_split_4_ok = (K % (4 * block_w)) == 0
    if M <= DISPATCH_M_A3_K8 and k_split_8_ok:
        return ("A.3k8",
                lambda: ext.fp8_w8a16_gemm_a3(x, w_u8_flat, s_flat, N, K, block_h, block_w, 8))
    if M <= DISPATCH_M_A3_K4 and k_split_4_ok:
        return ("A.3k4",
                lambda: ext.fp8_w8a16_gemm_a3(x, w_u8_flat, s_flat, N, K, block_h, block_w, 4))
    if M >= DISPATCH_M_A2:
        return ("A.2",
                lambda: ext.fp8_w8a16_gemm_a2(x, w_u8_flat, s_flat, N, K, block_h, block_w))
    return ("A.1",
            lambda: ext.fp8_w8a16_gemm_a1(x, w_u8_flat, s_flat, N, K, block_h, block_w))


def sweep_layer(ext, dev, label, w_fp8_shard, s_bf16_shard, ms):
    """Sweep M for one layer family. Compares dispatched kernel vs WMMA POC vs cuBLAS FP16."""
    N, K = w_fp8_shard.shape
    Nb, Kb = s_bf16_shard.shape
    s_fp16 = s_bf16_shard.to(torch.float16).contiguous().to(dev)
    w_fp8 = w_fp8_shard.to(dev)
    w_u8_flat = w_fp8.view(torch.uint8).reshape(-1).contiguous()
    s_flat = s_fp16.reshape(-1).contiguous()

    # Pre-dequant once for cuBLAS reference (no per-call dequant overhead).
    i_idx = (torch.arange(N, device=dev) // BLOCK).clamp_max(Nb - 1)
    j_idx = (torch.arange(K, device=dev) // BLOCK).clamp_max(Kb - 1)
    scale_grid = s_fp16[i_idx][:, j_idx]
    w_fp16 = (w_fp8.view(torch.float8_e4m3fn).to(torch.float16) * scale_grid).contiguous()

    have_wmma = hasattr(ext, "fp8_w8a16_gemm_wmma_poc")
    # WMMA POC limits: M, N, K all divisible by 64/64/16 and block_h=block_w=128.
    wmma_n_ok = (N % 64) == 0
    wmma_k_ok = (K % 16) == 0

    print(f"\n=== {label}  (per-rank N={N}, K={K}) ===")
    print(f"{'M':>6}  {'variant':>7}  {'ours (ms)':>10}  {'wmma (ms)':>10}  {'cuBLAS (ms)':>11}  {'wmma/ours':>9}  {'cuBLAS/wmma':>11}  {'TFLOP/s w':>10}  {'TFLOP/s cu':>10}")
    print("-" * 120)
    rows = []
    for m in ms:
        torch.manual_seed(0)
        x = (torch.randn(m, K, device=dev) * 0.1).to(torch.float16).contiguous()
        variant, call_ours = dispatch_ours(ext, x, w_u8_flat, s_flat, N, K, m)

        def call_cublas():
            return torch.matmul(x, w_fp16.t())

        ms_ours = time_kernel(call_ours)
        ms_cublas = time_kernel(call_cublas)

        # WMMA POC (only valid when M%64==0 along with N/K constraints)
        wmma_m_ok = (m % 64) == 0
        ms_wmma = None
        wmma_str = "    —    "
        wmma_ratio_str = "    —    "
        cublas_over_wmma_str = "    —    "
        tflops_wmma_str = "    —    "
        if have_wmma and wmma_m_ok and wmma_n_ok and wmma_k_ok:
            def call_wmma():
                return ext.fp8_w8a16_gemm_wmma_poc(x, w_u8_flat, s_flat, N, K, BLOCK, BLOCK)
            # Correctness check on first call for this layer/M
            try:
                out_wmma = call_wmma()
                ref = torch.matmul(x, w_fp16.t())
                max_abs = (out_wmma - ref).abs().max().item()
                ref_abs = ref.abs().max().item() + 1e-6
                rel = max_abs / ref_abs
                if rel > 0.10:
                    wmma_str = f"FAIL r={rel:.2g}"
                    print(f"{m:>6}  {variant:>7}  {ms_ours:>10.3f}  {wmma_str:>10}  {ms_cublas:>11.3f}  (correctness fail)",
                          flush=True)
                    rows.append((m, variant, ms_ours, None, ms_cublas, None, None))
                    continue
                ms_wmma = time_kernel(call_wmma)
                wmma_str = f"{ms_wmma:>10.3f}"
                wmma_ratio_str = f"{ms_ours/ms_wmma:>8.2f}x"
                cublas_over_wmma_str = f"{ms_wmma/ms_cublas:>10.2f}x"
                tflops_wmma = (2.0 * m * N * K) / (ms_wmma * 1e-3) / 1e12
                tflops_wmma_str = f"{tflops_wmma:>10.2f}"
            except RuntimeError as e:
                wmma_str = f"err: {str(e)[:20]}"

        tflops_cublas = (2.0 * m * N * K) / (ms_cublas * 1e-3) / 1e12
        print(f"{m:>6}  {variant:>7}  {ms_ours:>10.3f}  {wmma_str:>10}  {ms_cublas:>11.3f}  {wmma_ratio_str:>9}  {cublas_over_wmma_str:>11}  {tflops_wmma_str:>10}  {tflops_cublas:>10.2f}",
              flush=True)
        rows.append((m, variant, ms_ours, ms_wmma, ms_cublas, ms_ours / ms_wmma if ms_wmma else None,
                     ms_wmma / ms_cublas if ms_wmma else None))
    return rows


def bench_layer(ext, dev, label, w_fp8_shard, s_bf16_shard, layer_count, M):
    """Bench one layer family. Returns dict of mean ms per call for each method."""
    N, K = w_fp8_shard.shape
    Nb, Kb = s_bf16_shard.shape

    s_fp16 = s_bf16_shard.to(torch.float16).contiguous().to(dev)
    w_fp8 = w_fp8_shard.to(dev)
    w_u8_flat = w_fp8.view(torch.uint8).reshape(-1).contiguous()
    s_flat = s_fp16.reshape(-1).contiguous()

    # Pre-dequantize to FP16 once for the cuBLAS reference (no per-call
    # dequant overhead; we're asking "if you could call cuBLAS on FP16
    # weights directly, how fast would it be?")
    i_idx = (torch.arange(N, device=dev) // BLOCK).clamp_max(Nb - 1)
    j_idx = (torch.arange(K, device=dev) // BLOCK).clamp_max(Kb - 1)
    scale_grid = s_fp16[i_idx][:, j_idx]
    w_fp16 = (w_fp8.view(torch.float8_e4m3fn).to(torch.float16) * scale_grid).contiguous()

    torch.manual_seed(0)
    x = (torch.randn(M, K, device=dev) * 0.1).to(torch.float16)

    # ---- Our FP8 kernel (variant dispatch by M, same as _our_apply) ----
    def call_ours():
        if M >= 64:
            return ext.fp8_w8a16_gemm_a2(x, w_u8_flat, s_flat, N, K, BLOCK, BLOCK)
        else:
            return ext.fp8_w8a16_gemm_a1(x, w_u8_flat, s_flat, N, K, BLOCK, BLOCK)

    ms_ours = time_kernel(call_ours)
    variant = "A.2" if M >= 64 else "A.1"

    # ---- cuBLAS FP16 reference (torch.matmul; weight pre-dequant'd) ----
    def call_cublas():
        return torch.matmul(x, w_fp16.t())

    ms_cublas = time_kernel(call_cublas)

    # Total contribution to one profile_run forward
    tot_ours_s = ms_ours * layer_count / 1000.0
    tot_cublas_s = ms_cublas * layer_count / 1000.0

    print(f"  {label:<32} N={N:>5} K={K:>5} count={layer_count:>3}  "
          f"ours({variant}): {ms_ours:7.2f}ms  cublas: {ms_cublas:7.2f}ms  "
          f"speedup: {ms_ours/ms_cublas:5.2f}x  "
          f"layer-totals: ours={tot_ours_s:6.1f}s cublas={tot_cublas_s:6.1f}s",
          flush=True)
    return {"ms_ours": ms_ours, "ms_cublas": ms_cublas,
            "tot_ours_s": tot_ours_s, "tot_cublas_s": tot_cublas_s,
            "variant": variant}


def main():
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}  cap {torch.cuda.get_device_capability(0)}")
    print(f"Model:  {MODEL_DIR}")
    print(f"TP:     {TP_SIZE}")
    print(f"M:      {M}  (profile_run scale)")
    print()

    import json
    wmap = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    ext = load_kernel()

    # Fetch real weights from layer 3 (has self_attn) and layer 0 (has linear_attn).
    SA, LA, MLP = (
        "model.language_model.layers.3.self_attn",
        "model.language_model.layers.0.linear_attn",
        "model.language_model.layers.0.mlp",
    )
    sa_q = fetch(wmap, f"{SA}.q_proj.weight");          sa_q_s = fetch(wmap, f"{SA}.q_proj.weight_scale_inv")
    sa_k = fetch(wmap, f"{SA}.k_proj.weight");          sa_k_s = fetch(wmap, f"{SA}.k_proj.weight_scale_inv")
    sa_v = fetch(wmap, f"{SA}.v_proj.weight");          sa_v_s = fetch(wmap, f"{SA}.v_proj.weight_scale_inv")
    sa_o = fetch(wmap, f"{SA}.o_proj.weight");          sa_o_s = fetch(wmap, f"{SA}.o_proj.weight_scale_inv")
    la_qkv = fetch(wmap, f"{LA}.in_proj_qkv.weight");   la_qkv_s = fetch(wmap, f"{LA}.in_proj_qkv.weight_scale_inv")
    la_z = fetch(wmap, f"{LA}.in_proj_z.weight");       la_z_s = fetch(wmap, f"{LA}.in_proj_z.weight_scale_inv")
    la_o = fetch(wmap, f"{LA}.out_proj.weight");        la_o_s = fetch(wmap, f"{LA}.out_proj.weight_scale_inv")
    mlp_dn = fetch(wmap, f"{MLP}.down_proj.weight");    mlp_dn_s = fetch(wmap, f"{MLP}.down_proj.weight_scale_inv")
    mlp_gp = fetch(wmap, f"{MLP}.gate_proj.weight");    mlp_gp_s = fetch(wmap, f"{MLP}.gate_proj.weight_scale_inv")
    mlp_up = fetch(wmap, f"{MLP}.up_proj.weight");      mlp_up_s = fetch(wmap, f"{MLP}.up_proj.weight_scale_inv")

    # Per-rank shards at TP=4. Rank 0 is representative (others are symmetric).
    rank = 0

    # Building one fused qkv_proj shard: concat(q, k, v) along output dim,
    # each sharded by TP.
    qkv_w = torch.cat([
        shard_col(sa_q, TP_SIZE, rank),
        shard_col(sa_k, TP_SIZE, rank),
        shard_col(sa_v, TP_SIZE, rank),
    ], dim=0)
    qkv_s = torch.cat([
        shard_col(sa_q_s, TP_SIZE, rank),
        shard_col(sa_k_s, TP_SIZE, rank),
        shard_col(sa_v_s, TP_SIZE, rank),
    ], dim=0)

    # in_proj_qkvz: 4 partitions [key, key, value, value] from in_proj_qkv + in_proj_z
    KEY_DIM, VALUE_DIM = 2048, 6144
    qkvz_w = torch.cat([
        shard_col(la_qkv[0:KEY_DIM, :], TP_SIZE, rank),
        shard_col(la_qkv[KEY_DIM:2*KEY_DIM, :], TP_SIZE, rank),
        shard_col(la_qkv[2*KEY_DIM:2*KEY_DIM+VALUE_DIM, :], TP_SIZE, rank),
        shard_col(la_z, TP_SIZE, rank),
    ], dim=0)
    qkvz_s = torch.cat([
        shard_col(la_qkv_s[0:KEY_DIM//BLOCK, :], TP_SIZE, rank),
        shard_col(la_qkv_s[KEY_DIM//BLOCK:2*KEY_DIM//BLOCK, :], TP_SIZE, rank),
        shard_col(la_qkv_s[2*KEY_DIM//BLOCK:(2*KEY_DIM+VALUE_DIM)//BLOCK, :], TP_SIZE, rank),
        shard_col(la_z_s, TP_SIZE, rank),
    ], dim=0)

    # MLP gate_up_proj = concat(gate, up) along output dim
    gate_up_w = torch.cat([
        shard_col(mlp_gp, TP_SIZE, rank),
        shard_col(mlp_up, TP_SIZE, rank),
    ], dim=0)
    gate_up_s = torch.cat([
        shard_col(mlp_gp_s, TP_SIZE, rank),
        shard_col(mlp_up_s, TP_SIZE, rank),
    ], dim=0)

    # Layer counts per family in 27B model (64 total layers; 17 self-attn, 47 linear-attn).
    layer_counts = {
        "self_attn.qkv_proj":     17,
        "self_attn.o_proj":       17,
        "linear_attn.in_proj_qkvz": 47,
        "linear_attn.out_proj":   47,
        "mlp.gate_up_proj":       64,
        "mlp.down_proj":          64,
    }

    if BENCH_SWEEP:
        # Sweep mode: instead of layer-counted totals at fixed M, sweep M across
        # representative shapes to find the crossover where tensor cores matter.
        print(f"M sweep at TP={TP_SIZE}, rank 0 shards. Values: {SWEEP_MS}\n")
        all_rows = {}
        all_rows["self_attn.qkv_proj"]   = sweep_layer(ext, dev, "self_attn.qkv_proj  (col-parallel)", qkv_w, qkv_s, SWEEP_MS)
        all_rows["self_attn.o_proj"]     = sweep_layer(ext, dev, "self_attn.o_proj    (row-parallel)",
                                                       shard_row(sa_o, TP_SIZE, rank), shard_row(sa_o_s, TP_SIZE, rank), SWEEP_MS)
        all_rows["mlp.gate_up_proj"]     = sweep_layer(ext, dev, "mlp.gate_up_proj    (col-parallel)", gate_up_w, gate_up_s, SWEEP_MS)
        all_rows["mlp.down_proj"]        = sweep_layer(ext, dev, "mlp.down_proj       (row-parallel)",
                                                       shard_row(mlp_dn, TP_SIZE, rank), shard_row(mlp_dn_s, TP_SIZE, rank), SWEEP_MS)
        # Summary: WMMA POC speedup over current kernel, by layer
        # Row format: (m, variant, ms_ours, ms_wmma, ms_cublas, wmma_speedup_over_ours, wmma_over_cublas)
        print("\n" + "=" * 90)
        print("WMMA POC speedup over current kernel (key result):")
        print(f"  {'layer':<28} {'M=64':>8} {'M=128':>8} {'M=256':>8} {'M=512':>8} {'M=1024':>8} {'M=4096':>8}")
        for label, rows in all_rows.items():
            by_m = {m: r for (m, _, _, _, _, r, _) in rows if r is not None for m in [m]}
            cells = []
            for target_m in (64, 128, 256, 512, 1024, 4096):
                v = by_m.get(target_m)
                cells.append(f"{v:>7.2f}x" if v is not None else "    —   ")
            print(f"  {label:<28} " + " ".join(cells))
        print()
        print("Reading guide:")
        print("  - ≥ 5x speedup over current kernel at M=64-256: STRONG POC pass; commit to optimized version")
        print("  - 2-5x speedup                                : POC pass; layout/dequant tuning warranted")
        print("  - < 2x speedup                                : POC stuck; investigate (bank conflicts, occupancy, dequant cost)")
        print()
        print("WMMA-vs-cuBLAS gap (how much room is left to optimize):")
        print(f"  {'layer':<28} {'M=64':>10} {'M=128':>10} {'M=256':>10} {'M=512':>10} {'M=1024':>10} {'M=4096':>10}")
        for label, rows in all_rows.items():
            by_m_cu = {m: r for (m, _, _, _, _, _, r) in rows if r is not None for m in [m]}
            cells = []
            for target_m in (64, 128, 256, 512, 1024, 4096):
                v = by_m_cu.get(target_m)
                cells.append(f"{v:>8.2f}x slow" if v is not None else "       —    ")
            print(f"  {label:<28} " + " ".join(cells))
        return

    print(f"{'layer':<32} {'N':>5} {'K':>5} {'count':>5}  {'ours':>11}  {'cublas':>11}  speedup  layer-totals (s)")
    print("-" * 130)
    results = {}
    results["self_attn.qkv_proj"]       = bench_layer(ext, dev, "self_attn.qkv_proj", qkv_w, qkv_s, layer_counts["self_attn.qkv_proj"], M)
    results["self_attn.o_proj"]         = bench_layer_row(ext, dev, "self_attn.o_proj", sa_o, sa_o_s, TP_SIZE, rank, layer_counts["self_attn.o_proj"], M)
    results["linear_attn.in_proj_qkvz"] = bench_layer(ext, dev, "linear_attn.in_proj_qkvz", qkvz_w, qkvz_s, layer_counts["linear_attn.in_proj_qkvz"], M)
    results["linear_attn.out_proj"]     = bench_layer_row(ext, dev, "linear_attn.out_proj", la_o, la_o_s, TP_SIZE, rank, layer_counts["linear_attn.out_proj"], M)
    results["mlp.gate_up_proj"]         = bench_layer(ext, dev, "mlp.gate_up_proj", gate_up_w, gate_up_s, layer_counts["mlp.gate_up_proj"], M)
    results["mlp.down_proj"]            = bench_layer_row(ext, dev, "mlp.down_proj", mlp_dn, mlp_dn_s, TP_SIZE, rank, layer_counts["mlp.down_proj"], M)

    total_ours = sum(r["tot_ours_s"] for r in results.values())
    total_cublas = sum(r["tot_cublas_s"] for r in results.values())

    print()
    print("=" * 60)
    print(f"TOTAL Linear-GEMM time per profile_run forward at M={M}:")
    print(f"  Our FP8 kernel:  {total_ours:7.1f} s")
    print(f"  cuBLAS FP16:     {total_cublas:7.1f} s")
    print(f"  Speedup if we switched to cuBLAS: {total_ours/total_cublas:.1f}x")
    print()
    print("Compare to observed init engine times:")
    print("  Our FP8 init engine (with cache mount):  ~14 min = 840 s")
    print("  GPTQ init engine (cold):                  ~112 s")
    print()
    print("If our GEMM-only total ≈ (FP8 init engine - GPTQ init engine - Triton autotune ~60s),")
    print("then 'slow GEMM' is the dominant cost. Otherwise, autotune/Mamba/other dominates.")


def bench_layer_row(ext, dev, label, weight_full, scale_full, tp_size, tp_rank, layer_count, M):
    """Bench row-parallel layer (split K)."""
    w_shard = shard_row(weight_full, tp_size, tp_rank)
    s_shard = shard_row(scale_full, tp_size, tp_rank)
    return bench_layer(ext, dev, label, w_shard, s_shard, layer_count, M)


def shard_row_pair(w, s, tp_size, tp_rank):
    return shard_row(w, tp_size, tp_rank), shard_row(s, tp_size, tp_rank)


if __name__ == "__main__":
    main()
