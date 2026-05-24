"""
Phase A.2 A/B/C test: naive vs A.1 (vec) vs A.2 (vec + M-tiling).

For each M in {1, 4, 8, 32, 128}:
  - Run all three kernels on the same inputs
  - Confirm A.2 matches reference within FP16 noise
  - Note: A.2 will NOT be bit-equal to naive/A.1 because BLOCK_M-wise
    accumulation order changes. Within FP16 noise is the bar.
  - Report timings and speedups

Run:
    ./run_docker.sh a2
"""
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("Installing safetensors...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "safetensors"]
    )
    from safetensors import safe_open

import torch
from torch.utils.cpp_extension import load


HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/models/Qwen3.5-4B-FP8")


def load_kernel():
    print("Compiling kernel for sm_70 ...")
    ext = load(
        name="fp8_dequant_ext",
        sources=[str(HERE / "fp8_dequant.cu")],
        extra_cuda_cflags=[
            "-O3", "-gencode=arch=compute_70,code=sm_70", "--use_fast_math",
        ],
        extra_cflags=["-O3"],
        verbose=False,
    )
    print("Compiled OK.\n")
    return ext


def pick_fp8_weight(model_dir):
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())
            for k in sorted(keys):
                sl = f.get_slice(k)
                if sl.get_dtype() != "F8_E4M3" or not k.endswith(".weight"):
                    continue
                for sk in (k + "_scale_inv", k.replace(".weight", ".weight_scale_inv")):
                    if sk in keys:
                        return path, k, sk
    sys.exit("No FP8 weight + scale_inv found")


def timed(fn, *args, n_warmup=3, n_iter=10):
    for _ in range(n_warmup):
        out = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_iter, out


def main():
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability(0)})")

    ext = load_kernel()
    st_path, w_key, s_key = pick_fp8_weight(MODEL_DIR)
    print(f"Using file:   {st_path.name}")
    print(f"Using weight: {w_key}")

    with safe_open(st_path, framework="pt") as f:
        W_fp8 = f.get_tensor(w_key)
        scales_raw = f.get_tensor(s_key)
    N, K = W_fp8.shape
    block_h = N // scales_raw.shape[0]
    block_w = K // scales_raw.shape[1]
    print(f"Weight shape: [{N}, {K}], scale blocks: [{block_h}, {block_w}]\n")

    fp8_bytes = W_fp8.view(torch.uint8).reshape(-1).to(dev).contiguous()
    scales    = scales_raw.to(torch.float16).reshape(-1).to(dev).contiguous()

    # Reference dequantized weight (for FP32 matmul ground truth)
    W_dq_f32 = ext.fp8_e4m3_to_fp16_block_scaled(
        fp8_bytes, scales, N, K, block_h, block_w
    ).reshape(N, K).float()

    header = (f"{'M':>4} | {'naive':>8} | {'A.1':>7} | {'A.2':>7} | "
              f"{'A.1/naive':>9} | {'A.2/naive':>9} | {'A.2/A.1':>7} | "
              f"{'A.2 abs vs ref':>14}")
    print(header)
    print("-" * len(header))

    for M in [1, 4, 8, 32, 128]:
        torch.manual_seed(M)
        A = (torch.randn(M, K, device=dev) * 0.1).to(torch.float16)

        t_naive, C_naive = timed(
            ext.fp8_w8a16_gemm,    A, fp8_bytes, scales, N, K, block_h, block_w)
        t_a1, C_a1       = timed(
            ext.fp8_w8a16_gemm_a1, A, fp8_bytes, scales, N, K, block_h, block_w)
        t_a2, C_a2       = timed(
            ext.fp8_w8a16_gemm_a2, A, fp8_bytes, scales, N, K, block_h, block_w)

        sp_a1   = t_naive / t_a1
        sp_a2   = t_naive / t_a2
        sp_a2_1 = t_a1    / t_a2

        # A.2 will differ from A.1/naive in last few bits because the FMA
        # ordering across rows changes — check FP16-noise tolerance instead.
        ref_f32   = A.float() @ W_dq_f32.T
        diff_a2   = (C_a2.float() - ref_f32).abs()
        max_abs   = diff_a2.max().item()
        ref_thresh= 0.01 * ref_f32.abs().max().item()
        mask      = ref_f32.abs() > ref_thresh
        max_rel   = (diff_a2[mask] / ref_f32[mask].abs()).max().item() if mask.any() else 0.0
        ok = max_abs <= 1e-2

        print(f"{M:>4d} | {t_naive:>6.2f}ms | {t_a1:>5.2f}ms | {t_a2:>5.2f}ms | "
              f"{sp_a1:>8.2f}x | {sp_a2:>8.2f}x | {sp_a2_1:>6.2f}x | "
              f"{max_abs:>8.2e}  {'PASS' if ok else 'FAIL'}")

    print("\nNotes:")
    print("  - A.2 changes FMA accumulation order (across M rows) vs A.1, so")
    print("    bit-exact match is NOT expected. FP16-noise correctness IS.")
    print("  - At M=1, A.2 should be ~equal to A.1 (only 1 M row to tile).")
    print("  - At M >= 8, A.2 should show large speedup over A.1 due to W-read amortization.")


if __name__ == "__main__":
    main()
