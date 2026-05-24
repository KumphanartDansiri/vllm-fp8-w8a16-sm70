"""
Phase A.1 A/B test: naive `fp8_w8a16_gemm` vs vectorized `fp8_w8a16_gemm_a1`.

For each M in {1, 4, 32, 128}:
  - Run both kernels on the same inputs
  - Confirm output bit-equality (they should match exactly — only the W load
    width changed, math is identical)
  - Confirm correctness vs PyTorch reference
  - Report time, speedup, achieved memory bandwidth

Run:
    ./run_docker.sh a1
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
from fp8_w8a16_sm70.ext_loader import load_kernel as _load_kernel


HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/models/Qwen3.5-4B-FP8")


def load_kernel():
    print("Compiling kernel for sm_70 ...")
    ext = _load_kernel(name="fp8_dequant_ext")
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
    """Return (mean_ms, output) by averaging n_iter timed runs after warmup."""
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

    # Phase 4 dequant as the reference weight, used to build the matmul reference.
    W_dq_f32 = ext.fp8_e4m3_to_fp16_block_scaled(
        fp8_bytes, scales, N, K, block_h, block_w
    ).reshape(N, K).float()

    print(f"{'M':>4} | {'naive (ms)':>11} | {'A.1 (ms)':>9} | {'speedup':>7} | "
          f"{'kernels match':>13} | {'A.1 vs ref':>10}")
    print("-" * 80)

    for M in [1, 4, 32, 128]:
        torch.manual_seed(M)
        A = (torch.randn(M, K, device=dev) * 0.1).to(torch.float16)

        # Time both kernels.
        t_naive, C_naive = timed(
            ext.fp8_w8a16_gemm, A, fp8_bytes, scales, N, K, block_h, block_w
        )
        t_a1, C_a1 = timed(
            ext.fp8_w8a16_gemm_a1, A, fp8_bytes, scales, N, K, block_h, block_w
        )
        speedup = t_naive / t_a1

        # Kernels should match exactly — same math, different W-load width.
        kern_bit_match = int((C_naive.view(torch.int16) ==
                              C_a1.view(torch.int16)).sum().item())
        kern_total     = C_naive.numel()
        kernels_eq     = (kern_bit_match == kern_total)

        # A.1 vs PyTorch reference matmul (FP16 noise floor).
        ref_f32 = A.float() @ W_dq_f32.T
        diff = (C_a1.float() - ref_f32).abs()
        max_abs = diff.max().item()
        ref_thresh = 0.01 * ref_f32.abs().max().item()
        mask = ref_f32.abs() > ref_thresh
        max_rel = (diff[mask] / ref_f32[mask].abs()).max().item() if mask.any() else 0.0
        ref_ok = max_abs <= 1e-2

        # Memory bandwidth estimate: per A.1 invocation, the W tensor is read
        # once per M row (no M-tiling yet), so total bytes touched is
        # ~ M*N*K (W) + N*K*2 (output, neglected) + M*K*2 (A read, cached after
        # first chunk). Dominant term is M*N*K bytes.
        w_bytes = M * N * K
        bw_gbs = w_bytes / (t_a1 * 1e-3) / 1e9

        print(f"{M:>4d} | {t_naive:>11.3f} | {t_a1:>9.3f} | "
              f"{speedup:>6.2f}x | "
              f"{kern_bit_match}/{kern_total} {'OK' if kernels_eq else 'NEQ'} | "
              f"{'PASS' if ref_ok else 'FAIL'} "
              f"(abs={max_abs:.1e})")
        if not kernels_eq:
            # Surface a few mismatches for debugging.
            d = (C_naive.view(torch.int16) ^ C_a1.view(torch.int16)).nonzero()
            print(f"     first kernel mismatch idx: {d[0].tolist() if len(d) else 'none'}")

    # Print a coarse interpretation.
    print("\nNotes:")
    print("  - 'kernels match' should be 100% — A.1 only changes W load width, not math.")
    print("  - 'A.1 vs ref' is the same FP16-matmul bar as Phase 5 (max_abs <= 1e-2).")
    print("  - Memory BW shown reflects W reads only (per current naive model).")


if __name__ == "__main__":
    main()
