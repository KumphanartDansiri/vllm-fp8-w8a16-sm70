"""
Phase A.3 benchmark: low-M decode speedup via K-axis CTA splitting.

A.3 targets the regime where naive/A.1/A.2 grids under-utilize V100's 80 SMs.
At M=1 with N=2560, only 20 CTAs spawn — 75% of SMs idle. A.3 splits K
across K_SPLIT additional CTAs that atomic-add their partial sums.

For each M in {1, 4, 8, 16, 32, 128}:
  - Run naive, A.1, A.2, and A.3 (with K_SPLIT ∈ {2, 4, 8})
  - Confirm A.3 correctness vs PyTorch reference
  - Show timing + speedup vs the best non-A.3 option

Run:
    ./run_docker.sh a3
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
    print(f"Weight shape: [{N}, {K}], scale blocks: [{block_h}, {block_w}]")
    print(f"Naive CTAs per (M, K_split): N/128 = {N // 128} (V100 has 80 SMs)\n")

    fp8_bytes = W_fp8.view(torch.uint8).reshape(-1).to(dev).contiguous()
    scales    = scales_raw.to(torch.float16).reshape(-1).to(dev).contiguous()

    W_dq_f32 = ext.fp8_e4m3_to_fp16_block_scaled(
        fp8_bytes, scales, N, K, block_h, block_w
    ).reshape(N, K).float()

    # Pick K_SPLIT values that divide K cleanly (need K % (k_split * block_w) == 0).
    k_splits = [s for s in (2, 4, 8) if K % (s * block_w) == 0]
    print(f"Testable K_SPLIT values for K={K}: {k_splits}\n")

    cols = ["M", "naive", "A.1", "A.2"] + [f"A.3 k={k}" for k in k_splits] + \
           ["best non-A.3", "best A.3", "A.3/best"] + ["A.3 abs vs ref"]
    print(" | ".join(f"{c:>10}" for c in cols))
    print("-" * (len(cols) * 13))

    for M in [1, 2, 4, 8, 16, 32, 128]:
        torch.manual_seed(M)
        A = (torch.randn(M, K, device=dev) * 0.1).to(torch.float16)

        t_naive, _ = timed(ext.fp8_w8a16_gemm,    A, fp8_bytes, scales, N, K, block_h, block_w)
        t_a1,    _ = timed(ext.fp8_w8a16_gemm_a1, A, fp8_bytes, scales, N, K, block_h, block_w)
        t_a2,    _ = timed(ext.fp8_w8a16_gemm_a2, A, fp8_bytes, scales, N, K, block_h, block_w)

        # Run A.3 with each K_SPLIT.
        a3_results = {}
        for k_split in k_splits:
            t_a3, C_a3 = timed(
                ext.fp8_w8a16_gemm_a3, A, fp8_bytes, scales, N, K, block_h, block_w, k_split
            )
            a3_results[k_split] = (t_a3, C_a3)

        best_non_a3 = min(t_naive, t_a1, t_a2)
        best_a3_k   = min(a3_results, key=lambda k: a3_results[k][0])
        best_a3_t, best_a3_C = a3_results[best_a3_k]
        speedup = best_non_a3 / best_a3_t

        # Correctness of the best A.3 variant.
        ref_f32 = A.float() @ W_dq_f32.T
        diff = (best_a3_C.float() - ref_f32).abs()
        max_abs = diff.max().item()

        row = [
            f"{M}",
            f"{t_naive:.2f}",
            f"{t_a1:.2f}",
            f"{t_a2:.2f}",
        ]
        for k in k_splits:
            row.append(f"{a3_results[k][0]:.2f}")
        row.extend([
            f"{best_non_a3:.2f}",
            f"{best_a3_t:.2f}(k={best_a3_k})",
            f"{speedup:.2f}x",
            f"{max_abs:.2e}",
        ])
        print(" | ".join(f"{c:>10}" for c in row))

    print("\nNotes:")
    print(f"  - A.3 needs K % (k_split * {block_w}) == 0 -> testable k_splits = {k_splits}")
    print("  - 'best A.3 (k=X)' = the K_SPLIT value that won for that M")
    print("  - At low M, A.3 should win because it spawns K_SPLIT× more CTAs")
    print("  - At high M, A.2 should still win (M-tiling amortizes W traffic)")


if __name__ == "__main__":
    main()
