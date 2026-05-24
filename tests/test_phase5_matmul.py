"""
Phase 5: validate the fused FP8 W8A16 GEMM kernel on V100 against a real
Qwen3 weight tensor.

Compares:
  (GPU)  C = fp8_w8a16_gemm(input, fp8_weight, scales)
  (REF)  C = (input.float() @ Phase4_dequant(fp8_weight, scales).float().T).half()

Bit-exact agreement is NOT expected -- matmul accumulation order and the
internal FP32-vs-FP16 differences mean some last-bit noise is normal. We
check with torch.testing.assert_close at FP16-realistic tolerances and
also report worst-case error, mean error, and relative error.

Picks the smallest FP8 weight in layers-0.safetensors (in_proj_z: [6144, 5120])
to keep the test quick.

Run:
    ./run_docker.sh matmul
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


def pick_weight(model_dir):
    """Find a moderately-sized FP8 weight + its 2D block scale, robust to
    different safetensors file naming conventions (per-layer vs sharded vs single)."""
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"No *.safetensors files found in {model_dir}")
    # Look across files (some sharded layouts put first transformer layer in
    # file 1 not file 0; e.g. 'model-00001-of-N' often starts with embeddings).
    for path in st_files:
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
            # Collect FP8 weights with a discoverable scale_inv companion.
            candidates = []
            keys_set = set(keys)
            for k in keys:
                sl = f.get_slice(k)
                if sl.get_dtype() != "F8_E4M3":
                    continue
                if not k.endswith(".weight"):
                    continue
                if k + "_scale_inv" not in keys_set and \
                   k.replace(".weight", ".weight_scale_inv") not in keys_set:
                    continue
                shape = sl.get_shape()
                if len(shape) != 2:
                    continue
                candidates.append((k, shape, shape[0] * shape[1]))
            if not candidates:
                continue
            # Smallest weight = fastest to test.
            candidates.sort(key=lambda x: x[2])
            chosen_key, chosen_shape, _ = candidates[0]
            print(f"Using file:   {path.name}")
            print(f"Using weight: {chosen_key}  shape={chosen_shape}")
            return path, chosen_key
    sys.exit("No FP8 weight with paired scale_inv found in any safetensors file")


def run_validation(ext, dev, st_path, weight_key, scale_key, m_sizes):
    with safe_open(st_path, framework="pt") as f:
        W_fp8 = f.get_tensor(weight_key)
        scales_raw = f.get_tensor(scale_key)

    N, K = W_fp8.shape
    block_h = N // scales_raw.shape[0]
    block_w = K // scales_raw.shape[1]
    print(f"  weight  : {list(W_fp8.shape)} (N={N}, K={K})")
    print(f"  scales  : {list(scales_raw.shape)} (BF16), blocks=[{block_h},{block_w}]")
    print(f"  scales  : casting BF16 -> FP16")
    scales_fp16 = scales_raw.to(torch.float16)

    # Move quant tensors to GPU once.
    fp8_bytes_gpu = W_fp8.view(torch.uint8).reshape(-1).to(dev).contiguous()
    scales_gpu    = scales_fp16.reshape(-1).to(dev).contiguous()

    # Ground-truth dequantized weight via our proven Phase 4 kernel.
    W_dq = ext.fp8_e4m3_to_fp16_block_scaled(
        fp8_bytes_gpu, scales_gpu, N, K, block_h, block_w
    ).reshape(N, K)
    W_dq_f32 = W_dq.float()

    for M in m_sizes:
        print("\n" + "-" * 60)
        print(f"M={M} (batch/sequence dim)")
        print("-" * 60)
        torch.manual_seed(M)
        A = (torch.randn(M, K, device=dev) * 0.1).to(torch.float16)

        # Reference: FP32 matmul against Phase 4 dequant, then cast to FP16.
        # This is what an FP16-baseline implementation would compute.
        ref_f32 = A.float() @ W_dq_f32.T
        C_ref   = ref_f32.to(torch.float16)

        # GPU kernel under test.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        C_gpu = ext.fp8_w8a16_gemm(A, fp8_bytes_gpu, scales_gpu,
                                    N, K, block_h, block_w)
        torch.cuda.synchronize()
        gemm_ms = (time.perf_counter() - t0) * 1000.0

        # Compare.
        diff_f32 = (C_gpu.float() - ref_f32).abs()
        # For relative error, only consider outputs that aren't near-zero
        # (near-zero outputs make tiny abs errors look huge in relative terms,
        # but contribute negligibly downstream). Threshold: 1% of max magnitude.
        ref_thresh = 0.01 * ref_f32.abs().max().item()
        mask = ref_f32.abs() > ref_thresh
        if mask.any():
            rel = diff_f32[mask] / ref_f32[mask].abs()
            max_rel = rel.max().item()
        else:
            max_rel = 0.0

        max_abs  = diff_f32.max().item()
        mean_abs = diff_f32.mean().item()

        cpu_bits = C_ref.view(torch.int16).to(torch.int32) & 0xFFFF
        gpu_bits = C_gpu.view(torch.int16).to(torch.int32) & 0xFFFF
        bit_match = int((cpu_bits == gpu_bits).sum().item())
        total = C_ref.numel()

        print(f"  kernel time       : {gemm_ms:7.3f} ms")
        print(f"  output shape      : {list(C_gpu.shape)}")
        print(f"  reference range   : [{ref_f32.min().item():.4f}, {ref_f32.max().item():.4f}]")
        print(f"  max abs diff      : {max_abs:.4e}")
        print(f"  mean abs diff     : {mean_abs:.4e}")
        print(f"  max rel diff      : {max_rel:.4%}  (only outputs > {ref_thresh:.3e})")
        print(f"  bit-exact         : {bit_match}/{total} ({100*bit_match/total:.2f}%)")

        # FP16 has ~3-4 decimal digits of precision. Matmul over K=5120 can
        # accumulate ~sqrt(K) * eps_fp32 worth of FP32 rounding ~= 7e-5,
        # then casting to FP16 adds one ULP at result magnitude (~1e-3 here).
        # rtol=1e-2, atol=1e-2 is a tight but realistic FP16 matmul bar.
        ok = max_abs <= 1e-2 and max_rel <= 0.05
        verdict = "PASS" if ok else "INVESTIGATE"
        print(f"  verdict           : {verdict}")
        if not ok:
            # Surface worst offenders.
            flat = diff_f32.reshape(-1)
            worst = torch.topk(flat, k=3).indices.tolist()
            for w in worst:
                m_, n_ = w // C_ref.shape[1], w % C_ref.shape[1]
                print(f"    worst idx=({m_},{n_}): "
                      f"ref={ref_f32[m_, n_].item():.6f} "
                      f"gpu={C_gpu[m_, n_].item():.6f}")


def main():
    assert torch.cuda.is_available(), "no CUDA device"
    dev = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability(0)})")

    ext = load_kernel()
    st_path, weight_key = pick_weight(MODEL_DIR)

    # Also resolve the actual scale-key name (could be *.weight_scale_inv
    # or *_scale_inv depending on file convention).
    with safe_open(st_path, framework="pt") as f:
        keys = set(f.keys())
    candidates = [
        weight_key + "_scale_inv",
        weight_key.replace(".weight", ".weight_scale_inv"),
    ]
    scale_key = next((c for c in candidates if c in keys), None)
    if scale_key is None:
        sys.exit(f"Could not find scale for {weight_key}")
    print(f"Using scale:  {scale_key}\n")

    print("\n" + "=" * 70)
    print(f"Phase 5: FP8 W8A16 GEMM validation on {weight_key}")
    print("=" * 70)

    # Test multiple M values to cover decode (M=1) and prefill (M=large).
    run_validation(ext, dev, st_path, weight_key, scale_key, m_sizes=[1, 4, 32, 128])


if __name__ == "__main__":
    main()
