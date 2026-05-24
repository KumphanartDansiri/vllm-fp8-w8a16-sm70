"""
Phase B: validate that FP8W8A16Linear (the nn.Module wrapper) behaves
identically to calling the kernel directly, and integrates cleanly into a
PyTorch model.

Three checks:
  1. Module forward output == raw kernel output (bit-exact: same path)
  2. Module forward output ≈ PyTorch reference (A.float() @ W_dq.float().T).half()
  3. Module composes with other nn modules (Sequential, residual) without
     breaking — proving it's a real drop-in for nn.Linear.

Run:
    ./run_docker.sh phaseb
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
import torch.nn as nn
from fp8_w8a16_sm70.ext_loader import load_kernel as _load_kernel

from fp8_w8a16_sm70 import FP8W8A16Linear


HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/models/Qwen3.5-4B-FP8")


def load_kernel():
    print("Compiling kernel for sm_70 ...")
    ext = _load_kernel(name="fp8_dequant_ext")
    print("Compiled OK.\n")
    return ext


def pick_fp8_weight(model_dir):
    """Find an FP8 weight + scale companion. Returns (st_path, w_key, s_key)."""
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


def header(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def check_module_vs_kernel(ext, layer, st_path, w_key, s_key):
    """Sanity check #1: Module wraps the kernel — output should be bit-exact
    against a direct kernel call with the same inputs."""
    header("Check 1: Module.forward() == raw kernel call (bit-exact)")
    M = 8
    x = (torch.randn(M, layer.K, device="cuda") * 0.1).to(torch.float16)

    # Module path
    y_mod = layer(x)

    # Direct kernel path (using same flattened buffers the module holds)
    y_kern = ext.fp8_w8a16_gemm(
        x.contiguous(),
        layer.weight_fp8,
        layer.weight_scale,
        layer.N, layer.K, layer.block_h, layer.block_w,
    )

    bit_match = int((y_mod.view(torch.int16) == y_kern.view(torch.int16)).sum().item())
    total = y_mod.numel()
    print(f"  output shape : {list(y_mod.shape)}")
    print(f"  bit-exact    : {bit_match}/{total} ({100*bit_match/total:.2f}%)")
    if bit_match == total:
        print("  PASS: module produces identical bytes to direct kernel call.")
    else:
        print("  FAIL: module path differs from direct kernel call.")
        return False
    return True


def check_module_vs_reference(ext, layer, st_path, w_key, s_key, m_sizes):
    """Sanity check #2: Module output ≈ torch reference matmul on Phase-4-dequant.
    Same correctness bar as Phase 5 — within FP16 matmul noise floor."""
    header("Check 2: Module output vs PyTorch reference (within FP16 noise)")

    # Get the Phase 4 dequantized weight once for the reference path.
    W_dq = ext.fp8_e4m3_to_fp16_block_scaled(
        layer.weight_fp8, layer.weight_scale,
        layer.N, layer.K, layer.block_h, layer.block_w,
    ).reshape(layer.N, layer.K)
    W_dq_f32 = W_dq.float()

    all_pass = True
    for M in m_sizes:
        torch.manual_seed(M)
        x = (torch.randn(M, layer.K, device="cuda") * 0.1).to(torch.float16)
        ref = (x.float() @ W_dq_f32.T).to(torch.float16)
        y   = layer(x)

        diff = (y.float() - ref.float()).abs()
        max_abs  = diff.max().item()
        mean_abs = diff.mean().item()
        # Only consider rel error at outputs that aren't near zero.
        thresh = 0.01 * ref.float().abs().max().item()
        mask = ref.float().abs() > thresh
        max_rel = (diff[mask] / ref.float()[mask].abs()).max().item() if mask.any() else 0.0

        ok = max_abs <= 1e-2
        all_pass &= ok
        verdict = "PASS" if ok else "FAIL"
        print(f"  M={M:<4d}  max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  "
              f"max_rel(non-zero)={max_rel:.3%}  -> {verdict}")
    return all_pass


def check_module_in_sequential(layer):
    """Sanity check #3: Drop the module into nn.Sequential with a residual
    connection. Proves it composes like any other nn.Module."""
    header("Check 3: Module composes inside nn.Sequential + residual")

    class TinyBlock(nn.Module):
        def __init__(self, fp8_linear, K):
            super().__init__()
            self.norm = nn.LayerNorm(K, dtype=torch.float16).cuda()
            self.fp8 = fp8_linear
            # Project back to K for residual; just use a regular Linear here.
            self.out = nn.Linear(fp8_linear.N, K, dtype=torch.float16).cuda()

        def forward(self, x):
            return x + self.out(self.fp8(self.norm(x)))

    block = TinyBlock(layer, layer.K)
    x = (torch.randn(2, 16, layer.K, device="cuda") * 0.1).to(torch.float16)
    try:
        y = block(x)
        ok = (y.shape == x.shape) and torch.isfinite(y).all()
        print(f"  input  shape: {list(x.shape)} (3D: batch, seq, dim)")
        print(f"  output shape: {list(y.shape)}")
        print(f"  all finite  : {torch.isfinite(y).all().item()}")
        print(f"  -> {'PASS' if ok else 'FAIL'}: module survives 3D input, "
              f"LayerNorm composition, and residual add.")
        return ok
    except Exception as e:
        print(f"  FAIL: composition raised exception: {e}")
        return False


def main():
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability(0)})")

    ext = load_kernel()
    st_path, w_key, s_key = pick_fp8_weight(MODEL_DIR)
    print(f"Using file:   {st_path.name}")
    print(f"Using weight: {w_key}")
    print(f"Using scale:  {s_key}")

    # Construct the module from the real model file.
    layer = FP8W8A16Linear.from_safetensors(ext, st_path, w_key, s_key)
    print(f"\nConstructed: {layer}")

    results = []
    results.append(("kernel parity",   check_module_vs_kernel(ext, layer, st_path, w_key, s_key)))
    results.append(("reference parity",check_module_vs_reference(ext, layer, st_path, w_key, s_key,
                                                                  m_sizes=[1, 4, 32, 128])))
    results.append(("composition",     check_module_in_sequential(layer)))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, ok in results:
        print(f"  {name:<20s}: {'PASS' if ok else 'FAIL'}")
    overall = all(ok for _, ok in results)
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
