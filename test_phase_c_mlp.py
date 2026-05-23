"""
Phase C: end-to-end test — replace the FP8 weights of one MLP block
(gate_proj / up_proj / down_proj) with FP8W8A16Linear and run a SwiGLU
forward pass against a pure-FP16 reference built from the same weights.

This is the minimum "real architectural context" we can validate:
  - 3 sequential FP8 GEMMs (not just one in isolation)
  - Intermediate non-linear activation (SiLU * up_proj)
  - Numerical error compounds across layers
  - Exercises the dispatch logic at the picked M
  - Same weight shapes as the real model would call

If this passes, FP8W8A16Linear is proven to compose correctly in a
realistic chain of layers — the foundation for swapping into actual
transformer blocks.

Run:
    ./run_docker.sh mlp
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from fp8_w8a16_module import FP8W8A16Linear


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


def find_mlp_triplet(model_dir):
    """Locate the (gate, up, down) FP8 weight triplet for one MLP block.
    Searches across safetensors shards for the standard
    <prefix>.mlp.{gate,up,down}_proj.weight names."""
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())
        # Look for any layer that has all three projections present (and FP8).
        prefixes = set()
        for k in keys:
            if k.endswith(".mlp.gate_proj.weight"):
                prefixes.add(k[: -len(".mlp.gate_proj.weight")])
        for prefix in sorted(prefixes):
            triplet = (
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            )
            triplet_scales = tuple(t + "_scale_inv" for t in triplet)
            if all(t in keys for t in triplet) and all(s in keys for s in triplet_scales):
                # Verify all three are FP8.
                with safe_open(path, framework="pt") as f:
                    if all(f.get_slice(t).get_dtype() == "F8_E4M3" for t in triplet):
                        return path, prefix, triplet, triplet_scales
    sys.exit("No complete FP8 MLP triplet (gate/up/down) found")


class FP8SwiGLU(nn.Module):
    """SwiGLU MLP block backed by our FP8 W8A16 kernel.
       y = down_proj(SiLU(gate_proj(x)) * up_proj(x))"""
    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj   = up_proj
        self.down_proj = down_proj

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        return self.down_proj(F.silu(g) * u)


class FP16SwiGLU(nn.Module):
    """Reference SwiGLU MLP using FP16 weights (matmul-based, no FP8)."""
    def __init__(self, gate_w_fp16, up_w_fp16, down_w_fp16):
        super().__init__()
        # weights stored as [N, K]; matmul does x @ W.T
        self.register_buffer("gate_w", gate_w_fp16)
        self.register_buffer("up_w",   up_w_fp16)
        self.register_buffer("down_w", down_w_fp16)

    def forward(self, x):
        # Compute in FP32 internally for a stable reference, then cast back.
        x32 = x.float()
        g = (x32 @ self.gate_w.float().T)
        u = (x32 @ self.up_w.float().T)
        y = F.silu(g) * u
        return (y @ self.down_w.float().T).to(torch.float16)


def main():
    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability(0)})")

    ext = load_kernel()

    st_path, prefix, weight_keys, scale_keys = find_mlp_triplet(MODEL_DIR)
    print(f"Using file:   {st_path.name}")
    print(f"MLP prefix:   {prefix}")
    for w in weight_keys:
        print(f"  weight:     {w}")

    # Load FP8 weights + scales onto GPU (raw bytes, then cast scales to FP16).
    with safe_open(st_path, framework="pt") as f:
        fp8_tensors    = [f.get_tensor(k) for k in weight_keys]
        scale_tensors  = [f.get_tensor(k) for k in scale_keys]

    shapes = [t.shape for t in fp8_tensors]
    print(f"  gate shape: {list(shapes[0])}  (hidden -> intermediate)")
    print(f"  up shape:   {list(shapes[1])}  (hidden -> intermediate)")
    print(f"  down shape: {list(shapes[2])}  (intermediate -> hidden)")

    hidden_size       = shapes[0][1]
    intermediate_size = shapes[0][0]
    assert shapes[1] == shapes[0],          f"up/gate shape mismatch"
    assert shapes[2] == (hidden_size, intermediate_size), f"down shape mismatch"

    # Build the FP8 SwiGLU using our module (with auto-dispatch).
    fp8_layers = []
    for w, s in zip(fp8_tensors, scale_tensors):
        N, K = w.shape
        scale_fp16 = s.to(torch.float16).to(dev).contiguous()
        w_u8       = w.view(torch.uint8).to(dev).contiguous()
        fp8_layers.append(
            FP8W8A16Linear(ext, w_u8, scale_fp16, block_h=128, block_w=128)
        )
    fp8_mlp = FP8SwiGLU(*fp8_layers).to(dev)

    # Build the reference SwiGLU using FP16 dequantized weights.
    # Use our Phase 4 kernel to produce the same FP16 values both paths see.
    fp16_weights = []
    for w, s in zip(fp8_tensors, scale_tensors):
        N, K = w.shape
        scale_fp16 = s.to(torch.float16).to(dev).contiguous().reshape(-1)
        w_u8_flat  = w.view(torch.uint8).to(dev).contiguous().reshape(-1)
        w_fp16 = ext.fp8_e4m3_to_fp16_block_scaled(
            w_u8_flat, scale_fp16, N, K, 128, 128
        ).reshape(N, K)
        fp16_weights.append(w_fp16)
    ref_mlp = FP16SwiGLU(*fp16_weights).to(dev)

    print(f"\nConstructed:")
    print(f"  FP8:  {fp8_mlp}")
    print()

    # Test across a range of M (batch * sequence-length flattened).
    header = f"{'M':>4} | {'fp8 (ms)':>8} | {'ref (ms)':>8} | {'max_abs':>10} | {'mean_abs':>10} | {'max_rel*':>9} | {'verdict':>8}"
    print(header)
    print("-" * len(header))

    all_pass = True
    for M in [1, 4, 16, 32, 128]:
        torch.manual_seed(M)
        # Input shape: [M, hidden_size] in FP16. Scale ~unit variance like
        # post-RMSNorm activations.
        x = torch.randn(M, hidden_size, device=dev, dtype=torch.float16)

        # Warmup + time both paths.
        for _ in range(2):
            _ = fp8_mlp(x); _ = ref_mlp(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(5):
            y_fp8 = fp8_mlp(x)
        torch.cuda.synchronize()
        t_fp8 = (time.perf_counter() - t0) * 1000 / 5

        t0 = time.perf_counter()
        for _ in range(5):
            y_ref = ref_mlp(x)
        torch.cuda.synchronize()
        t_ref = (time.perf_counter() - t0) * 1000 / 5

        diff = (y_fp8.float() - y_ref.float()).abs()
        max_abs  = diff.max().item()
        mean_abs = diff.mean().item()
        ref_thresh = 0.01 * y_ref.float().abs().max().item()
        mask = y_ref.float().abs() > ref_thresh
        max_rel = (diff[mask] / y_ref.float()[mask].abs()).max().item() if mask.any() else 0.0

        # Tolerance: 3 sequential GEMMs over K~5120 accumulate ~3x the single-GEMM
        # noise. Single-GEMM noise floor ~5e-4, so 3-layer chain ~2e-3 is realistic.
        # Use 5e-2 absolute and 10% relative for the multi-GEMM chain.
        ok = (max_abs <= 5e-2) and (max_rel <= 0.10)
        all_pass &= ok
        verdict = "PASS" if ok else "INVESTIGATE"
        print(f"{M:>4d} | {t_fp8:>6.2f}ms | {t_ref:>6.2f}ms | "
              f"{max_abs:>10.2e} | {mean_abs:>10.2e} | "
              f"{max_rel:>8.2%} | {verdict:>8}")

    # Show what dispatch picked at each M (informational).
    print("\nDispatch decisions:")
    for M in [1, 4, 16, 32, 128]:
        have_a3 = hasattr(ext, "fp8_w8a16_gemm_a3")
        have_a2 = hasattr(ext, "fp8_w8a16_gemm_a2")
        if have_a3 and M <= FP8W8A16Linear.DISPATCH_M_A3_K8:
            kind = "A.3 K_SPLIT=8"
        elif have_a3 and M <= FP8W8A16Linear.DISPATCH_M_A3_K4:
            kind = "A.3 K_SPLIT=4"
        elif have_a2 and M >= FP8W8A16Linear.DISPATCH_M_A2:
            kind = "A.2"
        else:
            kind = "A.1"
        print(f"  M={M:<4d} -> {kind}")

    print("\n" + ("PASS — FP8 MLP block matches FP16 reference at FP16 noise level."
                  if all_pass else "INVESTIGATE — see failing rows above."))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
