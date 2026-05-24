"""
Inspect a real compressed-tensors FP8 model file and validate that our
Phase 1+2 GPU dequant kernel reproduces the exact same FP16 weight values
that PyTorch's CPU reference path produces.

Goal: derisk Phase 3 by confirming our quantization convention (scale layout,
group size, dtype) matches what real model files actually contain.

Run:
    ./run_docker.sh inspect                       # default: /mnt/models/Qwen3.6-27B-FP8
    ./run_docker.sh inspect /mnt/models/<other>   # any other compressed-tensors FP8 model
"""
import json
import os
import subprocess
import sys
from pathlib import Path

# Auto-install safetensors if missing (small package, faster than rebuilding image).
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


def header(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def list_files():
    header(f"Model directory: {MODEL_DIR}")
    if not MODEL_DIR.exists():
        sys.exit(f"ERROR: {MODEL_DIR} does not exist (is /mnt/models mounted?)")
    for f in sorted(MODEL_DIR.iterdir()):
        if f.is_file():
            sz = f.stat().st_size / (1024 ** 3)
            print(f"  {f.name:55s} {sz:7.2f} GB")
        else:
            print(f"  {f.name}/")


def show_quantization_config():
    header("quantization_config from config.json (compact)")
    cfg = MODEL_DIR / "config.json"
    if not cfg.exists():
        print("  (config.json not found)")
        return None
    data = json.loads(cfg.read_text())
    qc = data.get("quantization_config")
    if qc is None:
        print("  (no quantization_config in config.json)")
        return None
    # Print scalar fields directly; summarize large lists so output stays readable.
    for k, v in qc.items():
        if isinstance(v, list):
            print(f"  {k}: list with {len(v)} entries")
            for s in v[:3]:
                print(f"      e.g. {s}")
            if len(v) > 3:
                print(f"      ... ({len(v) - 3} more)")
        elif isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} keys")
        else:
            print(f"  {k}: {v}")
    return qc


def scan_tensors(max_show=10):
    header("Scanning first safetensors file for FP8 weights + scales")
    st_files = sorted(MODEL_DIR.glob("*.safetensors"))
    if not st_files:
        sys.exit("ERROR: no .safetensors files found")
    print(f"  {len(st_files)} safetensors file(s); reading {st_files[0].name}")

    fp8_weights = []
    other_dtypes = {}
    with safe_open(st_files[0], framework="pt") as f:
        keys = list(f.keys())
        # Use get_slice to peek dtype/shape without loading the tensor.
        for k in keys:
            sl = f.get_slice(k)
            dtype_str = sl.get_dtype()  # returns string like "F8_E4M3", "F16", "F32"
            shape = sl.get_shape()
            if "F8" in dtype_str:
                fp8_weights.append((k, dtype_str, shape))
            else:
                other_dtypes.setdefault(dtype_str, 0)
                other_dtypes[dtype_str] += 1

    print(f"\n  FP8 weight tensors found: {len(fp8_weights)}")
    print(f"  Non-FP8 dtype histogram: {other_dtypes}")
    print(f"\n  First {min(max_show, len(fp8_weights))} FP8 weights:")
    for k, dt, sh in fp8_weights[:max_show]:
        print(f"    {dt:10s} {str(sh):25s} {k}")

    # Surface the scale tensors paired with FP8 weights so we can see the
    # convention (per-tensor / per-channel / per-block / etc.) without loading
    # anything heavy. If we can't find a scale by name, fall back to listing
    # the sibling tensors under the same prefix so we see what IS stored.
    print(f"\n  Scale + sibling tensors for first {min(max_show, len(fp8_weights))} FP8 weights:")
    with safe_open(st_files[0], framework="pt") as f:
        all_keys = set(f.keys())
        for wk, _, wsh in fp8_weights[:max_show]:
            print(f"\n    weight: {wk}  shape={wsh}")
            sk = find_scale_key(wk, all_keys)
            if sk is not None:
                sl = f.get_slice(sk)
                print(f"      scale:   {sl.get_dtype():8s} {str(sl.get_shape()):20s} {sk}")
            else:
                print(f"      (no scale matched known naming conventions)")
            sibs = list_siblings(wk, all_keys)
            if sibs:
                print(f"      siblings under same prefix:")
                for s in sibs:
                    sl = f.get_slice(s)
                    print(f"        {sl.get_dtype():8s} {str(sl.get_shape()):20s} {s}")

    return st_files[0], fp8_weights


def find_scale_key(weight_key, all_keys):
    """Try the common scale-naming conventions for FP8 quant formats:
       compressed-tensors  -> <name>.weight_scale
       vLLM-native / DeepSeek block-FP8 -> <name>.weight_scale_inv
       generic             -> <name>.scale, <name>.scale_inv
    """
    candidates = [
        weight_key + "_scale",
        weight_key + "_scale_inv",
        weight_key.replace(".weight", ".weight_scale"),
        weight_key.replace(".weight", ".weight_scale_inv"),
        weight_key.replace(".weight", ".scale"),
        weight_key.replace(".weight", ".scale_inv"),
    ]
    for c in candidates:
        if c in all_keys:
            return c
    return None


def list_siblings(weight_key, all_keys, max_show=12):
    """Return tensor keys sharing the same prefix (up to and including .weight)."""
    # prefix = everything before the final ".weight"
    if not weight_key.endswith(".weight"):
        return []
    prefix = weight_key[: -len(".weight")] + "."
    sibs = sorted(k for k in all_keys if k.startswith(prefix) and k != weight_key)
    return sibs[:max_show]


def describe_scale_layout(weight_shape, scale_shape):
    if len(scale_shape) == 0 or scale_shape == [1] or scale_shape == [1, 1]:
        return "per-tensor"
    if len(weight_shape) == 2:
        N, K = weight_shape
        if list(scale_shape) == [N] or list(scale_shape) == [N, 1]:
            return f"per-output-channel (1 scale per row of {K})"
        if len(scale_shape) == 2 and scale_shape[0] == N and K % scale_shape[1] == 0:
            gs = K // scale_shape[1]
            return f"per-group along K (group_size={gs})"
        if len(scale_shape) == 2 and scale_shape[1] == K and N % scale_shape[0] == 0:
            gs = N // scale_shape[0]
            return f"per-group along N (group_size={gs}) -- unusual"
    return f"unknown layout for weight {weight_shape}"


def load_kernel():
    print("\nCompiling our kernel...")
    return _load_kernel(name="fp8_dequant_ext")


def validate_one_weight(st_file, weight_key, scale_key):
    header(f"Validating dequant on: {weight_key}")
    with safe_open(st_file, framework="pt") as f:
        W_fp8 = f.get_tensor(weight_key)
        scales_raw = f.get_tensor(scale_key)

    print(f"  weight: dtype={W_fp8.dtype} shape={list(W_fp8.shape)}")
    print(f"  scale : dtype={scales_raw.dtype} shape={list(scales_raw.shape)}")
    layout = describe_scale_layout(list(W_fp8.shape), list(scales_raw.shape))
    print(f"  layout: {layout}")

    # Cast scale to FP16 (W8A16 path uses FP16 scales for cheap fused multiply).
    # Note this MAY lose precision if model stores FP32 scales -- document it.
    scales_fp16 = scales_raw.to(torch.float16)
    if scales_raw.dtype != torch.float16:
        max_diff = (scales_raw.float() - scales_fp16.float()).abs().max().item()
        print(f"  note  : scale cast {scales_raw.dtype} -> float16; "
              f"max precision loss = {max_diff:.6e}")

    # CPU reference reconstruction: dequant FP8 -> FP16, then broadcast-multiply
    # by the scale tensor. Works for any scale rank (per-tensor, per-channel,
    # block-2D) by relying on PyTorch's broadcasting after upsampling.
    W_fp16 = W_fp8.to(torch.float16)
    if W_fp8.ndim == 2 and scales_fp16.ndim == 2:
        N_out, K_in = W_fp8.shape
        block_h = N_out // scales_fp16.shape[0]
        block_w = K_in  // scales_fp16.shape[1]
        print(f"  -> 2D block scales: block=[{block_h}, {block_w}]")
        scales_full = scales_fp16.repeat_interleave(block_h, 0).repeat_interleave(block_w, 1)
    elif W_fp8.ndim == 2 and scales_fp16.ndim == 1 and scales_fp16.numel() == W_fp8.shape[0]:
        print(f"  -> per-output-channel scales")
        scales_full = scales_fp16.view(-1, 1).expand_as(W_fp16)
    elif scales_fp16.numel() == 1:
        print(f"  -> per-tensor scale")
        scales_full = scales_fp16.expand_as(W_fp16)
    else:
        print(f"  -> unrecognized layout, attempting raw broadcast")
        scales_full = scales_fp16
    W_recon_cpu = W_fp16 * scales_full

    print(f"\n  CPU-reconstructed FP16 weight stats:")
    print(f"    min:  {W_recon_cpu.min().item():.6e}")
    print(f"    max:  {W_recon_cpu.max().item():.6e}")
    print(f"    mean: {W_recon_cpu.mean().item():.6e}")
    print(f"    std:  {W_recon_cpu.std().item():.6e}")
    print(f"    NaN count: {torch.isnan(W_recon_cpu).sum().item()}")

    # Phase 4: 2D-block scale path uses our new dedicated kernel.
    if scales_fp16.ndim == 2 and W_fp8.ndim == 2 and scales_fp16.shape[1] != 1:
        N_out, K_in = W_fp8.shape
        block_h = N_out // scales_fp16.shape[0]
        block_w = K_in  // scales_fp16.shape[1]
        print(f"\n  Phase 4: invoking 2D-block dequant kernel "
              f"(block=[{block_h}, {block_w}], scales={list(scales_fp16.shape)})")

        ext = load_kernel()
        fp8_bytes = W_fp8.view(torch.uint8).reshape(-1).cuda().contiguous()
        scales_gpu = scales_fp16.reshape(-1).cuda().contiguous()
        out_gpu_flat = ext.fp8_e4m3_to_fp16_block_scaled(
            fp8_bytes, scales_gpu, N_out, K_in, block_h, block_w
        )
        W_recon_gpu = out_gpu_flat.reshape(N_out, K_in).cpu()

        cpu_bits = W_recon_cpu.view(torch.int16).to(torch.int32) & 0xFFFF
        gpu_bits = W_recon_gpu.view(torch.int16).to(torch.int32) & 0xFFFF
        bit_match = int((cpu_bits == gpu_bits).sum().item())
        total = W_recon_cpu.numel()
        diff = (W_recon_cpu.float() - W_recon_gpu.float()).abs()
        print(f"\n  bit-exact GPU vs CPU: {bit_match}/{total} ({100*bit_match/total:.4f}%)")
        print(f"  max abs diff:  {diff.max().item():.6e}")
        print(f"  mean abs diff: {diff.mean().item():.6e}")
        if bit_match == total:
            print("\n  PASS: 2D-block kernel reproduces PyTorch reference bit-for-bit "
                  "on real model data.")
        elif diff.max().item() < 1e-3:
            print("\n  CLOSE: not bit-exact but within FP16 rounding noise.")
        else:
            print("\n  INVESTIGATE: real divergence between GPU and CPU.")
            # Surface a few worst offenders.
            flat_diff = diff.reshape(-1)
            worst = torch.topk(flat_diff, k=5).indices.tolist()
            for w in worst:
                i_, j_ = w // K_in, w % K_in
                print(f"    idx=({i_}, {j_}) fp8=0x{fp8_bytes[w].item():02X} "
                      f"cpu={W_recon_cpu[i_, j_].item()} "
                      f"gpu={W_recon_gpu[i_, j_].item()}")
        return

    # 1D path: use the existing kernel.
    N_out = W_fp8.shape[0]
    K_in  = W_fp8.shape[1] if W_fp8.ndim == 2 else W_fp8.numel() // N_out
    if scales_fp16.numel() == 1:
        group_size = W_fp8.numel()
        flat_scales = scales_fp16.reshape(1)
    else:
        group_size = K_in
        flat_scales = scales_fp16.reshape(N_out)
    print(f"  -> kernel call: numel={W_fp8.numel()} group_size={group_size}")

    ext = load_kernel()
    fp8_bytes = W_fp8.view(torch.uint8).reshape(-1).cuda().contiguous()
    scales_gpu = flat_scales.cuda().contiguous()
    W_recon_gpu = ext.fp8_e4m3_to_fp16_scaled(fp8_bytes, scales_gpu, group_size).cpu()

    cpu_flat = W_recon_cpu.reshape(-1)
    cpu_bits = cpu_flat.view(torch.int16).to(torch.int32) & 0xFFFF
    gpu_bits = W_recon_gpu.view(torch.int16).to(torch.int32) & 0xFFFF
    bit_match = int((cpu_bits == gpu_bits).sum().item())
    total = cpu_flat.numel()
    diff = (cpu_flat.float() - W_recon_gpu.float()).abs()
    print(f"\n  bit-exact GPU vs CPU: {bit_match}/{total} ({100*bit_match/total:.4f}%)")
    print(f"  max abs diff:  {diff.max().item():.6e}")
    print(f"  mean abs diff: {diff.mean().item():.6e}")
    if bit_match == total:
        print("\n  PASS: kernel reproduces PyTorch reference bit-for-bit on real model data.")


def main():
    list_files()
    show_quantization_config()
    st_file, fp8_weights = scan_tensors()

    if not fp8_weights:
        print("\nNo FP8 weights found — nothing to validate.")
        return

    # Pick the first FP8 weight that has a discoverable scale.
    with safe_open(st_file, framework="pt") as f:
        all_keys = set(f.keys())
    for wk, _, _ in fp8_weights:
        sk = find_scale_key(wk, all_keys)
        if sk:
            validate_one_weight(st_file, wk, sk)
            return

    print("\nNo paired scale tensor found for any FP8 weight — "
          "this model may use a layout we haven't seen yet.")


if __name__ == "__main__":
    main()
