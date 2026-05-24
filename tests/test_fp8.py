"""
Hello-world FP8 (E4M3-FN) -> FP16 dequantization test on V100.

Goal: prove the CUDA kernel produces *bit-exact* results vs PyTorch's
built-in CPU conversion (torch.float8_e4m3fn -> torch.float16).

Run:
    cd experiments/v100_fp8_test
    python test_fp8.py
"""
import os
import sys
import torch

from fp8_w8a16_sm70.ext_loader import load_kernel

# JIT-compile the CUDA extension targeting sm_70 (V100).
print("Compiling fp8_dequant.cu for sm_70 ...")
ext = load_kernel(name="fp8_dequant_ext", verbose=True)
print("Compiled OK.\n")


def bits_of(t: torch.Tensor) -> torch.Tensor:
    """Return bit pattern of an FP16 tensor as uint16, for bit-exact compare."""
    return t.cpu().contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def describe(byte: int) -> str:
    sign = "-" if byte & 0x80 else "+"
    exp  = (byte >> 3) & 0x0F
    mant = byte & 0x07
    return f"0x{byte:02X} ({sign}sign exp={exp:04b} mant={mant:03b})"


FP8_MAX_E4M3 = 448.0   # largest finite value representable in E4M3-FN


def phase2_scale_broadcast(dev):
    """Quantize a known FP16 weight vector to FP8 + per-group scale,
    then reconstruct on GPU using our kernel and verify."""
    print("\n" + "=" * 70)
    print("Phase 2: FP8 + per-group scale -> FP16 weight reconstruction")
    print("=" * 70)

    torch.manual_seed(0)
    N = 1024
    group_size = 32
    n_groups = N // group_size

    # 1. Original FP16 weights with a realistic spread.
    W_orig = (torch.randn(N) * 2.0).to(torch.float16)

    # 2. Per-group scale = max(|W_group|) / FP8_MAX. Compute in FP32, store FP16.
    grouped = W_orig.float().view(n_groups, group_size)
    scales_fp32 = grouped.abs().amax(dim=1) / FP8_MAX_E4M3
    scales_fp32 = torch.where(scales_fp32 > 0, scales_fp32, torch.ones_like(scales_fp32))
    scales_fp16 = scales_fp32.to(torch.float16)

    # 3. Quantize: divide by scale, clamp, cast to FP8 (PyTorch handles rounding).
    W_scaled = grouped.float() / scales_fp16.float().view(-1, 1)
    W_scaled = W_scaled.clamp(-FP8_MAX_E4M3, FP8_MAX_E4M3)
    W_fp8 = W_scaled.to(torch.float8_e4m3fn).reshape(-1)
    fp8_bytes_cpu = W_fp8.view(torch.uint8)

    # 4. CPU reference reconstruction: FP8 -> FP16 -> * scale (broadcast).
    W_recon_cpu = W_fp8.to(torch.float16) * scales_fp16.repeat_interleave(group_size)

    # 5. GPU reconstruction using our kernel.
    W_recon_gpu = ext.fp8_e4m3_to_fp16_scaled(
        fp8_bytes_cpu.to(dev),
        scales_fp16.to(dev),
        group_size,
    ).cpu()

    # 6. Compare GPU vs CPU. Bit-exact would be nice but FP16 multiplication
    #    can differ in the last bit between backends; report both.
    cpu_bits = W_recon_cpu.view(torch.int16).to(torch.int32) & 0xFFFF
    gpu_bits = W_recon_gpu.view(torch.int16).to(torch.int32) & 0xFFFF
    bit_match = int((cpu_bits == gpu_bits).sum().item())
    diff = (W_recon_cpu.float() - W_recon_gpu.float()).abs()
    print(f"\nGPU vs CPU reconstruction (N={N}, group_size={group_size}):")
    print(f"  bit-exact:      {bit_match}/{N}")
    print(f"  max abs diff:   {diff.max().item():.6e}")
    print(f"  mean abs diff:  {diff.mean().item():.6e}")

    # 7. Inherent FP8 quantization error vs the original FP16.
    qerr = (W_recon_cpu.float() - W_orig.float()).abs()
    rel  = qerr / W_orig.float().abs().clamp_min(1e-6)
    print(f"\nFP8 quantization error vs original FP16:")
    print(f"  max abs error:  {qerr.max().item():.6e}")
    print(f"  mean abs error: {qerr.mean().item():.6e}")
    print(f"  max rel error:  {rel.max().item():.4%}")
    print(f"  weight range:   [{W_orig.min().item():.3f}, {W_orig.max().item():.3f}]")

    # 8. Verdict.
    # Bit-exact when same op order (FP16 mul) is used. PyTorch's CPU FP16 mul
    # may upcast internally — if so, we expect small last-bit differences but
    # max abs diff should be < scale_max * 2^-10 (one FP16 ULP at scale).
    max_scale = scales_fp16.float().max().item()
    fp16_ulp_at_scale = max_scale * (2 ** -10)
    if diff.max().item() <= fp16_ulp_at_scale:
        print(f"\nReconstruction GPU vs CPU within 1 FP16 ULP (<= {fp16_ulp_at_scale:.3e}). PASS")
    else:
        print(f"\nReconstruction GPU vs CPU exceeds 1 FP16 ULP threshold. INVESTIGATE")
        bad_idx = diff.argmax().item()
        print(f"  worst: idx={bad_idx} cpu={W_recon_cpu[bad_idx].item()} "
              f"gpu={W_recon_gpu[bad_idx].item()} orig={W_orig[bad_idx].item()}")


def main():
    assert torch.cuda.is_available(), "no CUDA device"
    dev = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability(0)})\n")
    print("=" * 70)
    print("Phase 1: raw FP8 -> FP16 dequant (no scale)")
    print("=" * 70)

    # ---- 1. Hand-picked edge cases -----------------------------------------
    edge_bytes = [
        0x00,  # +0
        0x80,  # -0
        0x38,  # +1.0   : sign=0 exp=0111=7 mant=000 -> 2^(7-7)*(1+0)   = 1.0
        0xB8,  # -1.0
        0x3C,  # +1.5   : exp=7 mant=100 -> 2^0*(1+4/8)                  = 1.5
        0x40,  # +2.0   : exp=8 mant=000 -> 2^1*1.0                       = 2.0
        0x30,  # +0.5   : exp=6 mant=000 -> 2^-1*1.0                      = 0.5
        0x7E,  # +max normal: exp=15 mant=110 -> 2^8*(1+6/8)             = 448
        0xFE,  # -max normal                                              = -448
        0x01,  # smallest +subnormal: exp=0 mant=001 -> 1 * 2^-9 ~ 0.001953125
        0x07,  # largest  +subnormal: exp=0 mant=111 -> 7 * 2^-9 ~ 0.013671875
        0x87,  # largest  -subnormal
        0x7F,  # +NaN (E4M3-FN only NaN encoding is exp=1111 mant=111)
        0xFF,  # -NaN
    ]
    edge = torch.tensor(edge_bytes, dtype=torch.uint8)

    # CPU reference via PyTorch's built-in FP8 dtype.
    cpu_fp8  = edge.view(torch.float8_e4m3fn)
    cpu_fp16 = cpu_fp8.to(torch.float16)

    # GPU result via our custom kernel.
    gpu_fp16 = ext.fp8_e4m3_to_fp16(edge.to(dev))

    print(f"{'byte':<24} {'CPU fp16':>14} {'GPU fp16':>14}   bits CPU/GPU   match")
    print("-" * 90)
    ok_count = 0
    for i, b in enumerate(edge_bytes):
        cv = cpu_fp16[i].item()
        gv = gpu_fp16[i].item()
        cb = int(cpu_fp16[i:i+1].cpu().view(torch.int16).item()) & 0xFFFF
        gb = int(gpu_fp16[i:i+1].cpu().view(torch.int16).item()) & 0xFFFF
        # NaN compares unequal by value; compare bit patterns instead.
        match = (cb == gb)
        ok_count += int(match)
        print(f"{describe(b):<24} {cv:>14} {gv:>14}   0x{cb:04X}/0x{gb:04X}   {'OK' if match else 'MISMATCH'}")

    print(f"\nEdge cases: {ok_count}/{len(edge_bytes)} match.\n")

    # ---- 2. Exhaustive sweep: all 256 possible FP8 bytes -------------------
    all_bytes = torch.arange(256, dtype=torch.uint8)
    cpu_all = all_bytes.view(torch.float8_e4m3fn).to(torch.float16)
    gpu_all = ext.fp8_e4m3_to_fp16(all_bytes.to(dev)).cpu()

    cpu_bits = bits_of(cpu_all)
    gpu_bits = bits_of(gpu_all)
    mismatches = (cpu_bits != gpu_bits).nonzero(as_tuple=True)[0].tolist()

    print(f"Exhaustive sweep (all 256 FP8 byte patterns): "
          f"{256 - len(mismatches)}/256 bit-exact.")

    if mismatches:
        print("\nMismatches:")
        for i in mismatches:
            print(f"  byte=0x{i:02X}  CPU=0x{int(cpu_bits[i]):04X}  "
                  f"GPU=0x{int(gpu_bits[i]):04X}  "
                  f"(CPU={cpu_all[i].item()}, GPU={gpu_all[i].item()})")
        sys.exit(1)
    else:
        print("\nAll 256 FP8 byte patterns convert bit-exactly. GPU kernel agrees with CPU reference.")

    # Phase 2: dequant with per-group scale broadcast.
    phase2_scale_broadcast(dev)


if __name__ == "__main__":
    main()
