# V100 FP8 (E4M3-FN) -> FP16 dequant — hello world

Goal: prove a CUDA kernel on V100 (sm_70) converts raw FP8 E4M3-FN bytes to
FP16 **bit-exactly**, validated against PyTorch's reference CPU conversion
(`torch.float8_e4m3fn.to(torch.float16)`).

This is the smallest possible step toward a Volta-native FP8 W8A16 path. It
intentionally does **not** use Marlin's optimized byte-perm trick (which
produces a *scaled* FP16 that requires the per-group scale to compensate).
Here we do a "proper" sign/exp/mantissa conversion so the answer can be
compared directly against the CPU reference.

## Files

- `fp8_dequant.cu` — CUDA kernel + torch extension binding
- `test_fp8.py`    — JIT-compile + edge cases + 256-byte exhaustive sweep

## Run

Requires: PyTorch (any recent version with `torch.float8_e4m3fn` dtype, i.e.
torch >= 2.1), CUDA toolkit, nvcc visible on PATH.

```bash
cd experiments/v100_fp8_test
python test_fp8.py
```

## What success looks like

```
Edge cases: 14/14 match.
Exhaustive sweep (all 256 FP8 byte patterns): 256/256 bit-exact.
All 256 FP8 byte patterns convert bit-exactly. GPU kernel agrees with CPU reference.
```

If any byte mismatches, the script prints the offending byte, the CPU bit
pattern, and the GPU bit pattern. That tells you exactly which case the
kernel's logic got wrong (sign, normal exponent, subnormal renormalization,
or NaN handling).

## What this proves (and doesn't)

✓ The bit-manipulation FP8->FP16 logic is correct on V100 CUDA cores
✓ Build + nvcc + torch JIT extension all work for sm_70
✓ E4M3-FN special cases (signed zero, subnormals, single NaN encoding) handled

✗ Does NOT prove anything about matmul performance
✗ Does NOT use tensor cores
✗ Does NOT integrate with a quantized linear layer yet

That's all next-step work. This is the foundation: if this test fails,
nothing built on top can be trusted.
