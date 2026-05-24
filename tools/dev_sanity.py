"""
Sanity check for the cu128 dev image (vllm-v100-dev:cu128).

Verifies:
  1. Package versions match AIAGENT_ENV.md (vllm 0.18.0, torch 2.10.0+cu128, etc.)
  2. GPU is visible and torch can initialize CUDA on the host driver (535)
  3. Our fp8_dequant.cu kernel JIT-compiles against torch 2.10 + cu128 headers
     and produces the same bit-exact FP8->FP16 output we validated earlier

Run:
    ./run_docker.sh dev-test dev_sanity.py
"""
import sys
import torch
from fp8_w8a16_sm70.ext_loader import load_kernel as _load_kernel


def section(s):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main():
    # ─── 1. Version check ─────────────────────────────────────────────
    section("1. Package versions (compare to AIAGENT_ENV.md)")
    import vllm, transformers, triton, safetensors
    rows = [
        ("python",       sys.version.split()[0],     "3.10.x"),
        ("torch",        torch.__version__,          "2.10.0+cu128"),
        ("cuda runtime", torch.version.cuda,         "12.8"),
        ("vllm",         vllm.__version__,           "0.18.0"),
        ("transformers", transformers.__version__,   "4.57.6"),
        ("triton",       triton.__version__,         "3.6.0"),
        ("safetensors",  safetensors.__version__,    "0.7.0"),
    ]
    print(f"  {'package':<14} {'installed':<20} {'expected':<14}")
    print(f"  {'-'*14} {'-'*20} {'-'*14}")
    all_match = True
    for name, got, want in rows:
        ok = got == want or (want.endswith("x") and got.startswith(want[:-1]))
        all_match &= ok
        flag = "OK" if ok else "MISMATCH"
        print(f"  {name:<14} {got:<20} {want:<14} {flag}")

    archs = torch.cuda.get_arch_list()
    sm70_ok = "sm_70" in archs
    print(f"\n  torch arch list: {archs}")
    print(f"  sm_70 present  : {'YES' if sm70_ok else 'NO -- V100 will not work'}")

    # ─── 2. GPU visibility ────────────────────────────────────────────
    section("2. GPU access from inside container")
    n_gpus = torch.cuda.device_count()
    print(f"  CUDA devices visible: {n_gpus}")
    if n_gpus == 0:
        print("  FAIL: no GPUs visible. Check `--gpus all` in docker run.")
        sys.exit(1)
    for i in range(n_gpus):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        print(f"    GPU {i}: {name} (cap {cap})")

    # Confirm we can actually init CUDA (catches driver/lib mismatch).
    try:
        torch.cuda.init()
        # Allocate a small tensor on each visible GPU.
        x = torch.zeros(1024, device="cuda:0") + 1.0
        torch.cuda.synchronize()
        print(f"  Alloc+sync on cuda:0: OK ({x.sum().item():.0f})")
    except Exception as e:
        print(f"  FAIL: CUDA init/alloc error: {e}")
        sys.exit(1)

    # ─── 3. Our kernel compiles + runs against torch 2.10 + cu128 ──
    section("3. JIT-compile fp8_dequant.cu under cu128 / torch 2.10")
    print("  Compiling (may take ~30s on first run, cached after)...")
    try:
        ext = _load_kernel(name="fp8_dequant_ext_dev")
        print("  Compile: OK")
    except Exception as e:
        print(f"  FAIL: compile error: {e}")
        sys.exit(1)

    # Smoke test: one known FP8 byte -> known FP16 value.
    test_cases = [
        (0x00,  0.0,         "+0"),
        (0x38,  1.0,         "+1.0"),
        (0xB8, -1.0,         "-1.0"),
        (0x3C,  1.5,         "+1.5"),
        (0x7E,  448.0,       "+max"),
        (0x01,  0.001953125, "+min subnormal"),
    ]
    print("\n  Phase 1 smoke test (FP8 -> FP16):")
    bytes_t = torch.tensor([b for b, _, _ in test_cases], dtype=torch.uint8, device="cuda")
    out = ext.fp8_e4m3_to_fp16(bytes_t).cpu()
    all_ok = True
    for i, (byte, expected, label) in enumerate(test_cases):
        got = out[i].item()
        ok = abs(got - expected) < 1e-9 or (got == expected)
        all_ok &= ok
        flag = "OK" if ok else "MISMATCH"
        print(f"    0x{byte:02X} ({label:<14}) -> {got:>12.6f}  (expect {expected:>10.6f})  {flag}")

    # Phase 4 smoke test: real 2D-block dequant via our kernel.
    print("\n  Phase 4 smoke test (block-scaled dequant, small random weight):")
    N, K = 128, 256
    block_h = block_w = 128
    torch.manual_seed(0)
    w_real = (torch.randn(N, K) * 0.1).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    w_u8   = w_real.view(torch.uint8).reshape(-1).cuda().contiguous()
    scales = torch.rand(N // block_h, K // block_w, dtype=torch.float16, device="cuda")
    out2 = ext.fp8_e4m3_to_fp16_block_scaled(
        w_u8, scales.reshape(-1).contiguous(), N, K, block_h, block_w
    ).reshape(N, K)
    cpu_ref = w_real.to(torch.float16) * scales.cpu().repeat_interleave(block_h, 0).repeat_interleave(block_w, 1)
    diff = (out2.cpu().float() - cpu_ref.float()).abs().max().item()
    print(f"    GPU vs CPU max abs diff: {diff:.6e}")
    block_ok = diff < 1e-3
    print(f"    {'OK' if block_ok else 'FAIL'}")

    # ─── Summary ──────────────────────────────────────────────────────
    section("Summary")
    overall = all_match and sm70_ok and all_ok and block_ok
    print(f"  versions match    : {'YES' if all_match else 'NO'}")
    print(f"  sm_70 in archs    : {'YES' if sm70_ok else 'NO'}")
    print(f"  GPU init+alloc    : YES")
    print(f"  kernel compiles   : YES")
    print(f"  Phase 1 smoke     : {'PASS' if all_ok else 'FAIL'}")
    print(f"  Phase 4 smoke     : {'PASS' if block_ok else 'FAIL'}")
    print()
    print(f"  OVERALL: {'PASS — dev image ready for vllm integration' if overall else 'FAIL — investigate above'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
