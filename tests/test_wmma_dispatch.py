"""
test_wmma_dispatch.py
─────────────────────
Sanity check that the FP8W8A16Linear / serve dispatch with WMMA produces
numerically correct results across the M regimes:
  - M=1      decode (A.3 k=8)
  - M=32     short prefill (A.1)
  - M=64     WMMA aligned, no tail
  - M=100    WMMA(64) + A.2(36) tail
  - M=128    WMMA aligned, no tail
  - M=200    WMMA(192) + A.2(8) tail

Compares against torch.matmul on pre-dequant FP16 weights.

Run:
    ./run_docker.sh dev-test test_wmma_dispatch.py
"""
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

try:
    from safetensors import safe_open
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "safetensors"])
    from safetensors import safe_open


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fp8_w8a16_module import FP8W8A16Linear  # noqa: E402

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/mnt/models/Qwen3.6-27B-FP8"))
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
RANK = 0
BLOCK = 128


def main():
    assert torch.cuda.is_available()
    print(f"Device: {torch.cuda.get_device_name(0)}  cap {torch.cuda.get_device_capability(0)}")
    dev = torch.device("cuda:0")

    print("Compiling kernel ...", flush=True)
    ext = load(
        name="fp8_dequant_ext_wmma_test",
        sources=[str(HERE / "fp8_dequant.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )
    print("Compiled.\n", flush=True)
    assert hasattr(ext, "fp8_w8a16_gemm_wmma_poc"), "WMMA POC binding missing!"

    # Load gate_up_proj rank-0 shard (col-parallel, big)
    wmap = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]

    def fetch(name):
        with safe_open(MODEL_DIR / wmap[name], framework="pt") as f:
            return f.get_tensor(name)

    def shard_col(t, tp, rank):
        n = t.shape[0]
        per = n // tp
        return t[rank * per:(rank + 1) * per, :].contiguous()

    gp   = fetch("model.language_model.layers.0.mlp.gate_proj.weight")
    gp_s = fetch("model.language_model.layers.0.mlp.gate_proj.weight_scale_inv")
    up   = fetch("model.language_model.layers.0.mlp.up_proj.weight")
    up_s = fetch("model.language_model.layers.0.mlp.up_proj.weight_scale_inv")

    w_fp8  = torch.cat([shard_col(gp, TP_SIZE, RANK), shard_col(up, TP_SIZE, RANK)], dim=0).to(dev)
    s_bf16 = torch.cat([shard_col(gp_s, TP_SIZE, RANK), shard_col(up_s, TP_SIZE, RANK)], dim=0)
    s_fp16 = s_bf16.to(torch.float16).contiguous().to(dev)
    N, K = w_fp8.shape
    print(f"Layer shape (gate_up_proj per-rank): N={N} K={K}\n", flush=True)

    # FP16 reference: pre-dequant W
    Nb, Kb = s_fp16.shape
    i_idx = (torch.arange(N, device=dev) // BLOCK).clamp_max(Nb - 1)
    j_idx = (torch.arange(K, device=dev) // BLOCK).clamp_max(Kb - 1)
    scale_grid = s_fp16[i_idx][:, j_idx]
    w_fp16 = (w_fp8.view(torch.float8_e4m3fn).to(torch.float16) * scale_grid).contiguous()

    # Build module — uses our dispatch internally (FP8W8A16Linear expects 2D)
    w_u8_2d = w_fp8.view(torch.uint8).contiguous()       # [N, K] uint8
    s_2d    = s_fp16.contiguous()                         # [Nb, Kb] fp16
    mod = FP8W8A16Linear(ext, w_u8_2d, s_2d, BLOCK, BLOCK, bias=None)

    # Test matrix
    test_cases = [
        ("M=1     decode",       1),
        ("M=8     A.3 boundary", 8),
        ("M=32    short prefill",32),
        ("M=63    < WMMA tile",  63),
        ("M=64    WMMA aligned", 64),
        ("M=65    WMMA+tail(1)", 65),
        ("M=100   WMMA+tail(36)",100),
        ("M=128   WMMA aligned",128),
        ("M=200   WMMA+tail(8)", 200),
        ("M=512   WMMA aligned",512),
    ]

    print(f"{'case':<26} {'M':>4}  {'max_abs':>10}  {'max_rel':>10}  {'status':<6}")
    print("-" * 70)
    all_pass = True
    for label, M in test_cases:
        torch.manual_seed(M)
        x = (torch.randn(M, K, device=dev) * 0.1).to(torch.float16).contiguous()
        out = mod(x)
        ref = torch.matmul(x, w_fp16.t())
        max_abs = (out - ref).abs().max().item()
        ref_max = ref.abs().max().item() + 1e-6
        max_rel = max_abs / ref_max
        ok = max_rel < 0.10
        status = "PASS" if ok else "FAIL"
        all_pass = all_pass and ok
        print(f"{label:<26} {M:>4}  {max_abs:>10.4e}  {max_rel:>10.4e}  {status:<6}")

    print()
    if all_pass:
        print("All correctness checks PASS.")
        sys.exit(0)
    else:
        print("FAIL — at least one M did not match torch reference.")
        sys.exit(1)


if __name__ == "__main__":
    main()
