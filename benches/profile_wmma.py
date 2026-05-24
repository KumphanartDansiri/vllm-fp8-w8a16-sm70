"""
profile_wmma.py
───────────────
Single-purpose: run only the WMMA POC kernel a few times so nsight-compute
can capture metrics. Used to identify the actual bottleneck behind the
22 TFLOP/s ceiling (after padding didn't help).

Run via:
    ./run_docker.sh ncu-wmma
which wraps:
    ncu --kernel-name fp8_w8a16_gemm_wmma_kernel \
        --launch-skip 0 --launch-count 1 --set full \
        -o /work/wmma_profile python3 profile_wmma.py
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
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/mnt/models/Qwen3.6-27B-FP8"))
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
M = int(os.environ.get("PROFILE_M", "512"))
RANK = 0
BLOCK = 128


def main():
    assert torch.cuda.is_available()
    print(f"Device: {torch.cuda.get_device_name(0)}  cap {torch.cuda.get_device_capability(0)}")
    print(f"Profiling shape: M={M}, gate_up_proj per-rank at TP={TP_SIZE}", flush=True)

    print("Compiling kernel ...", flush=True)
    ext = load(
        name="fp8_dequant_ext_profile",
        sources=[str(HERE / "fp8_dequant.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )
    print("Compiled.", flush=True)

    # Load gate_up_proj from layer 0 (col-parallel, large N+K)
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
    w_fp8 = torch.cat([shard_col(gp, TP_SIZE, RANK), shard_col(up, TP_SIZE, RANK)], dim=0)
    s_bf16 = torch.cat([shard_col(gp_s, TP_SIZE, RANK), shard_col(up_s, TP_SIZE, RANK)], dim=0)

    dev = torch.device("cuda:0")
    w_fp8 = w_fp8.to(dev)
    s_fp16 = s_bf16.to(torch.float16).contiguous().to(dev)
    N, K = w_fp8.shape
    w_u8_flat = w_fp8.view(torch.uint8).reshape(-1).contiguous()
    s_flat = s_fp16.reshape(-1).contiguous()
    print(f"  per-rank N={N}, K={K}", flush=True)

    torch.manual_seed(0)
    x = (torch.randn(M, K, device=dev) * 0.1).to(torch.float16).contiguous()

    # Warmup (not profiled — --launch-skip should cover anyway)
    for _ in range(5):
        _ = ext.fp8_w8a16_gemm_wmma_poc(x, w_u8_flat, s_flat, N, K, BLOCK, BLOCK)
    torch.cuda.synchronize()
    print("Warmup done.", flush=True)

    # The profiled call(s) — ncu --launch-count 1 captures the first.
    for _ in range(3):
        _ = ext.fp8_w8a16_gemm_wmma_poc(x, w_u8_flat, s_flat, N, K, BLOCK, BLOCK)
    torch.cuda.synchronize()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
