#!/usr/bin/env python3
"""
Shared builder for the vendored TurboMind SM70 FP8 engine (dense + MoE ops).
Compiles the sm70-only source list from THIS repo's third_party/ via torch cpp_extension
in the cu126 image; returns `torch.ops.turbomind_fp8_sm70`. See PROVENANCE.md for the pin
(lmdeploy v0.14.0 + V100 deltas) and build requirements.
"""
import os, sys, time
import torch
from torch.utils.cpp_extension import load

BLOCK = 128
TP = os.path.dirname(os.path.abspath(__file__))          # third_party/turbomind_gemm_sm70
S = f"{TP}/src/turbomind"
G = f"{S}/kernels/gemm"

_CUTLASS_CANDIDATES = [
    "/vllm-src/.deps/qutlass-src/third_party/cutlass/include",
    "/usr/local/lib/python3.12/dist-packages/tilelang/3rdparty/cutlass/include",
    "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include",
]
_FMT_CANDIDATES = [
    "/vllm-src/.deps/deepgemm-src/third-party/fmt/include",
    "/usr/local/lib/python3.12/dist-packages/torch/include",
]

# v0.14.0 build source list (sm70-only): binding + engine core/utils + gemm .cu + sm70 kernels.
# v0.14.0's TM_LOG uses {fmt} (header-only) + async logger (moodycamel).
_CORE_CC = ["allocator", "buffer", "check", "context", "copy", "data_format", "layout",
            "logger", "module", "registry", "scope", "stream", "tensor"]
_UTILS_CC = ["cuda_utils", "nvtx_utils", "parser"]
_GEMM_CU = ["gemm", "kernel", "dispatch_cache", "context", "convert_v3", "cast", "unpack"]
_TUNER = ["tuner/cache_utils.cu", "tuner/measurer.cu", "tuner/sampler.cu",
          "tuner/stopping_criterion.cc", "tuner/params.cc"]


def build_ops():
    """Compile (or load from cache) the vendored engine; return torch.ops.turbomind_fp8_sm70."""
    cutlass = next((c for c in _CUTLASS_CANDIDATES if os.path.exists(c + "/cutlass/cutlass.h")), None)
    if not cutlass:
        sys.exit("[FATAL] no CUTLASS include found in image")
    fmt = next((c for c in _FMT_CANDIDATES if os.path.exists(c + "/fmt/format.h")), None)
    if not fmt:
        sys.exit("[FATAL] no fmt headers found in image")

    sources = (
        [f"{TP}/binding/fp8_sm70_bindings.cpp",
         f"{TP}/binding/awq_sm70_gemm.cu",
         f"{TP}/binding/tm_registry_sm70.cu"]
        + [f"{S}/core/{n}.cc" for n in _CORE_CC]
        + [f"{S}/utils/{n}.cc" for n in _UTILS_CC]
        + [f"{G}/{n}.cu" for n in _GEMM_CU]
        + [f"{G}/{t}" for t in _TUNER]
        + [f"{G}/kernel/sm70_884_{k}.cu" for k in (4, 8, 16)]
    )
    missing = [s for s in sources if not os.path.exists(s)]
    if missing:
        sys.exit("[FATAL] missing sources:\n  " + "\n  ".join(missing))

    print(f"[vendor] compiling {len(sources)} sources from third_party/ for sm_70 ...")
    t0 = time.time()
    load(
        name="turbomind_fp8_sm70",
        sources=sources,
        extra_include_paths=[TP, cutlass, f"{TP}/3rdparty/moodycamel", fmt],
        extra_cflags=["-std=c++17", "-O2", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DFMT_HEADER_ONLY"],
        extra_cuda_cflags=[
            "-std=c++17", "-O2", "-gencode=arch=compute_70,code=sm_70",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DFMT_HEADER_ONLY",
            "--expt-relaxed-constexpr", "--expt-extended-lambda",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        extra_ldflags=["-L/usr/local/cuda/lib64/stubs", "-lcuda"],  # CUDA driver API
        is_python_module=False,
        verbose=True,
    )
    print(f"[vendor] BUILD OK in {time.time()-t0:.0f}s")
    return torch.ops.turbomind_fp8_sm70
