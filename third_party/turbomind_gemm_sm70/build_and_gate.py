#!/usr/bin/env python3
"""
Reproducible build of the vendored TurboMind SM70 FP8 engine — FROM OUR TREE (no external
checkout) — plus the Stage-D real-Qwen dense gate.

Pin: lmdeploy v0.14.0 + carried V100 patch (see PROVENANCE.md). Compiles the sm70-only source
list via torch cpp_extension in our cu126 image; registers dense ops under `turbomind_fp8_sm70`;
round-trips a real Qwen3.5-35B-A3B-FP8 expert vs an fp32 dequant reference (expect cos=1.0000).

Run inside vllm-v100:vllm021-cu126 with /mnt/models mounted ro.
"""
import glob, os, re, sys, time
import torch
from torch.utils.cpp_extension import load

TP = os.path.dirname(os.path.abspath(__file__))          # third_party/turbomind_gemm_sm70
S = f"{TP}/src/turbomind"
G = f"{S}/kernels/gemm"
BLOCK = 128

CUTLASS_CANDIDATES = [
    "/vllm-src/.deps/qutlass-src/third_party/cutlass/include",
    "/usr/local/lib/python3.12/dist-packages/tilelang/3rdparty/cutlass/include",
    "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include",
]
CUTLASS_INC = next((c for c in CUTLASS_CANDIDATES if os.path.exists(c + "/cutlass/cutlass.h")), None)
if not CUTLASS_INC:
    sys.exit("[FATAL] no CUTLASS include found in image")
# v0.14.0's TM_LOG uses {fmt}; no libfmt in the image -> header-only fmt (inline, no link symbol)
FMT_CANDIDATES = [
    "/vllm-src/.deps/deepgemm-src/third-party/fmt/include",
    "/usr/local/lib/python3.12/dist-packages/torch/include",
]
FMT_INC = next((c for c in FMT_CANDIDATES if os.path.exists(c + "/fmt/format.h")), None)
if not FMT_INC:
    sys.exit("[FATAL] no fmt headers found in image")

# v0.14.0 build source list (sm70-only): binding + engine core/utils + gemm .cu + sm70 kernels
CORE_CC = ["allocator", "buffer", "check", "context", "copy", "data_format", "layout",
           "logger", "module", "registry", "scope", "stream", "tensor"]
UTILS_CC = ["cuda_utils", "nvtx_utils", "parser"]
GEMM_CU = ["gemm", "kernel", "dispatch_cache", "context", "convert_v3", "cast", "unpack"]
TUNER = ["tuner/cache_utils.cu", "tuner/measurer.cu", "tuner/sampler.cu",
         "tuner/stopping_criterion.cc", "tuner/params.cc"]
SOURCES = (
    [f"{TP}/binding/fp8_sm70_bindings.cpp",
     f"{TP}/binding/awq_sm70_gemm.cu",
     f"{TP}/binding/tm_registry_sm70.cu"]
    + [f"{S}/core/{n}.cc" for n in CORE_CC]
    + [f"{S}/utils/{n}.cc" for n in UTILS_CC]
    + [f"{G}/{n}.cu" for n in GEMM_CU]
    + [f"{G}/{t}" for t in TUNER]
    + [f"{G}/kernel/sm70_884_{k}.cu" for k in (4, 8, 16)]
)
missing = [s for s in SOURCES if not os.path.exists(s)]
if missing:
    sys.exit("[FATAL] missing sources:\n  " + "\n  ".join(missing))

print(f"[vendor] compiling {len(SOURCES)} sources from third_party/ for sm_70 ...")
t0 = time.time()
load(
    name="turbomind_fp8_sm70",
    sources=SOURCES,
    extra_include_paths=[TP, CUTLASS_INC, f"{TP}/3rdparty/moodycamel", FMT_INC],
    extra_cflags=["-std=c++17", "-O2", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DFMT_HEADER_ONLY"],
    extra_cuda_cflags=[
        "-std=c++17", "-O2", "-gencode=arch=compute_70,code=sm_70",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DFMT_HEADER_ONLY",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    extra_ldflags=["-L/usr/local/cuda/lib64/stubs", "-lcuda"],  # CUDA driver API (cuGetErrorString etc.)
    is_python_module=False,
    verbose=True,
)
print(f"[vendor] BUILD OK in {time.time()-t0:.0f}s")
ops = torch.ops.turbomind_fp8_sm70

# ---- Stage-D real-Qwen dense gate ----
dev = "cuda:0"
MODEL = os.environ.get("MODEL_DIR", "/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8")
from safetensors import safe_open


def cossim(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def dequant(w, s, block=BLOCK):
    N, K = w.shape
    e = s.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return w.float() * e


k2f = {}
for f in sorted(glob.glob(f"{MODEL}/*.safetensors")):
    with safe_open(f, framework="pt") as fh:
        for k in fh.keys():
            k2f[k] = f
L = sorted({int(m.group(1)) for k in k2f
            if (m := re.search(r"layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight$", k))})[0]
pre = next(k.split(".experts.0.")[0] for k in k2f if f"layers.{L}.mlp.experts.0.gate_proj.weight" in k)


def lt(k):
    with safe_open(k2f[k], framework="pt") as fh:
        return fh.get_tensor(k)


gw, gs = lt(f"{pre}.experts.0.gate_proj.weight"), lt(f"{pre}.experts.0.gate_proj.weight_scale_inv")
uw, us = lt(f"{pre}.experts.0.up_proj.weight"), lt(f"{pre}.experts.0.up_proj.weight_scale_inv")
w13 = torch.cat([gw, uw], 0).to(dev).contiguous()
w13_s = torch.cat([gs, us], 0).float().to(dev).contiguous()
N13, K13 = w13.shape
W_dq = dequant(w13, w13_s)
tm_w, tm_s, meta = ops.fp8_sm70_prepare(w13, w13_s, BLOCK)
k_ld, q_ld = int(meta[0]), int(meta[1])
print(f"[vendor] real Qwen w13={tuple(w13.shape)}  prepare OK k_ld={k_ld} q_ld={q_ld}")

ok = True
print(f"{'M':>4} {'cos':>10} {'max_abs':>12}")
for M in (1, 4, 16):
    g = torch.Generator().manual_seed(100 + M)
    A = (torch.randn(M, K13, generator=g) * 0.1).half().to(dev)
    ref = A.float() @ W_dq.T
    out = torch.empty(M, N13, dtype=torch.float16, device=dev)
    ops.fp8_gemm_sm70_out(out, A, tm_w, tm_s, BLOCK, k_ld, q_ld)
    c = cossim(out, ref); e = (out.float() - ref).abs().max().item()
    print(f"{M:>4} {c:>10.4f} {e:>12.3e}")
    ok = ok and c >= 0.99

print(f"\n=== VENDOR GATE {'PASS' if ok else 'FAIL'}: v0.14.0+V100patch builds from our tree "
      f"+ real-Qwen dense cos>=0.99 = {ok} ===")
sys.exit(0 if ok else 1)
