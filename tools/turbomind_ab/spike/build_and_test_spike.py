#!/usr/bin/env python3
"""
Stage-F build spike: prove the upstream-lmdeploy SM70 FP8 gemm builds in OUR cu126 image
and still passes the Stage-D real-Qwen gate.

Superset-first: reuse 1catai's PROVEN sm70-trimmed source list (custom tm_registry_sm70.cu,
only sm70_884 kernels) verbatim, retargeted to our image via torch cpp_extension. If it fights
the build, we do NOT over-trim on the first pass — we fix the flag/include, prove feasibility,
prune later.

Mounts expected in-container:
  /work   = our repo            /catai = ~/1catai-vllm (engine + binding source, ro)
CUTLASS   = the image's vendored copy (vLLM ships it).
"""
import glob, os, struct, json, sys, time
import torch
from torch.utils.cpp_extension import load

LM = "/catai/lmdeploy"
CAT = "/catai/csrc"
T = f"{LM}/src/turbomind"
G = f"{T}/kernels/gemm"
SPIKE = "/work/tools/turbomind_ab/spike"

# CUTLASS include dir present in vllm-v100:vllm021-cu126 (probe a few known locations)
CUTLASS_CANDIDATES = [
    "/vllm-src/.deps/qutlass-src/third_party/cutlass/include",
    "/usr/local/lib/python3.12/dist-packages/tilelang/3rdparty/cutlass/include",
    "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include",
]
CUTLASS_INC = next((c for c in CUTLASS_CANDIDATES if os.path.exists(c + "/cutlass/cutlass.h")), None)
if not CUTLASS_INC:
    sys.exit("[FATAL] no CUTLASS include found in image")
print(f"[spike] CUTLASS include = {CUTLASS_INC}")

SOURCES = [
    f"{SPIKE}/spike_binding.cpp",
    f"{CAT}/quantization/awq/awq_sm70_gemm.cu",
    f"{CAT}/quantization/awq/tm_registry_sm70.cu",
    f"{T}/core/check.cc", f"{T}/core/layout.cc", f"{T}/core/context.cc",
    f"{T}/core/allocator.cc", f"{T}/core/buffer.cc", f"{T}/core/stream.cc",
    f"{T}/utils/logger.cc", f"{T}/utils/parser.cc",
    f"{G}/gemm.cu", f"{G}/kernel.cu", f"{G}/dispatch_cache.cu", f"{G}/context.cu",
    f"{G}/convert_v3.cu", f"{G}/cast.cu", f"{G}/unpack.cu",
    f"{G}/tuner/cache_utils.cu", f"{G}/tuner/measurer.cu", f"{G}/tuner/sampler.cu",
    f"{G}/tuner/stopping_criterion.cc", f"{G}/tuner/params.cc",
    f"{G}/kernel/sm70_884_4.cu", f"{G}/kernel/sm70_884_8.cu", f"{G}/kernel/sm70_884_16.cu",
]
for s in SOURCES:
    if not os.path.exists(s):
        sys.exit(f"[FATAL] missing source {s}")

print(f"[spike] compiling {len(SOURCES)} sources for sm_70 (this is long)...")
t0 = time.time()
mod = load(
    name="fp8sm70_spike",
    sources=SOURCES,
    extra_include_paths=[LM, CUTLASS_INC],
    extra_cflags=["-std=c++17", "-O2", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED"],
    extra_cuda_cflags=[
        "-std=c++17", "-O2",
        "-gencode=arch=compute_70,code=sm_70",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    is_python_module=False,   # TORCH_LIBRARY-only ext: load via op registry, no PyInit_
    verbose=True,
)
print(f"[spike] BUILD OK in {time.time()-t0:.0f}s")

# ---- Stage-D real-Qwen gate against the spike ops ----
dev = "cuda:0"
BLOCK = 128
MODEL = os.environ.get("MODEL_DIR", "/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8")


def cossim(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def dequant(w, s, block=BLOCK):
    N, K = w.shape
    e = s.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return w.float() * e


from safetensors import safe_open
import re
files = sorted(glob.glob(f"{MODEL}/*.safetensors"))
k2f = {}
for f in files:
    with safe_open(f, framework="pt") as fh:
        for k in fh.keys():
            k2f[k] = f
L = sorted({int(m.group(1)) for k in k2f
            if (m := re.search(r"layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight$", k))})[0]
pre = next(k.split(".experts.0.")[0] for k in k2f
           if f"layers.{L}.mlp.experts.0.gate_proj.weight" in k)


def load_t(k):
    with safe_open(k2f[k], framework="pt") as fh:
        return fh.get_tensor(k)


gw = load_t(f"{pre}.experts.0.gate_proj.weight"); gs = load_t(f"{pre}.experts.0.gate_proj.weight_scale_inv")
uw = load_t(f"{pre}.experts.0.up_proj.weight");   us = load_t(f"{pre}.experts.0.up_proj.weight_scale_inv")
w13 = torch.cat([gw, uw], 0).to(dev).contiguous()
w13_s = torch.cat([gs, us], 0).float().to(dev).contiguous()
N13, K13 = w13.shape
print(f"[spike] real Qwen expert w13={tuple(w13.shape)} scale={tuple(w13_s.shape)}")

W_dq = dequant(w13, w13_s)
tm_w, tm_s, meta = torch.ops.fp8sm70_spike.fp8_sm70_prepare(w13, w13_s, BLOCK)
k_ld, q_ld = int(meta[0]), int(meta[1])
print(f"[spike] prepare OK  k_ld={k_ld} q_ld={q_ld}")

ok = True
print(f"{'M':>4} {'cos':>10} {'max_abs':>12}")
for M in (1, 4, 16):
    g = torch.Generator().manual_seed(100 + M)
    A = (torch.randn(M, K13, generator=g) * 0.1).half().to(dev)
    ref = A.float() @ W_dq.T
    out = torch.empty(M, N13, dtype=torch.float16, device=dev)
    torch.ops.fp8sm70_spike.fp8_gemm_sm70_out(out, A, tm_w, tm_s, BLOCK, k_ld, q_ld)
    c = cossim(out, ref); e = (out.float() - ref).abs().max().item()
    print(f"{M:>4} {c:>10.4f} {e:>12.3e}")
    ok = ok and c >= 0.99

print(f"\n=== SPIKE {'PASS' if ok else 'FAIL'}: upstream-lmdeploy SM70 FP8 builds in cu126 "
      f"+ real-Qwen gate cos>=0.99 = {ok} ===")
sys.exit(0 if ok else 1)
