# Stage B — kernel lineage & adoptability matrix (ai-bond vs 1catai)

Time-boxed comparative read (2026-07-03). Read-only; no rebuild. Goal is an adoption
decision aid, not exhaustive archaeology.

Local clones read:
- `~/aibond-vllm-v100` — github ai-bond/vllm-v100, HEAD `3e92043` (2026-05-31)
- `~/1catai-vllm` — github 1CatAI/1Cat-vLLM, HEAD `f1a64a76f`

## Matrix

| Dimension | **ai-bond/vllm-v100** | **1catai/1Cat-vLLM** |
|---|---|---|
| Base vLLM | fork of **v0.19.1** (README) | vLLM fork (0.19-era), vendors lmdeploy |
| Quant coverage | **AWQ Int4 only** (no FP8 weight GEMM) | **AWQ Int4 + FP8 W8A16** (dense **and** grouped MoE) |
| Kernel origin | **Own kernel** — `gemm_forward_4bit_cuda_m16n16k16` (Volta `m16n16k16` tensor-core GEMM w/ inline int4 dequant). Dequant uses the classic AWQ `0x64006400` fp16-inject trick (technique lineage = llm-awq) but the Volta GEMM is hand-written. | **Adapted from LMDeploy TurboMind s884** (header: *"Adapted from LMDeploy TurboMind (Apache-2.0)"*); the FP8 path *is* the TurboMind s884 HMMA GEMM + layout converters. |
| Engine vendoring | **None.** 2 self-contained files: `csrc/quantization/awq/gemm_kernels.cu` (32 KB) + `dequantize.cuh` (6 KB). | **Vendors lmdeploy** (`~/1catai-vllm/lmdeploy`, 3.1 MB, 33 turbomind gemm `.cu`); CUTLASS pulled via FetchContent. sm70 kernel has **224** `turbomind::` references — deeply coupled. |
| Exported op surface | Overrides the **standard** `awq_gemm` / `awq_dequantize` ops (same signatures as upstream). **No new op names.** | **17 new `*_sm70` ops**: `awq_gemm_sm70{,_out}`, `awq_moe_gemm_sm70{,_out}`, `awq_moe_single_token_sm70_out`, `awq_sm70_prepare`, `fp8_sm70_prepare`, `fp8_gemm_sm70_out{,_auto,_meta}`, `fp8_moe_gemm_sm70_out`, `sm70_f16_{gemm,gemm_out,gate_mul_out,prepare}`, `sm70_gemm_{export,import}_cache`. |
| Python plumbing | **~zero** — vLLM's existing AWQ quant method calls the standard op; the Volta kernel is transparent. | Dedicated quant methods/modules: `fp8_sm70_moe.py`, `awq_sm70_moe.py`, `warmup/awq_sm70_warmup.py`, prepare/meta + strided-ptr machinery. |
| vLLM invasiveness | **Minimal / professional** — strips `csrc/cpu`, `csrc/rocm`; `vllm/platforms/` = cuda only; preserves upstream structure. | **Sprawling** — keeps full platform surface (cpu/cuda/rocm/tpu/xpu); large added op + module surface. |
| License (for adoption) | Kernel files: **BSD-3-Clause**, © 2026 D.Skryabin (repo top-level Apache-2.0). | **Apache-2.0** (repo + vendored lmdeploy both Apache-2.0). |
| Correctness (our tests) | not yet gated here (Int4, off-mission) | **Stage A: dense fixed/`_meta` cos=1.0000 M=1..32; grouped-MoE cos gate PASS 18/18** |

## Integration consequence (the decision-relevant column)

- **ai-bond** is trivial to wrap — two self-contained BSD-3 files behind the *standard*
  `awq_gemm` op, no vendored engine, no new Python method. **But it is AWQ-Int4 only**, so
  it does **not** serve our FP8 W8A16 mission. It's the reference for "clean minimal Volta
  fork," and a candidate if/when we want a first-class AWQ path — not an FP8 backend.
- **1catai** is the only FP8 W8A16 option, and its FP8 math is now **proven correct** on
  V100 (Stage A). The cost of adoption is **structural**: FP8 requires the vendored
  TurboMind gemm subsystem (lmdeploy, CUTLASS FetchContent) as a build dependency plus the
  `*_sm70` op surface — you cannot lift "just the kernel." Wrapping cleanly means treating
  the vendored TurboMind gemm dir as an external engine and building a thin vLLM adapter
  over `fp8_sm70_prepare` + `fp8_gemm_sm70_out` / `fp8_moe_gemm_sm70_out`, not merging their
  whole fork.

## Serving contract (carry forward into any TurboMind integration) — FIRM

- **Never call `fp8_gemm_sm70_out_auto` in serving.** It silently returns cos≈0 garbage
  (Stage A root cause: nominal vs 32× packed leading dim). Silent low-cos is the danger,
  not the bug itself.
- **`prepare` → `meta` (k_ld, q_ld) is part of the required engine contract.** The wrapper
  must persist and thread the packed `k_ld/q_ld` explicitly (dense) or via
  `awq_moe_build_strided_ptrs` (MoE), exactly as `fp8_sm70_moe.py` does.
- **Fail loudly** if `meta`/ld is missing or a shape is unsupported — no geometry-based
  reconstruction, no fallback that can produce silently-wrong output.

## Open (defer to Stage C/D)
- Why ai-bond chose AWQ-only is **inferred** from design (single compact Int4 tensor-core
  kernel, no FP8 weight GEMM; single/dual-V100 AWQ target) — README doesn't state it
  explicitly; don't quote it as a documented rationale.
- Stage C perf harness must isolate: kernel-only / prepare-repack / route-materialization /
  scatter-index-add / end-to-end, over spread + hot1 + hot8/skew, tpe ∈ {1,2,4,8}.
