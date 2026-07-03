# PROVENANCE — vendored TurboMind SM70 FP8 GEMM engine

This directory vendors a **minimal subtree** of the TurboMind GEMM engine so our vLLM can run
the SM70 FP8 (E4M3) s884 kernels without depending on an external checkout. Licensed Apache-2.0
(see `LICENSE`, copied from upstream lmdeploy).

## Base pin
- **Upstream:** `InternLM/lmdeploy` tag **`v0.14.0`** (commit `5d4fac9`, 2026-06-24).
- Verified at that tag: `Config_E4M3 = Sm70_s884<..., Operand_B_Pack<fp8_e4m3_t>, ...>` is defined in
  `src/turbomind/kernels/gemm/arch/config_sm70_s884.h` and registered in
  `src/turbomind/kernels/gemm/kernel/sm70_884_8.cu` (see `docs/FP8_ENGINE_STAGE_E_SOURCE_AUDIT.md`).

## Vendored files (`src/turbomind/`)
- `kernels/gemm/**` (the GEMM engine; sm90/sm80/sm75/tma sources are present but NOT compiled — the
  sm70-only build list is in `build_and_gate.py`; `test/` dropped).
- `kernels/core/**`, `kernels/attention/quantization.h` (the single gemm↔attention coupling header),
  `core/**`, `utils/**`, `macro.h` — the transitive deps of the sm70 gemm path.
- `3rdparty/moodycamel/` — moodycamel concurrentqueue headers (header-only, own LICENSE), which
  v0.14.0's `core/logger.cc` uses (`#include <blockingconcurrentqueue.h>`). Upstream lmdeploy
  FetchContents this; we vendor the 3 headers. From `github.com/cameron314/concurrentqueue`.

## Deltas from pure v0.14.0 (the "V100 patch" — carried deliberately)
Source of the deltas: **`1catai/1Cat-vLLM`** (Apache-2.0), whose lmdeploy base is slightly older than
v0.14.0. Diff established the FP8 kernel itself (`sm70_884_8.cu`, `config_sm70_s884.h`) is **byte-identical**
to v0.14.0; the only functional deltas we carry are V100 (sm_70) tuning:
1. **`kernels/gemm/gemm.cu`** — an `if (arch_ == 700) { tuning_.max_splits=16; max_waves=32; swizzle=…; }`
   block in the `Gemm::Impl` constructor (broader autotuner search for V100's many tiny GEMM/GEMV decode
   problems). Applied on top of v0.14.0's file; bracketed by `>>> V100 PATCH` / `<<< V100 PATCH`. All
   `tuning_` fields verified present in v0.14.0 `tuner/params.h`.
2. **`kernels/gemm/kernel/sm70_884_4.cu` + `sm70_884_16.cu`** — 1catai's variants (extra sm70 tile-config
   registrations, incl. a GroupSizeV=128 config). `sm70_884_8.cu` kept as v0.14.0 (identical).
3. **`kernels/core/mma.h`** — added `#include <cuda_bf16.h>` (build-enablement). Without it the bf16
   `mma_m16n8k8_row_col` overloads have undefined `__nv_bfloat16` (`<error-type>`) in a standalone
   cpp_extension build, causing an overload-ambiguity failure in `sm70_884_*.cu`. Marked inline.

Everything else is pure v0.14.0. `thread_map.h` differs from 1catai (v0.14.0 is newer — added a `WarpC`
template param) but is self-consistent and kept as v0.14.0.

## Binding (`binding/`)
- `awq_sm70_gemm.cu` + `tm_registry_sm70.cu` — the vLLM integration layer from `1catai/1Cat-vLLM`
  (Apache-2.0). This is NOT upstream lmdeploy; it wraps the engine's public API
  (`gemm.h`/`convert.h`/`cast.h`/`types.h`/`utils.h`, all byte-identical to v0.14.0 → compiles unchanged).
- **`fp8_gemm_sm70_out_auto` has been REMOVED** from `awq_sm70_gemm.cu` (both the impl and the wrapper) —
  it silently returns garbage (Stage-A). Contract is `prepare -> meta(k_ld,q_ld) -> gemm`; never `_auto`.
- `fp8_sm70_bindings.cpp` — our own thin torch registration (dense `fp8_sm70_prepare` + `fp8_gemm_sm70_out`;
  MoE ops added in a follow-up commit).

## Build requirements (see `build_and_gate.py`)
- **CUTLASS** headers (header-only) — taken from the image's vendored copy at build time.
- **fmt** — v0.14.0's `TM_LOG` uses {fmt}; the image has no `libfmt`, so we build with
  `-DFMT_HEADER_ONLY` + the image's fmt headers (no link symbol needed).
- **CUDA driver** — link `-lcuda` (stub at link time; real driver via `--gpus` at run time) for
  `cuGetErrorString` etc.
- `-gencode arch=compute_70,code=sm_70`, `-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED`, C++17.
- sm70-only source list (all archs' headers vendored, but only `sm70_884_{4,8,16}` + shared `.cu`
  compiled; the non-sm70 kernel `.cu` and `test/` are pruned from the tree).

## Re-sync steps
1. `git clone --depth 1 --branch <tag> https://github.com/InternLM/lmdeploy` (currently v0.14.0).
2. Copy `src/turbomind/{core,utils,macro.h,kernels/{core,gemm,attention}}` here (drop `test/`).
3. Re-apply the two V100 deltas above (grep `>>> V100 PATCH`; re-copy 1catai's `sm70_884_4/16.cu`).
4. Re-remove `fp8_gemm_sm70_out_auto` from `binding/awq_sm70_gemm.cu`.
5. `python3 build_and_gate.py` → expect real-Qwen dense gate cos=1.0000.
