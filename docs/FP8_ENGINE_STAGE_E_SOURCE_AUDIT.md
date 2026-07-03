# Stage E (1/2) — engine source audit: is SM70 FP8 upstream lmdeploy or 1catai's?

Branch `fp8-engine-stage-e-adapter`. Resolves the open sourcing question before writing the
adapter (Codex: keep source audit and adapter as separate commits).

## Question
Is the SM70 s884 **FP8 (E4M3)** GEMM engine part of **upstream InternLM/lmdeploy**, or is it a
**1catai addition** on top? The answer decides where we source the engine (vendor 1catai's fork
vs build against upstream lmdeploy) and how we credit it.

## Method
1. Enumerated the SM70 s884 surface in 1catai's **vendored** lmdeploy
   (`~/1catai-vllm/lmdeploy/src/turbomind/kernels/gemm`): `arch/config_sm70_s884.h`,
   `arch/operand_sm70_s884.h`, `arch/mma_sm70.h`, `mainloop_sm70.h`, `iterator_sm70.h`,
   `scheduler_sm70.cuh`, `kernel/sm70_884_{4,8,16}.cu`, `convert_v3.cu` (the FP8 layout converter).
   Provenance markers (version.txt / .git) were **stripped** from the vendored copy.
2. Found FP8 is a **first-class config**, not a bolt-on: `config_sm70_s884.h` defines
   `Config_E4M3 = Sm70_s884<..., Operand_B_Pack<fp8_e4m3_t>, ...>` in the *same* template family
   as `Config_U4_d/_g` (AWQ Int4), `Config_MXF4`, `Config_F16`; `kernel/sm70_884_8.cu` registers
   `Config_E4M3<kColMajor,0>`.
3. Compared against **upstream** `InternLM/lmdeploy@main` via raw GitHub:
   - `arch/config_sm70_s884.h` — **upstream defines `Config_E4M3`** (verbatim
     `Operand_B_Pack<fp8_e4m3_t>` … same as vendored).
   - `kernel/sm70_884_8.cu` — **upstream `Registry::sm70_884_8()` registers `Config_E4M3`** (built, not just defined).

## Finding
**The SM70 FP8 (E4M3) s884 GEMM engine is UPSTREAM lmdeploy** (defined *and* registered on `main`),
alongside the long-standing Volta AWQ-Int4 s884 path. It is **not** a 1catai invention.

**1catai's actual delta is the vLLM integration layer**, not the kernel:
- `csrc/quantization/awq/awq_sm70_gemm.cu` wrappers: `fp8_sm70_prepare`, `fp8_gemm_sm70_out`
  (+`_auto`/`_meta`), `fp8_moe_gemm_sm70_out`, `awq_moe_build_strided_ptrs`;
- vLLM quant methods (`fp8_sm70_moe.py`) + torch bindings.
- **The Stage-A `_auto` garbage bug lives in this 1catai wrapper, NOT in the lmdeploy engine**
  (the engine's `convert_v3.cu` packs `Ddesc.ld` correctly; the wrapper's `_auto` fails to thread it).

## Implication for sourcing (Stage E adapter + Stage G)
- **Source the engine directly from upstream lmdeploy** (Apache-2.0, clean provenance/credit) — do
  **not** need to vendor 1catai's fork to get the kernel.
- **1catai = wiring reference only**: how to call prepare→meta→grouped_gemm, build strided ptrs, and
  sequence the MoE apply. We re-implement that thin adapter ourselves, correctly (never `_auto`).
- Sourcing from upstream also **sidesteps the `_auto` wrapper bug** by construction.
- The engine underneath is still a **subsystem** (the `turbomind/kernels/gemm` tree + CUTLASS), so the
  build cost stands — but it's upstream-lmdeploy's subsystem, vendored/submoduled from the clean source.

## Residual (minor, before pinning)
- [ ] Pin a specific lmdeploy **release/tag** that includes `Config_E4M3` sm70 (main has it; find the
  earliest stable tag) rather than a moving `main`.
- [ ] Full file-level diff vendored-1catai-lmdeploy ↔ that tag to confirm no *essential* 1catai patches
  to the gemm subsystem (expected minor/none — the FP8 path is architecturally upstream). If 1catai
  did patch the engine (not just the wrapper), capture those as the real deltas to carry.

## Verdict
Engine source question **RESOLVED**: upstream lmdeploy carries the SM70 FP8 engine. Build the adapter
against upstream lmdeploy; treat 1catai as the vLLM-wiring reference and correctness oracle, not the
kernel source. Proceed to Stage E (2/2): the `VLLM_V100_FP8_BACKEND` adapter.
