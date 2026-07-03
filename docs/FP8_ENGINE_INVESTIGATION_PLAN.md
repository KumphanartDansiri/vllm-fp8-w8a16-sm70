# SM70 FP8 engine — investigation & evaluation plan (draft, uncommitted)

**Purpose.** Decide, with evidence, whether to adopt an external SM70 quant engine
(1catai/TurboMind FP8, or ai-bond AWQ) as the V100 FP8 backend — or keep/improve our own.
Written at the end of the 2026-07-03 session; entry point for the next (fresh) session.

> **Restraint (firm):** do NOT rebrand the repo as "based on TurboMind"/"based on ai-bond"
> until correctness + grouped-MoE + serving are validated and code is actually integrated.
> Public framing stays: *"working compressed-tensors FP8 W8A16/MoE on V100"* + *"TurboMind is a
> candidate backend."* Never claim a kernel-speed edge — measured FP8 dense primitive says
> TurboMind s884 beats our GEMV at all M (see [[project_turbomind_fp8_sm70]]).

## 0. What we already know (don't re-derive)
- **1catai FP8 s884 > our FP8 GEMV at ALL M** (dense primitive, both cos=1.0): 1.5×@M1 → 11×@M16.
  Correct call = `fp8_gemm_sm70_out(...k_ld,q_ld)`; `fp8_gemm_sm70_out_auto` returns garbage (cos≈0).
- **ai-bond/vllm-v100 = AWQ-4bit only** (no FP8 weight GEMM), own Volta kernel, professional/minimal
  (strips CPU/XPU/ROCM), preserves vLLM structure. HEAD 3e92043 (2026-05-31).
- **1catai/1Cat-vLLM = AWQ + FP8** via vendored lmdeploy TurboMind s884. HEAD f1a64a76f.
- ai-bond ACCEPTED our FA-V100 BLOCK_N straddle finding (separate thread, done).

## 1. Repo map (local clones)
| Local path | Remote | Role |
|---|---|---|
| `~/vllm-fp8-w8a16-sm70` | github KumphanartDansiri (public) | OUR repo: FP8 W8A16 GEMV + FA bridge + harnesses |
| `~/1catai-vllm` | upstream github 1CatAI/1Cat-vLLM | AWQ+FP8 TurboMind fork (benchmarked) |
| `~/aibond-vllm-v100` | github ai-bond/vllm-v100 | AWQ-only Volta fork (**cloned 2026-07-03**) |
| `~/flash-attention-v100` | upstream github ai-bond/flash-attention-v100 | ai-bond FA-2 V100 (integrated) |
| image `1catai-vllm-v100:cu128-fp8sm70` | — | 1catai `_C` built for arch 7.0 (FP8 ops live); reuse, no recompile |
| harness `tools/turbomind_ab/` | — | prepare_inputs / bench_ours / bench_1catai(+_awq) / compare + build recipe |

## 2. Investigation stages (correctness FIRST — nothing skips ahead)

### A. Correctness & engine contract  *(cheap; committed image, no rebuild)*  — **DONE 2026-07-03**
- [x] **Why is `fp8_gemm_sm70_out_auto` garbage?** CONFIRMED (source read + repro). Hypothesis was
  right: the packed leading dim is written by `LayoutConverter::Convert` (dest descriptor passed by
  non-const ref, `convert.h:15-19`; `convert_v3.cu:48` sets `Ddesc.ld = mk2cs(Packing_v2::apply(...))`).
  `prepare` stores it in `meta` (`awq_sm70_gemm.cu:887-889`); fixed/`_meta` thread it back
  (`:1149,:1173`); `_auto` (`:1215-1279`) rebuilds descriptors from geometry only, never runs Convert,
  so it passes the *nominal* ld (`:1277-1278`). Measured: packed `k_ld = 32×K` (not recoverable from
  geometry) → `_auto` reads weight with 32×-wrong stride → cos≈0 at all M; fixed/`_meta` cos=1.0000.
  Repro `tools/turbomind_ab/repro_fp8_auto_bug.py`; report `docs/FP8_SM70_AUTO_BUG_REPORT.md`;
  raw log `tools/turbomind_ab/repro_auto_bug_20260703_052923.log`. Kept OUT of perf discussion.
- [x] **Fixed-path correctness envelope SEALED:** DENSE fixed/`_meta` cos=1.0000 across **M=1..32**
  on square (256²) + four non-square shapes — N>K (1024×512, 1536×1024, 1024×768) and K>N (512×1024)
  — plus 3072×4096; max_abs ~5–9e-4. Packed `k_ld = 32×K` confirmed on every shape.
  Log `tools/turbomind_ab/dense_nonsquare_seal_20260703_053552.log`.
- [x] **Grouped MoE cos gate:** `fp8_moe_gemm_sm70_out` vs FP32 dequant ref — **PASS 18/18, cos=1.0000**
  across w13 (N=2I=1536,K=H=1024) + w2 (N=H=1024,K=I=768), routing = spread / hot / skew, tpe∈{1,2,4}
  (effective decode M), E=8, incl. empty-expert handling. MoE `k_ld` also = 32×K (w13 32768, w2 24576),
  corroborating the root cause. Gate `tools/turbomind_ab/moe_cos_gate.py`; log
  `tools/turbomind_ab/moe_cos_gate_20260703_053120.log`. (Gate is vs a kernel-neutral FP32 dequant
  ref — the A/B vs our grouped coalesced GEMV is a Stage-C *perf* comparison, not a correctness gate.)

### B. Kernel lineage & design (comparative)  — **DONE 2026-07-03 → `docs/FP8_ENGINE_STAGE_B_LINEAGE.md`**
- [x] ai-bond AWQ Volta kernel: **OWN** (`gemm_forward_4bit_cuda_m16n16k16`, BSD-3 © D.Skryabin, no
  vendored engine, 2 files ~38KB; AWQ `0x64006400` dequant-technique lineage). Overrides the STANDARD
  `awq_gemm`/`awq_dequantize` ops → minimal/transparent, preserves vLLM structure (strips cpu/rocm/xpu).
- [x] 1catai FP8/AWQ: **Adapted from vendored lmdeploy TurboMind s884** (3.1MB, 33 gemm .cu, CUTLASS
  FetchContent; 224 `turbomind::` refs). Adds 17 `*_sm70` ops + dedicated Fp8SM70 quant methods →
  invasive; wrapping = treat vendored TurboMind gemm as an external engine + thin adapter.
- [x] ai-bond AWQ-only: **inferred** from design (single compact Int4 tensor-core kernel, no FP8 weight
  GEMM; single/dual-V100 AWQ target) — README does NOT state it explicitly; don't quote as documented.

### C. Performance (only cos=1.0 kernels)  — **DONE 2026-07-03 → `docs/FP8_ENGINE_STAGE_C_FINDINGS.md`**
- [x] Grouped-MoE A/B `fp8_moe_gemm_sm70_out` vs our coalesced GEMV, cross-image, 5-timing split,
  spread/hot1/hot8 × tpe{1,2,4,8}, both Qwen3.5-35B-A3B (E=256,I=512) + GLM-4.5-Air (E=128,I=1408) dims,
  all cos=1.0000. **RESULT: TurboMind faster e2e across the decode envelope both models; kernel faster at
  ALL M (2.2–6.3×). Ours only ~ties at Qwen M=1–2 (skips permute); GLM ours never wins.** Our GEMV is
  per-row (M=1 semantics, O(R)) → loses as M/R grow; permute tax (~0.09ms) is the only thing we save.
- [x] Envelope map: ours' niche = single-user M=1 decode, few active experts, and/or channel-scale
  checkpoints TurboMind's block-128 kernel can't load (GLM-4.5-Air real ckpt = channel W8A8 → Stage-D gap).
- [ ] Optional: full serving A/B (cross-fork, label as such) — deferred; primitive A/B is decisive.

### D. Integration & compatibility  — **format gate DONE 2026-07-03 → `docs/FP8_ENGINE_STAGE_D_FORMAT_GATE.md`**
DIRECTION (user+Codex): make TurboMind/lmdeploy FP8 work INSIDE our vLLM + patch behind the smallest
clean adapter; ours = compat/fallback/control; 1catai = reference oracle (NOT baseline); ai-bond handoff
= later Stage G. Roadmap: **D format/loader → E adapter (`VLLM_V100_FP8_BACKEND`) → F serving → G maintenance.**
- [x] **Format gate PROVEN on real weights:** real Qwen3.5-35B-A3B-FP8 block-FP8 expert round-trips
  `fp8_sm70_prepare`→`fp8_gemm_sm70_out` at **cos=1.0000** with only bf16→fp32 scale cast + gate/up fusion
  (our loader already does both; block dequant `W*scale` [N/128,K/128] matches prepare exactly). Channel-scale
  [N,1] **rejected loudly** → GLM W8A8 falls back to ours. `tools/turbomind_ab/stage_d_format_gate.py`.
- [x] **w2/down round-trip cos=1.0000; full real-ckpt MoE layer (256 experts, top_k=8) cos=1.0000**
  (`stage_d_full_moe.py`). Gotcha: `moe_unpermute` topk_weights must be fp32 (fp16 → cos≈0 garbage).
- [x] **TP block-128 constraint found:** intermediate `I/tp` must be mult of 128 → Qwen (I=512) clean only
  to TP≤4; TP8 (I/tp=64) BREAKS. Property of block-128 + intermediate-TP → constrains BOTH engines;
  Stage F must confirm real TP8 sharding (expert-parallel vs intermediate-sharded vs pad).
- [ ] vLLM serving: TP≤4 clean / TP8 sharding question, cudagraph, routing, real prompts (Stage F).
- **SERVING CONTRACT (firm):** NEVER `fp8_gemm_sm70_out_auto` in serving; `prepare`→`meta`(k_ld,q_ld)
  is a REQUIRED contract — wrapper threads packed ld explicitly (dense) / via `awq_moe_build_strided_ptrs`
  (MoE) or **fails loudly**. No geometry reconstruction. Silent low-cos is the risk, not the bug.

### E. Adapter integration + licensing / adoptability  (Codex: 2 commits — source audit, then adapter)
- [x] **Source audit DONE → `docs/FP8_ENGINE_STAGE_E_SOURCE_AUDIT.md`.** RESOLVED: the SM70 FP8 (E4M3)
  s884 engine is **UPSTREAM InternLM/lmdeploy** (`Config_E4M3` defined in `config_sm70_s884.h` AND
  registered in `kernel/sm70_884_8.cu` on `main`). 1catai's delta = the vLLM wrapper/quant-method/bindings
  (where the `_auto` bug lives), NOT the kernel. → **source the engine from upstream lmdeploy (Apache-2.0),
  write our OWN thin adapter, 1catai = wiring reference only.** Residual: pin a lmdeploy release tag w/
  Config_E4M3 + full diff for any essential 1catai engine patches (expected none/minor).
- [x] **Adapter (commit 2) DONE (scaffolding) → `src/fp8_w8a16_sm70/turbomind_fp8_backend.py` +
  `docs/FP8_ENGINE_STAGE_E_ADAPTER.md`.** `VLLM_V100_FP8_BACKEND=ours|turbomind|auto`; full eligibility
  predicate (block-FP8 gs128 + fp32 scales + local block-128 align incl. `I/tp%128==0` + k_ld/q_ld +
  op-present + not channel/tensor + never `_auto`); `mode=turbomind` on ineligible RAISES (no silent
  fallback). Self-test PASS; real-image check: ops absent → all block-FP8 → ours w/ reason (ours path
  unchanged). Wrappers `prepare/gemm_out/moe_gemm_out` thread packed ld, never `_auto`.
  REMAINING (Stage F): build the engine into the image so turbomind goes live; wire select_backend into
  `compressed_tensors_v100.py`; serving validation.
- [ ] License: ai-bond FA = BSD-3; `~/1catai-vllm` + lmdeploy = Apache-2.0 → adopt/credit path.
- [ ] License of each: ai-bond FA = BSD-3; check `~/aibond-vllm-v100` + `~/1catai-vllm` +
  vendored lmdeploy (Apache-2.0) licenses → what can be adopted, how to credit.

### F. Engine build + serving  — **scope DONE (planning) → `docs/FP8_ENGINE_STAGE_F_BUILD_SCOPE.md`**
- [x] **Build scope PINNED:** source = **lmdeploy v0.14.0** (Config_E4M3 verified at tag); method =
  **copied minimal subtree** (~1MB `third_party/`, not full-vendor/submodule); runtime coupling = ONE
  header (`kernels/attention/quantization.h`), prune sm75/80/90 + tma + test/, trim registry to sm70;
  CUTLASS headers-only for `mainloop_sm70.h`; target CUDA 12.6 (our image). Op names match the adapter.
- [ ] **IMPLEMENT (after push+review):** build spike (trimmed subtree + `fp8_gemm_sm70_out` binding →
  Stage-D gate in OUR image) → full ops → wire `select_backend` into `compressed_tensors_v100.py`.
- [ ] Serving: TP≤4 clean, TP8 MoE sharding decision, eager+cudagraph, real-prompt sanity + throughput.

### G. Decision gate → narrative + maintenance
- [ ] Only after A–F: adopt TurboMind FP8 as backend / keep ours for channel-scale + M=1 niche.
- [ ] Update repo narrative to "candidate backend" ONLY if code is actually integrated + validated.
- [ ] Maintenance home: if adapter stays thin+stable, consider offering to ai-bond (else keep local/plugin).

## 3. Progress / next steps
- [x] **Stage A DONE + SEALED (2026-07-03):** `_auto` root cause + minimal repro
  (`repro_fp8_auto_bug.py`) + bug report (`docs/FP8_SM70_AUTO_BUG_REPORT.md`, ready to relay to 1catai).
  Dense fixed/`_meta` cos=1.0000 M=1..32 on square + 4 non-square shapes. Grouped-MoE cos gate PASS
  18/18 cos=1.0000 (`moe_cos_gate.py`). Correctness cleared, dense + grouped MoE.
- [x] **Stage B DONE (2026-07-03):** lineage/adoptability matrix → `docs/FP8_ENGINE_STAGE_B_LINEAGE.md`.
  ai-bond = own BSD-3 Int4 kernel, drop-in on standard `awq_gemm`, minimal but AWQ-only (off-mission);
  1catai = vendored TurboMind s884 (Apache-2.0), only FP8 option, proven correct, adoption cost is structural.
- [x] **Stage C DONE (2026-07-03):** cross-image grouped-MoE A/B (harness `prepare_moe_inputs.py` +
  `bench_moe_ours.py` + `bench_moe_1catai.py` + `compare_moe.py`), both models, all cos=1.0000 →
  `docs/FP8_ENGINE_STAGE_C_FINDINGS.md`. **TurboMind faster e2e across the decode envelope + kernel
  faster at ALL M; ours only ~ties at Qwen M=1–2.** Kernel speed is NOT our differentiator.
- [ ] **NEXT — Stage D (integration/compat):** the real differentiators are non-speed — does TurboMind
  load OUR compressed-tensors FP8? **channel-scale gap** (GLM-4.5-Air real ckpt = channel W8A8, 1catai
  grouped kernel is block-128 only); TP/cudagraph/serving; fail-loudly on unsupported. Then E (licensing,
  mostly done in Stage B) → F decision.
- No repo narrative change until Stage F decision gate.
