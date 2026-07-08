# FP8 Engine — Stage H2: wire compressed-tensors BLOCK-128 MoE → TurboMind (SCOPE / DEFERRED)

Status: **DEFERRED — implement-ready, not coded.** Blocked on a test target (§1). H1 (dense CT-block →
TurboMind) shipped in `f359cf3` on branch `fp8-ct-block-turbomind-wiring`; H2 is the remaining CT-block
loose end. This note captures the exact plan so H2 can be implemented the moment a checkpoint appears.

## 1. Why deferred — no test target, and it's a delicate change

**No CT-block-MoE checkpoint exists on disk, and the format is rare in the wild.** Disk scan (2026-07-08):
the only compressed-tensors MoE FP8 model present is `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` — **channel**
strategy (→ ours). MoE FP8 checkpoints are almost always either native `quant_method=fp8` block
(Qwen3-MoE, DeepSeek — already handled by the native TurboMind MoE path) or CT channel/dynamic (RedHatAI →
ours channel path). **Compressed-tensors + block-128 + MoE is uncommon**; no known public checkpoint.

Consequence: unlike H1 dense, H2 **cannot get a real serving-agreement A/B** (the gate that made H1
trustworthy — 517/517 bit-identical). Max achievable rigor today = an offline synthetic numtest. Per the
correctness-first mission ("agreement is proof; coherent output ≠ proof"), we do **not** ship blind MoE
wiring for a format with no test target. Implement when a real CT-block-MoE checkpoint materializes.

**Also riskier than H1.** H1 was a clean drop-in; H2 has two extra hazards (§3): the native MoE prepare
reads attrs the CT MoE method lacks, and the CT MoE *apply* has shared-expert storage side-effect semantics
that must be reproduced exactly.

## 2. Interception points (both in `patch_compressed_tensors_moe_for_v100`, the CTMoE method)

- **Load** — `patched_process_weights_after_loading` (the CT MoE one; ends ~`compressed_tensors_v100.py:1184`,
  banner "CHANNEL/TENSOR only; FP16-resident"). Add a BLOCK-turbomind branch at the TOP, before the existing
  dequant-to-FP16 / mixed-w13 logic. On miss → fall through unchanged.
- **Apply** — `patched_apply` (`:1186`), signature `(self, layer, x, topk_weights, topk_ids,
  shared_experts_input)`. Add a `_v100_ct_tm_moe` branch FIRST, mirroring the mixed-path block (`:1196-1221`):
  return routed-only; let the MoERunner combine the shared expert (do NOT touch `shared_experts_input`).

Reuse targets (the proven native MoE engine path, already validated on real Qwen3.5-35B-A3B block-MoE):
`_tm_moe_prepare` (`vllm_serve.py:1285`) and `_our_moe_apply_turbomind` (`:1337`).

## 3. The two adaptations vs H1's clean drop-in

**(a) `_tm_moe_prepare` reads native attrs the CT method lacks.** It uses `self.weight_block_size` and
`self.weight_scale_name` (→ `w13_{name}`, `w2_{name}`). The native Fp8MoEMethod has these; CTMoE does not —
its scales are `layer.w13_weight_scale`, `layer.w2_weight_scale`. Fix: before calling, set on the CT method
`self.weight_block_size = (128, 128)` and `self.weight_scale_name = "weight_scale"` (CT names are
`w13_weight_scale`/`w2_weight_scale`, so `weight_scale_name="weight_scale"` resolves correctly). Verify the
CT block MoE scale layout is `[E, N/128, K/128]` per expert (same as native `w13_weight_scale_inv`); H1 proved
CT `weight_scale` ≡ native `weight_scale_inv` for dense — confirm it holds per-expert for MoE via the numtest.

**(b) Shared-expert apply semantics.** The CT MoE runner STORES the shared-expert output during the forward
and COMBINES it AFTER apply (`moe_runner.py:552`; see the mixed-path notes at `:1188-1206`). So the tm apply
must return **routed-only [M,H]** and NOT touch shared — exactly what `_our_moe_apply_turbomind` already does.
Wrap it to match the CT signature (ignore `shared_experts_input`). Double-storing or consuming shared here is
the classic hazard (tripped the `assert _output is None` guard historically).

## 4. Eligibility & gating

- Per-expert dims: w13 `[E, N13=2I/tp, K13=H]`, w2 `[E, N2=H, K2=I/tp]`. Call `select_backend(strategy="BLOCK",
  weight_block_size=(128,128), local_n, local_k, need_moe=True)` for BOTH w13 and w2; require both == turbomind.
  **w2 `K2 = I/tp` is the alignment risk** (native TP8 w2 K=I/tp=64 fails %128 → falls back). Mixed
  eligible/fallback across experts should NOT be allowed within one layer for the grouped kernel — if any
  expert/tensor is ineligible, fall the whole layer back to the existing FP16-dequant path.
- EP (`layer.expert_map is not None`) not supported by the native tm MoE path → exclude (fall back), and guard
  in apply (native path raises if expert_map present — we freed raw weights, so silent-wrong is the hazard).
- Kill switch: reuse `VLLM_V100_CT_BLOCK_TM` (or add `VLLM_V100_CT_BLOCK_TM_MOE` for independent control).
  Default: **OFF for MoE until real-checkpoint validated**, even after H2 lands (dense stays on).
- Distinct logs (buyer writeup), mirroring H1: `ct-block-moe->TurboMind`, `ct-block-moe->FP16 (selfcheck…)`,
  `ct-block-moe->FP16 (ineligible/no-engine/killswitch/EP)`.

## 5. Validation ladder (when a checkpoint exists)

1. **Synthetic numtest** — random block-MoE experts (E, w13 [2I,H], w2 [H,I], block-128 scales): tm grouped
   apply vs FP16-dequant reference on a routed probe, cos/L2 gate. (Mirror `moe_cos_gate.py` + `_ct_moe_selfcheck_w13`.)
   This is the ONLY gate achievable without a checkpoint; do it first as the pre-flight even now if desired.
2. **Per-layer loader self-check** — same numtest on the layer's REAL experts at load; divergence → FP16 fallback.
3. **Serving agreement** — ours(FP16-dequant) vs auto(turbomind), eager/greedy, on the real checkpoint
   (CT-aware harness like `scratchpad/ct_block_exactness_ab.sh`, adjusted TP so ours fits). Expect MoE benign
   ties (like Qwen 35B-A3B), not necessarily bit-identical.
4. **Memory** — turbomind FP8-resident MoE vs FP16-dequant footprint; confirm it fits lower TP.
5. **Perf** — decode C1/C8 + TTFT, cudagraph (native tm MoE cudagraphs; verify CT-block-MoE captures too).

## 6. Trigger to implement

A compressed-tensors block-128 MoE FP8 checkpoint appears (e.g. RedHatAI ships an FP8-block MoE), OR a buyer
specifically needs one. Until then: no code — this note is the spec. Estimated effort once unblocked: small
(mirror H1 + the two §3 adaptations), dominated by validation.

## 7. Relationship to the FP8 arc

With H1 (dense CT-block) shipped, the FP8 coverage picture:
- native fp8 block (Qwen3/DeepSeek): dense + MoE → TurboMind ✓ (Stage F).
- CT channel/dynamic (RedHatAI): dense + MoE → ours FP8-resident ✓.
- CT block-128 **dense** (gemma-4-31B-FP8-block): → TurboMind ✓ (Stage H1).
- CT block-128 **MoE**: → ours FP16-dequant today; TurboMind wiring = this note (H2), deferred (no target).

CT-block-MoE is the last FP8 loose end, and it has no checkpoint that needs it. FP8 is otherwise complete;
GPTQ/AWQ (real checkpoints, aftermarket majority) is the higher-value next arc.
