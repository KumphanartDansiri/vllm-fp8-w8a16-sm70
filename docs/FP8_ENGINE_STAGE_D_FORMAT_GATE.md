# Stage D — format gate + adapter contract (block-FP8 ↔ TurboMind)

Branch `fp8-engine-stage-d-compat`. Direction (user + Codex, 2026-07-03): make the
TurboMind/lmdeploy FP8 engine work **inside our original vLLM + our patch layer** behind the
smallest clean adapter; keep our kernels as the compatibility/fallback/control path; 1catai
stays a reference oracle, not the baseline; ai-bond handoff is a later (Stage G) question.

## Format gate — PROVEN on real weights (`stage_d_format_gate.py`)
Loaded a **real** Qwen3.5-35B-A3B-FP8 block-FP8 expert (layer 0, experts.0) straight off disk
and round-tripped it through the TurboMind engine:

| check | result |
|---|---|
| gate/up on disk | `[512,2048] float8_e4m3fn`, scale `[4,16] **bf16**` (weight_scale_inv) |
| fused w13 | `[1024,2048] fp8` + scale `[8,16]` = `[2I/128, H/128]` — orientation matches prepare |
| `fp8_sm70_prepare` | OK (meta k_ld=65536=2048·32, q_ld=1024) |
| `fp8_gemm_sm70_out` vs fp32 dequant | **cos = 1.0000** @ M=1/4/16 (max_abs ~5e-5) |
| channel-scale `[N,1]` → prepare | **REJECTED loudly**: "output scale block mismatch" |

**Conclusion:** our real block-FP8 compressed-tensors checkpoints are consumable by TurboMind
with only (a) a **bf16→fp32 scale cast** and (b) the **gate/up fusion our loader already does**
(`compressed_tensors_v100.py`: block dequant is `weight_fp8 * scale`, `[N/128,K/128]` — identical
semantics/orientation to `fp8_sm70_prepare`). Channel-scale fails fast → must fall back to ours.

## Backend eligibility (grounded, not hypothetical)
| checkpoint strategy | scale layout | backend | why |
|---|---|---|---|
| BLOCK, `[128,128]` | `[N/128,K/128]` | **turbomind** eligible | round-trip cos=1.0 proven above |
| CHANNEL (W8A8, e.g. GLM-4.5-Air) | `[N,1]` | **ours** (fallback) | prepare rejects; block≠channel; W8A8≠W8A16 |
| TENSOR / non-128 block | — | **ours** (fallback) | outside TurboMind block-128 path |

There is already a shipping checkpoint (GLM-4.5-Air-FP8, channel W8A8) that **only our path
serves** — so the fallback path is load-bearing, not decorative.

## Adapter contract (Stage E target)
`VLLM_V100_FP8_BACKEND = ours | turbomind | auto`  (auto = turbomind iff eligible else ours)
- **turbomind path:** cast scale→fp32 → fuse gate/up → `prepare(w, scale) -> packed_w, meta(k_ld,q_ld)`
  → grouped `fp8_moe_gemm_sm70_out(sorted_x, packed_w, meta, expert_offsets, ...)`. **Never `_auto`.**
- **ours path:** existing coalesced/tiled W8A16 GEMV (channel/tensor/non-128, and where per-expert
  M=1 single-user decode favors it).
- **Fail loudly / log:** on unsupported layout, and always log the selected backend + the reason
  (eligibility or fallback cause) so the choice is observable.

## Open (rest of Stage D, before serving)
- [ ] w2 (down_proj) block-FP8 round-trip (same as w13; expected pass — same layout).
- [ ] Full MoE layer via the real checkpoint (all E experts) end-to-end cos, not one expert.
- [ ] TP sharding: how block scales `[N/128,K/128]` split across TP ranks vs prepare's per-shard
  N,K; confirm packed meta is per-shard.
- [ ] Then Stage F serving (TP2/4 smoke → TP8 target, eager + cudagraph, real-prompt sanity).

## Caveat
This gate proves the **weight/scale format** round-trips. It does **not** yet prove a full
served layer (routing + all experts + TP + cudagraph) — those are the remaining Stage D/F items.
