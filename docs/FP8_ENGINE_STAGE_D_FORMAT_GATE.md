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

## Full-layer + w2 + TP — DONE 2026-07-03 (`stage_d_full_moe.py`)
Real Qwen3.5-35B-A3B-FP8 **layer 0, all 256 experts** loaded off disk, prepared, and run through
the **complete serving path** (`moe_permute → fp8_moe_gemm(w13) → silu_and_mul → fp8_moe_gemm(w2)
→ moe_unpermute`) with **real top_k=8 routing** vs an independent fp32 per-token reference:

| check | result |
|---|---|
| w2 / down_proj single-expert round-trip | **cos = 1.0000** |
| full MoE layer e2e (256 experts, top_k=8, 16 tokens, 104 uniq experts) | **cos = 1.0000** |

Gotcha found + fixed: `moe_unpermute` `topk_weights` must be **float32** — passing float16 is
byte-reinterpreted (cos≈0 garbage). Same discipline class as the `_auto` hazard: dtype/layout
contracts must be exact or fail loudly.

### TP scale-sharding constraint (block-128) — applies to ANY block-128 FP8 MoE, not TurboMind-specific
Intermediate `I` is sharded across TP ranks (w13 col-parallel on 2I, w2 row-parallel on I). Block-128
requires `I/tp` to be a multiple of 128:

| tp | I/tp (Qwen I=512) | block-128 |
|---|---|---|
| 1 | 512 | OK |
| 2 | 256 | OK |
| 4 | 128 | OK |
| 8 | **64** | **BREAKS** |

**Qwen block-FP8 MoE is cleanly intermediate-sharded only to TP≤4; TP8 gives a 64-wide shard that
violates block-128 scale granularity.** This is a property of block-128 + intermediate-TP, so it
constrains our coalesced path too — Stage F must confirm how the real stack shards MoE at TP8
(expert-parallel vs intermediate-sharded, or pad `I/tp`→128) for both backends.

## Status
Format + loader + full-layer correctness for **block-FP8** is CLEARED (weight/scale round-trip,
w13+w2, full real-ckpt layer with top_k combine). Remaining before adopting as serving code:
Stage E adapter (`VLLM_V100_FP8_BACKEND`) + engine-source diff; Stage F serving (TP≤4 clean, the
TP8 sharding question, eager+cudagraph, real-prompt sanity).
