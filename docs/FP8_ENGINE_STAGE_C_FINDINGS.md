# Stage C — grouped-MoE perf A/B: findings (Qwen3.5-35B-A3B + GLM-4.5-Air dims)

Branch `fp8-engine-stage-c-perf`. Cross-image A/B: our coalesced FP8 W8A16 grouped GEMV
(`vllm-v100:vllm021-cu126`) vs 1catai TurboMind s884 grouped FP8 MoE
(`1catai-vllm-v100:cu128-fp8sm70`). Byte-identical synthetic block-128 weights at each
model's real MoE dims; every timed config **cos=1.0000** vs a shared FP32 reference (gated
first). Times in ms; **ratio > 1 ⇒ ours faster**. Our coalesced path has route = unperm = 0.

## Headline
**TurboMind's grouped FP8 MoE is faster than our coalesced GEMV across the realistic
decode envelope, for both models.** Our only edge is a razor-thin tie (1.02–1.06×) at the
*extreme* smallest Qwen decode point (per-expert M = 1–2), and it evaporates with bigger
experts (GLM: ours never wins) or as soon as M ≥ 4. On the kernel itself, TurboMind wins at
**every** M (2.2–6.3×). This is the grouped-MoE confirmation of the dense Stage-A result —
and it endorses the Stage-B call: **validate/integrate TurboMind, don't compete on kernel speed.**

## Qwen3.5-35B-A3B (E=256, top_k=8, H=2048, I=512 — small experts)
| regime | tpe | R | 1cat e2e | our e2e | **e2e×** | 1cat kern | our tiled | kern× |
|---|---|---|---|---|---|---|---|---|
| hot1 | 1 | 1 | 0.153 | 0.145 | **1.06** | 0.060 | 0.241 | 0.25 |
| hot1 | 2 | 2 | 0.152 | 0.147 | **1.03** | 0.060 | 0.244 | 0.25 |
| hot1 | 8 | 8 | 0.152 | 0.149 | **1.02** | 0.060 | 0.245 | 0.24 |
| hot8 | 1 | 8 | 0.152 | 0.149 | **1.02** | 0.065 | 0.257 | 0.25 |
| hot8 | 4 | 32 | 0.152 | 0.240 | 0.63 | 0.066 | 0.258 | 0.25 |
| hot8 | 8 | 64 | 0.152 | 0.445 | 0.34 | 0.066 | 0.255 | 0.26 |
| spread | 1 | 256 | 1.152 | 1.704 | 0.68 | 1.058 | 2.732 | 0.39 |
| spread | 8 | 2048 | 1.341 | 12.128 | 0.11 | 1.263 | 2.804 | 0.45 |

ours faster e2e in **6/12** (all razor-thin, M=1–2). prepare/repack one-time = **125 ms**.

## GLM-4.5-Air (E=128, top_k=8, H=4096, I=1408 — big experts)
| regime | tpe | R | 1cat e2e | our e2e | **e2e×** | 1cat kern | our tiled | kern× |
|---|---|---|---|---|---|---|---|---|
| hot1 | 1 | 1 | 0.152 | 0.169 | 0.90 | 0.084 | 0.513 | 0.16 |
| hot1 | 8 | 8 | 0.152 | 0.302 | 0.50 | 0.086 | 0.519 | 0.16 |
| hot8 | 1 | 8 | 0.230 | 0.304 | 0.75 | 0.211 | 0.671 | 0.31 |
| hot8 | 8 | 64 | 0.247 | 1.902 | 0.13 | 0.215 | 0.611 | 0.35 |
| spread | 1 | 128 | 2.892 | 3.825 | 0.76 | 2.752 | 8.328 | 0.33 |
| spread | 8 | 1024 | 3.264 | 30.896 | 0.11 | 3.185 | 8.699 | 0.37 |

ours faster e2e in **0/12**. prepare/repack one-time = **146 ms**.

## Why (mechanism, not just numbers)
- **Our coalesced kernel is a per-row GEMV (M=1 semantics): no weight reuse across an
  expert's rows.** It re-streams each expert's weight per routed row, so cost scales O(R).
  TurboMind's grouped **GEMM** amortizes each expert's weight load over its M rows — textbook
  GEMV-vs-GEMM. That's why ours is only competitive at M=1 and collapses as M/R grow.
- **The permute/unpermute "tax" is real but small in absolute terms** (~0.075 ms route +
  ~0.016 ms unperm, flat). It's the *only* thing our path saves, and it only matters when the
  kernels are otherwise near-tied — i.e. the tiniest Qwen point. GLM's larger kernels dwarf it.
- **spread (all experts active)** is the worst case for our GEMV (it must touch every expert)
  and TurboMind wins 6–9× e2e.

## Honest caveats / what this does NOT say
- **Not a kernel we should keep for throughput.** But our path has two non-speed advantages
  that Stage D must weigh: (1) **no prepare/permute** (simpler serving loop), and (2) it
  **handles channel-scale** weights. The real **GLM-4.5-Air-FP8 checkpoint is channel-scale
  W8A8 — 1catai's block-128 grouped kernel cannot load it as-is** (this bench used synthetic
  block-128 at GLM dims for a shape-fair *kernel* comparison; the format gap is a Stage-D item).
- Routing model = per-slot, `top_k=1` permute (per-expert M = tpe is the timing driver);
  combine-reduction arity is second-order and symmetric across engines (not modeled).
- Synthetic weights (values don't affect timing; cos gate uses self-consistent FP8).
- Single V100, clean box, warmup + 30-iter wall timing.

## Verdict → Stage D/F
Kernel speed is **not** our differentiator — TurboMind wins the grouped MoE GEMM at all M and
e2e across the decode envelope. Adopt/validate TurboMind as the FP8 MoE backend (Stage D:
compressed-tensors load, **channel-scale support gap**, TP/cudagraph/serving, fail-loudly on
unsupported). Keep our coalesced path only where its niche holds — **single-user M=1 decode
with few active experts and/or channel-scale checkpoints TurboMind can't consume.**
Enforce the Stage-A serving contract (never `_auto`; prepare→meta or fail loudly).
