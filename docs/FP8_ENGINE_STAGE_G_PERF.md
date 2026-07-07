# Stage G — FP8 serving performance: TurboMind vs our FP8 dequant (buyer-facing)

Correctness is locked (see `FP8_ENGINE_STAGE_F_LOADER_WIRING.md`: cos=1.0000 + serving agreement with
`ours`). This document is the **performance** answer only — *how much do I actually get* on real vLLM
serving, same checkpoint, same hardware (V100 / sm_70), TurboMind vs our FP8 dequant path.

## Headline
**TurboMind turns FP8 on V100 from a *compatible fallback* into a *real serving backend*** — better
prefill (TTFT), better batched decode, and (uniquely) cudagraph-capable MoE. It is **not** simply
"tensor cores are faster": at a single user's M=1 decode the two paths essentially tie. The wins appear
exactly where the effective GEMM height M rises — concurrency and prefill — and, for MoE, from being the
only FP8 path that can be captured into a CUDA graph at all.

## Method
- Serve the same checkpoint twice (`VLLM_V100_FP8_BACKEND=ours` vs `=auto`→TurboMind), one job at a time,
  isolated extension cache, baked-`.so` image (no runtime JIT). `MODE=cudagraph` (mode-0 FULL_DECODE_ONLY +
  TRITON_ATTN) is the meaningful serving path; `MODE=eager` is a diagnostic. Backend engagement asserted
  from logs every run. Concurrency C=1/2/4/8; report per-user tok/s, aggregate tok/s, TTFT, resident/peak MiB.
  Harness: `tools/turbomind_ab/fp8_turbomind_vs_ours_perf.sh` (+ `fp8_perf_matrix.sh`, `perf_bench_client.py`,
  `fp8_perf_render.py`).
- Models: Qwen3.5-27B-FP8 (dense), Qwen3.5-35B-A3B-FP8 (block-FP8 MoE). GENTOK 256 (cudagraph) / 128 (eager).

## Dense — Qwen3.5-27B-FP8, cudagraph (per-user | aggregate tok/s; x = tm/ours)

| TP | C1 per-user | C1 agg | C8 per-user | C8 agg | TTFT (ours→tm) |
|----|-------------|--------|-------------|--------|----------------|
| 2  | 33.4→36.0 (1.08×) | 22.6→27.8 | 12.3→31.2 (**2.54×**) | 77.8→231.6 (**2.98×**) | ~4–5s → ~0.4–0.7s |
| 4  | 50.9→54.0 (1.06×) | 34.4→37.7 | 20.8→46.1 (**2.22×**) | 155.4→328.5 (**2.11×**) | ~2–5s → ~0.4–0.7s |

Single-user: ~+6–8%. Under load `ours` per-user collapses (33→12) while TurboMind holds (36→31) →
**~2–3× aggregate at 8 users**. TTFT **~10×** better.

## MoE — Qwen3.5-35B-A3B-FP8

**(a) Same-mode kernel, eager (apples-to-apples, no cudagraph):** aggregate tok/s, x = tm/ours

| C | TP2 (ours→tm) | TP4 (ours→tm) |
|---|---------------|---------------|
| 1 | 5.8→6.4 (1.10×) | 5.2→6.4 (1.22×) |
| 4 | 20.5→25.4 (1.23×) | 18.2→25.6 (1.41×) |
| 8 | 15.9→48.9 (**3.08×**) | 13.7→50.9 (**3.71×**) |

At low load the kernels tie; at C8 `ours`' per-row GEMV collapses (per-user 6.7→2.1) while TurboMind's
grouped GEMM holds (~6.3) → **3–3.7×**. TTFT ~9× better even in eager.

**(b) The decisive factor — cudagraph capturability:**
- **TurboMind MoE captures a cudagraph** → **77–88 tok/s** single-user, up to **442 (TP2) / 499 (TP4) tok/s**
  aggregate at C8.
- **Our MoE path *cannot* be captured.** `_our_moe_apply_grouped` does data-dependent host work
  (`torch.nonzero` + `int(numel())` + `torch.unique().tolist()`) which is illegal during CUDA-graph capture
  (`cudaErrorStreamCaptureUnsupported`, reproduced isolated at TP2 **and** TP4). So `ours` MoE is stuck at
  eager ≈ **6 tok/s** — below the usability floor. This is a pre-existing property of the `ours` path (the
  TurboMind wiring is additive/gated), not a regression.

**⇒ For MoE, TurboMind is categorical: the only FP8 path that reaches usable serving speed (~11× over
`ours` eager-only).**

## Mechanism (why)
1. **M=1 decode ties.** Both paths are memory-bandwidth-bound reading the FP8 weights once; tensor cores
   don't help a single row. (Dense +6–8%, MoE +2% eager.)
2. **Concurrency + prefill win.** As requests batch (C≥2) or during prefill, the effective GEMM height M
   grows; TurboMind's grouped/tensor-core GEMM amortizes the weight load while `ours`' per-row GEMV does not.
   → 2–3.7× aggregate at C8, ~10× TTFT.
3. **MoE cudagraph win.** TurboMind's MoE uses fixed-size `moe_permute` + grouped GEMM (no host sync) →
   capturable. Ours' routing needs data-dependent host ops → not capturable → eager-only.

## Buyer-facing guidance ("what do I get?")
- **Dense, single user (hobbyist):** `ours` FP8 is already fine (33–54 tok/s); TurboMind is a little nicer
  (+~7%) and much snappier to first token.
- **Dense, multi-user serving:** TurboMind strongly preferred — ~2–3× the throughput at 8 concurrent users.
- **MoE (any use):** TurboMind is **required** for usable serving speed — it is the only FP8 path that
  cudagraphs; `ours` MoE is limited to ~6 tok/s eager.
- **Memory:** identical footprint (both FP8); the KV-cache preallocation (`gpu-mem-util 0.90`) dominates, so
  the weight-size difference isn't visible in these numbers.

## Caveats
- **TP8 MoE:** at TP8 the MoE `w2` shard is `K = I/tp = 64`, which breaks block-128 eligibility → falls back
  to `ours` → eager-only (slow). The current flagship 122B-A10B deployment at TP8 hits this; TurboMind MoE
  needs TP≤4 (I/tp≥128). Document as a deployment reality, not a TurboMind-MoE benchmark.
- Eager numbers are diagnostic (correctness/mechanism), not the headline; cudagraph is the real serving path.
- Single 256/128-token run per (config, C); ballpark, not a tuned throughput campaign.
