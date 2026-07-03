# Stage C — grouped-MoE perf methodology (design, pre-build)

Branch `fp8-engine-stage-c-perf`. Correctness is already cleared (Stage A: 1catai
grouped-MoE cos gate PASS 18/18). This stage measures **whether TurboMind's grouped
FP8 MoE is actually faster than our coalesced GEMV in our serving-shaped path**, not
just whether its core HMMA GEMM is faster (it is — dense primitive, Stage A card).

## The hypothesis this harness exists to test
At ≤5 users with top_k routing, **per-expert M is tiny and discrete** (often 1–4 rows).
TurboMind wins the *core GEMM* but pays a **permute + unpermute** tax our coalesced path
does not. So the serving-relevant question is whether that per-call overhead erases the
HMMA advantage at small M. Kernel-only numbers alone would be misleading — hence the
five-component split below.

## Two engines, different call graphs (why the 5 timings don't map 1:1)

| Component | **Ours (coalesced GEMV, production path)** | **1catai (TurboMind grouped)** |
|---|---|---|
| 1. prepare / repack | ~0 at runtime — weights resident as `[E,N,K]` FP8 (`view(uint8)`); one-time load only | `fp8_sm70_prepare` per expert + `awq_moe_build_strided_ptrs`; one-time load only |
| 2. route materialization | coalesced: **none** (kernel takes per-row `eids` directly). tiled/mtile: `argsort(eids)`+`bincount`+offsets | `moe_permute` → sorted input + `expert_offsets` + `inv_permuted_idx` (the sort/scatter) |
| 3. kernel-only | `fp8_w8a16_grouped_gemv_coalesced` (w13) + `silu_and_mul` + (w2) | `fp8_moe_gemm_sm70_out` (w13) + `silu_and_mul` + (w2) |
| 4. scatter / index-add | coalesced writes per-row in-place; top_k combine = weighted `index_add` to output | `moe_unpermute` (weighted combine back to token order) |
| 5. end-to-end MoE call | route + w13 + silu + w2 + combine | `Fp8SM70MoEMethod.apply` (permute → w13 → silu → w2 → unpermute) |

**Fair-comparison rules**
- prepare/repack (row 1) is a **one-time load cost** → reported separately, **excluded**
  from the per-decode-call comparison (both amortize it over the whole run).
- Kernel-only (row 3) is the apples-to-apples HMMA-vs-GEMV number. To keep row 3 honest,
  also run **our sorted `tiled` variant** (same sorted-grouped shape as 1catai) alongside
  our production `coalesced` — so we separate "kernel algorithm" from "we skip the sort."
- End-to-end (row 5) is the serving-relevant number: **each engine uses its own best full
  path** (ours = coalesced, no sort; 1catai = permute+grouped+unpermute).
- **Cos gate FIRST** on our side too (1catai already gated): every timed config must be
  cos=1.0000 vs the shared FP32 dequant ref before its time is reported.

## Sweep
- **Regimes:** `spread` (rows even across all E), `hot1` (all rows → 1 expert),
  `hot8`/`skew` (concentrated on ~8 experts). Matches router concentration extremes.
- **tpe (tokens-per-expert / effective decode M):** 1, 2, 4, 8 (min).
- **Dims (default):** GLM-4.5-Air-FP8 — our validated flagship MoE (E, top_k, H, I from
  the real config). Also a synthetic `[E=128,N=352,K=4096]` GLM-Air-like point to match the
  existing NCU probe. *(Override-able; say if you want Qwen3.5-35B-A3B dims instead.)*

## Cross-image harness (mirrors the dense A/B pattern already in `tools/turbomind_ab/`)
1. `prepare_moe_inputs.py` — freeze E experts (w13 `[2I,H]`, w2 `[H,I]`) FP8 block-128 +
   scales + per-regime/tpe routing + **byte-identical** FP32 reference. Runs once, any env.
2. `bench_moe_ours.py` — our image (`fp8_dequant_ext_vllm`): times components 2–5 for
   `coalesced` (+ `tiled` for row 3), cos-gated.
3. `bench_moe_1catai.py` — `1catai-vllm-v100:cu128-fp8sm70` (`--entrypoint /bin/bash`):
   times components 1–5 via the real `Fp8SM70MoEMethod` op sequence, cos-gated.
4. `compare_moe.py` — merge, enforce cos gate, emit the 5-timing table + a
   ≤5-user-envelope verdict (where, if anywhere, each engine wins).

Timing hygiene: `cudaProfilerStart/Stop`-free wall timing with warmup + `torch.cuda.synchronize`,
n_iter≥20, clean-GPU guard (shared box). Bulky raw logs summarized into a results CSV +
short findings doc; only compact proof logs committed.

## Deliverable
A `results_moe_*.csv` + a short findings section answering: **at tpe∈{1,2,4,8}, does
TurboMind's grouped MoE beat our coalesced GEMV end-to-end, or does permute/unpermute
overhead flip it at small M?** Then Stage D/F decide adopt / keep / hybrid.
