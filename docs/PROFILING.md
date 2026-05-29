# Profiling & diagnostics

This package ships extensive runtime instrumentation so you can see *where
decode time actually goes* on V100 — and use that to decide what's worth
optimizing. Everything here is **opt-in via environment variables and OFF by
default**, so it costs nothing in normal serving.

There are two unrelated families of knobs. Don't confuse them:

| Family | Purpose | cudagraph-safe? | Default |
|---|---|:---:|:---:|
| **Production MoE knobs** (`VLLM_V100_FP8_MOE_GROUPED_*`, `…_FAST_ROUTE_PREP`) | Make the kernels faster | ✅ yes | tuned ON (see README "Serve FP8") |
| **Diagnostic knobs** (`…_PROFILE`, `…_BREAKDOWN`, `…_DEBUG`) | Measure / inspect, don't change perf | ❌ **no — eager only** | OFF |

## ⚠️ Read this before turning anything on

1. **Profiling is eager-only.** The timing hooks use CUDA events and host-side
   reads of GPU timing state. Some capture-sensitive hooks are guarded and
   skipped under cudagraph, while lower-level profile paths can still be
   unsafe. Either way, cudagraph does **not** produce useful decode
   attribution. Always add `--enforce-eager` when profiling.
2. **Eager is ~6.8–8× slower than the cudagraph baseline.** Profiling numbers
   are for **relative attribution** — "the all-reduce is 90% of `out_proj`" —
   **not** for reporting throughput. Never quote a tok/s figure measured under
   `--enforce-eager` as a headline number.
3. Some knobs add their own overhead on top of eager (e.g. per-call CUDA
   syncs, `torch.unique`). Each is noted below.

## Decode timing breakdown

The headline diagnostic: a per-section attribution of each decode step
(attention / MoE router / experts / shared expert / dense MLP / GDN core),
flushed periodically to rank-0.

| Env var | Default | Effect |
|---|:---:|---|
| `VLLM_V100_FP8_DECODE_BREAKDOWN` | `0` | Master switch. `=1` enables the per-section decode breakdown. |
| `VLLM_V100_FP8_DECODE_BREAKDOWN_EVERY` | `32` | Flush a breakdown report every N decode steps. |
| `VLLM_V100_FP8_DECODE_BREAKDOWN_MOE_SUBS` | `1`* | Sub-hook the SparseMoeBlock (gate / experts / shared_expert). |
| `VLLM_V100_FP8_DECODE_BREAKDOWN_GDN_SUBS` | `1`* | Sub-hook the GatedDeltaNet (`in_proj_qkvz` / `in_proj_ba` / norm / `out_proj` + `gdn_core`). |
| `VLLM_V100_FP8_DECODE_BREAKDOWN_DENSEMLP_SUBS` | `1`* | Sub-hook the dense MLP (`gate_up` / `act` / `down`). |

\* only active when `DECODE_BREAKDOWN=1`.

```bash
PORT=8002 GPUS=all VLLM_V100_FP8_DECODE_BREAKDOWN=1 \
  ./docker/run_docker_vllm018_py312.sh serve-fp8 \
    --model /mnt/models/Qwen3.5-122B-A10B-FP8 \
    --enforce-eager \
    ... # rest of the Serve FP8 args
```

`--enforce-eager` is required for meaningful diagnostics; remove it again for
headline throughput measurements.

## All-reduce attribution (TP>1)

| Env var | Default | Effect |
|---|:---:|---|
| `VLLM_V100_FP8_ROW_PARALLEL_AR_PROFILE` | `0` | Times **only** the `tensor_model_parallel_all_reduce` call inside every `RowParallelLinear.forward`, aggregated as a cross-cutting `row_parallel_ar` bucket (not a sibling of the module sections — its time is already counted inside them). |

This is the knob that revealed the V100 tax: at TP=8 the post-GEMM all-reduce
dominates `out_proj` time, which is the direct motivation for MTP.

## MoE GEMM profiling

| Env var | Default | Effect |
|---|:---:|---|
| `VLLM_V100_FP8_MOE_PROFILE` | `0` | Per-shape-bucket timing of the grouped routed MoE GEMM, split prefill vs decode. |
| `VLLM_V100_FP8_MOE_PROFILE_EVERY` | `64` | Report cadence (MoE calls). |
| `VLLM_V100_FP8_MOE_PROFILE_WARMUP_CALLS` | `200` | Discard the first N MoE calls per rank before recording, so warmup/prefill don't pollute decode stats. |
| `VLLM_V100_FP8_MOE_PROFILE_ACTIVE_STAT` | `0` | Add `torch.unique(local_expert_ids)` active-expert stats. **~6% overhead** on the MoE call — debugging only. |
| `VLLM_V100_FP8_MOE_OTHER_PROFILE` | `0` | CUDA-event timing around the MoE combine + all-reduce ("moe_other"). |

## Structure & shape inspection (one-shot dumps)

| Env var | Default | Effect |
|---|:---:|---|
| `VLLM_V100_FP8_DEBUG_SHAPES` | `off` | Dump tensor shapes flowing through the FP8 linear path. |
| `VLLM_V100_FP8_DEBUG_APPLY` | `off` | Trace the FP8 `apply()` calls as patches fire. |
| `VLLM_V100_FP8_MOE_DEBUG` | `0` | Verbose MoE path logging. |
| `VLLM_V100_FP8_DEBUG_SHARED_EXPERTS` | `0` | One-shot rank-0 dump of the shared-expert runtime structure. |

## Numerical sanity guards (always on, tunable thresholds)

These are lightweight magnitude checks on the FP8 linear output — not timing
hooks — and stay on in normal serving:

| Env var | Default | Effect |
|---|:---:|---|
| `VLLM_V100_FP8_APPLY_WARN` | `1000` | Warn on the first call whose output abs-max exceeds this (early sign of a dequant/scale bug). |
| `VLLM_V100_FP8_APPLY_MAG` | `10000` | Treat output abs-max above this as "bad" (likely NaN/overflow). |

## Determinism / exactness

| Env var | Default | Effect |
|---|:---:|---|
| `VLLM_V100_FP8_HASH_LAYERS` | off | Hash per-layer activations for the exactness comparison used to validate MTP (see README "Validation matrix" and SESSION_LOG Stage 4). |

## How to use this to improve the kernels

1. Launch with `--enforce-eager` + `VLLM_V100_FP8_DECODE_BREAKDOWN=1`.
2. Send steady decode traffic; read the periodic per-section report.
3. Find the dominant bucket. If it's `row_parallel_ar`, the bottleneck is
   communication, not compute — spec-decode (MTP) or a better all-reduce is
   the lever, not a faster GEMM. If it's `moe_experts`, the grouped GEMM is
   the lever.
4. Re-measure the *real* speedup back under cudagraph (profiling off), because
   eager attribution and cudagraph wall-clock are different regimes.

The complete list of knobs (including transient ones not documented here)
lives in `src/fp8_w8a16_sm70/vllm_serve.py` — every `os.environ.get(...)` call
carries its own default.
