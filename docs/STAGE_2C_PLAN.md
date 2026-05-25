# Stage 2C plan

Status: **CLOSED 2026-05-25**, total production lift +5-7% vs v0.2.0 baseline
(4.866 → 5.093 tok/s warmed mean, fast=0 leg of the A/B). Win is
attributable to the layer caches (Step D), **not** the route-prep fast path
(Step C) which measured ~1.3% slower in production despite removing real
CUDA-stream work. See "Outcome" section below.

Revised 2026-05-25 after Step A profile data. Original ordering (fused
scatter as the headline lever) was based on a profile that mixed
warmup/prefill into decode stats; the segmented Step A run disproved it.

Starting point: v0.2.0 (commit `198e9e4`). Stage 2A active-list + Stage 2B
grouped routed GEMM shipped. Long-run decode on Qwen3.5-122B-A10B-FP8 TP=8
is `4.866 tok/s` (Stage 2A baseline `2.619 tok/s`, 1.86x).

## Why measurement was Step A (not a new kernel) — and what it told us

Original prior (smeared profile, late-call ~1600):

| Section | ms/call (old) |
|---|---:|
| `w13_gemm` | 0.981 |
| `scatter` (`index_add_`) | 0.904 |
| `w2_gemm` | 0.577 |
| `mask_sync` | 0.469 |
| `index_select` | 0.317 |
| `activation` | 0.269 |
| `routing` | 0.036 |
| **avg_wall** | **5.65** |

Stage 2C plan A originally had this read: `scatter ≈ 0.9 ms` was the
biggest named cost, and the implied next lever was fusing scatter into
the w2 epilogue. That read was wrong.

Stage 2C, Step A segmented decode from warmup/prefill via
`(phase, M, route_slots, grouped)` buckets and added route-materialization
timers. The decode-only bucket on Qwen3.5-122B-A10B-FP8 TP=8
(`phase=decode, M=1, route_slots=8, grouped=1`) settled at:

| Section | ms/call | share of avg_wall |
|---|---:|---:|
| `w13_gemm` | 0.136 | 7.5% |
| `nonzero` | 0.110 | 6.1% |
| `route_w_build` | 0.096 | 5.3% |
| `activation` | 0.091 | 5.1% |
| `w2_gemm` | 0.091 | 5.1% |
| `local_expert_ids` | 0.080 | 4.4% |
| `index_select` | 0.062 | 3.4% |
| `route_weight_apply` | 0.060 | 3.3% |
| `scatter` | 0.054 | 3.0% |
| `hidden_contig` | 0.031 | 1.7% |
| `cuda_sections` total | **0.81** | **45.0%** |
| `active_experts_stat` (instrumentation) | 0.112 | 6.2% |
| `out_zeros` | 0.039 | 2.2% |
| `routing` (`expert_map.to(device)`) | 0.037 | 2.1% |
| `view_uint8_contig` | 0.019 | 1.1% |
| `wall_sections` total | **0.21** | **11.6%** |
| **unattributed** | **0.78** | **43.4%** |
| **avg_wall** | **1.80** | **100%** |

Key findings:

1. **Scatter is not the headline lever.** 0.054 ms/call x 48 layers ≈
   2.6 ms/token. The old 0.9 ms was warmup/prefill contamination, not a
   decode-time signal. Fused scatter as the planned Stage 2C kernel is
   deprioritized.
2. **MoE is ~42% of decode wall**, not 75%. 48 x 1.80 ms = 86 ms/token
   of MoE at the v0.2.0 long-run 4.87 tok/s (205 ms/token total). The
   prior 75% figure was the Stage 2A baseline (2.62 tok/s), not Stage 2B.
   This caps any MoE-only optimization at ~8.4 tok/s before GDN /
   attention / framework dominate.
3. **The route-prep cluster (`nonzero` + `local_expert_ids` +
   `route_w_build`) is 0.286 ms/call = 16% of MoE call.** All three are
   launch-overhead-bound ops on `[1, 8]` tensors. Eliminating them via a
   reshape-based fast path is the new headline lever for Stage 2C.
4. **On this model `route_count == route_slots` (8 of 8 valid).**
   Qwen3.5-122B-A10B-FP8 on this serve replicates experts across TP
   shards rather than EP-partitioning them, so `expert_map is None` and
   no `valid_mask` filtering is needed. Fast path is gated on
   `expert_map is None` with the existing path preserved as fallback.
5. **`unattributed` is 0.78 ms/call (43%).** Profile mode synchronizes
   per call, so this is inflated by GPU wait time. Real Python overhead
   is smaller but non-trivial; finer wall timers around the C++ boundary
   of the grouped GEMM are added in Step A.5 to attribute more of it.
6. **`active_experts_stat` is instrumentation-only (`torch.unique`).**
   Gated behind `VLLM_V100_FP8_MOE_PROFILE_ACTIVE_STAT=1` (default 0).
   Reduces profile overhead 6%.

## Plan (revised after Step A data)

### A. Profile segmentation + route-materialization timers (DONE)

- Phase tag: derive in `_our_moe_apply` without depending on `max_num_seqs`.
  - `decode` if `M <= VLLM_V100_FP8_MOE_DECODE_M_MAX` (default 8) AND
    `route_slots <= _MOE_GROUPED_MAX_ROUTE_SLOTS`.
  - `prefill` otherwise.
  - The phase label is a best-effort categorization; the stats key
    below also carries M and route_slots, so even an imperfect label
    keeps the shape bucket honest.
- Warmup-skip: discard the first
  `VLLM_V100_FP8_MOE_PROFILE_WARMUP_CALLS` calls per-rank (default 200).
  Each rank tracks its own counter; only rank 0 / rank -1 prints, as
  current code does. The log line must include `warmup_skip=N` so future
  profile reads are not mysterious about which calls were counted.
- Stats key: replace the current `(layer-prefix)` key with
  `(phase, M, route_slots, grouped)`. Per-layer top offenders still
  reported separately under the existing log line.
- New per-call CUDA-event timers added inside both `_our_moe_apply` and
  `_our_moe_apply_grouped`. Priority order (top targets the unknown
  residual first):
  1. `out_zeros`: the `torch.zeros((..., hidden_size), ...)` alloc.
  2. `nonzero` (grouped only): `torch.nonzero(valid_mask, as_tuple=True)`.
  3. `local_expert_ids`: `local_topk[token_idx, route_idx].to(int64).contiguous()`.
  4. `route_w_build`: `topk_weights[token_idx, route_idx].to(float16)`.
  5. `index_select` (already exists): keep as-is.
  6. `view_uint8_contig`: `layer.w13_weight.view(torch.uint8).contiguous()`
     and `layer.w2_weight.view(torch.uint8).contiguous()`.
  7. `route_weight_apply`: `expert_out * route_w.unsqueeze(-1)` (only when
     `apply_router_weight_on_input` is False; that branch is the default
     today).
  8. `hidden_contig`: `hidden.contiguous()` after activation.
  9. `w13_gemm`, `activation`, `w2_gemm`, `scatter` already exist; keep.
- Rename in grouped mode: `mask_sync` is misleading; it is now the
  `torch.unique(local_expert_ids).numel()` call done only for the
  `active_experts` stat. Rename to `active_experts_stat` in grouped mode
  and skip the unique() call entirely when active_experts stat is
  disabled.
- Wall-vs-CUDA accounting. Python `perf_counter` timers around CUDA ops
  measure *launch* overhead, not GPU elapsed time, unless we sync. The
  log must distinguish:
  - `cuda_event_sections_ms`: sum of timed-with-cuda-events sections
    (w13_gemm, w2_gemm, scatter, activation, index_select, etc.).
  - `python_wall_sections_ms`: sum of perf_counter-timed sections that
    do not sync (routing, future Python-only sections).
  - `unattributed_wall_ms = call_wall_ms - cuda_event_sections_ms -
    python_wall_sections_ms`. This is the post-Step-A residual signal we
    optimize against. It will never be mathematically perfect.

### B. Profile-mode hygiene (Step B)

Standalone, decoupled from throughput. Three changes:

- Gate `active_experts_stat` behind
  `VLLM_V100_FP8_MOE_PROFILE_ACTIVE_STAT` (default 0). The
  `torch.unique(local_expert_ids).numel()` it currently computes is
  instrumentation-only; in production we never call it.
- Add finer wall timers inside `_our_moe_apply_grouped`:
  - `py_inner_loop` (`time.perf_counter()` around the entire grouped
    function body).
  - `py_dispatch_w13` (wall around the `_ext.fp8_w8a16_grouped_routed_gemm_a3`
    Python call for w13).
  - `py_dispatch_w2` (same for w2).
  These isolate C++/launch dispatch overhead from GPU kernel time
  (already CUDA-event-timed as `w13_gemm`/`w2_gemm`).
- Per GPT, **do not change the default per-call sync semantics.** The
  current behavior makes `wall_ms` close to per-call GPU wall, which is
  intuitive. A future opt-in deferred-sync mode can be added under
  `VLLM_V100_FP8_MOE_PROFILE_DEFER_SYNC=1` if we need clean Python-only
  wall measurements; not in this step.

### C. Route-prep fast path (Step C, headline lever)

Replace the `nonzero` + `local_expert_ids` + `route_w_build` cluster
(`0.286 ms/call = 14 ms/token`) with a reshape-based fast path when
`expert_map is None`. Existing path preserved as fallback.

Per GPT-locked design:

```
expert_map = getattr(layer, "expert_map", None)
if expert_map is None and grouped_path_enabled and route_slots <= max_route_slots:
    # Fast path: no EP remap/filtering, dense top-k assumed.
    route_slots = int(topk_ids.numel())
    local_expert_ids = topk_ids.reshape(-1).to(int64).contiguous()
    route_w = topk_weights.reshape(-1).to(float16)
    token_idx = _get_token_idx_cached(M, K, device)
    # No nonzero, no valid_mask GPU sync.
else:
    # Existing safe path: clamp -> expert_map -> valid_mask -> nonzero.
```

- `_get_token_idx_cached(M, K, device)`: returns a cached
  `[M*K]` int64 tensor. M=1 is all-zeros of length K; M>1 is
  `arange(M).repeat_interleave(K)`. Cached by `(M, K, str(device))` in
  a process-level dict. No per-call allocation.
- **Do not check `topk_ids >= 0`** on the fast path. Dense top-k is the
  invariant. If a future no-`expert_map` model emits negative IDs, add
  a debug assertion or env kill switch then; not now.
- Fast path is opt-in via `VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1`
  (default 1 once correctness is verified; ship default-on after the
  validation run).
- Expected save: 10-14 ms/token. Best case 205 ms/token -> ~193 ms =
  5.2 tok/s, plus the cache wins below.

### D. Free caches (Step D)

Ship alongside Step C; bracketed so they don't break the fallback path.

- `expert_map.to(device=topk_ids.device)` cached on the layer at first
  call as `layer._v100_expert_map_dev`. Saves 0.037 ms/call x 48 =
  1.8 ms/token. Only applies when `expert_map is not None`; current
  model has it None so this is dormant for Qwen3.5-122B-A10B-FP8 but
  needed for any future EP-partitioned model.
- `layer.w13_weight.view(torch.uint8).contiguous()` cached on layer as
  `layer._v100_w13_u8` (and `_v100_w2_u8`). Saves 0.019 ms/call x 48 =
  0.9 ms/token. Confirmed zero-copy by the existing log line.

Do not reuse the returned `out` tensor across calls. vLLM's MoE combine
path may hold the storage past return for TP all-reduce. Only
internal scratch is reusable.

### E. Validation (Step E)

Re-measure on V100, Qwen3.5-122B-A10B-FP8 TP=8, standard prompt:

- **Coherence gate.** Greedy `"The capital of France is"` returns
  coherent Paris text under both `FAST_ROUTE_PREP=0` and `=1`.
- **Profile-on decode bucket.** Confirm the route-prep cluster
  (`nonzero` + `local_expert_ids` + `route_w_build`) drops from
  0.286 ms to ~0 in the new `fast` sub-bucket. Add a `fast` flag in
  the bucket key to distinguish.
- **Decode-breakdown cross-check.** Set
  `VLLM_V100_FP8_DECODE_BREAKDOWN=1` simultaneously with MoE profile
  to confirm MoE drops by ~10-14 ms/token and GDN/attention shares are
  unchanged.
- **Profile-off long-run.** `max_tokens=200`. Pass criteria:
  - Coherent text.
  - Decode tok/s `>= 5.2` (target). Stretch 5.5 tok/s.
  - No prefill wall regression vs v0.2.0 baseline (guarded by
    `VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=32`, unchanged from
    Stage 2B).

### F. Deferred (not Stage 2C)

- **Fused scatter.** 2.6 ms/token saving. Too small to justify a new
  CUDA kernel on its own at this decode shape. Bundle with the stretch
  fused-MoE kernel (below) if we ever ship it.
- **Stretch: one-shot fused grouped MoE kernel.** Subsumes route-prep +
  w13 + activation + w2 + route_w + scatter into one launch per layer.
  Only attempt if Step C+D do not get to 5.2 tok/s. Large design space;
  not in scope unless the data demands it.
- **Grouped kernel partial-CTA `__syncthreads()` hardening.** Inherited
  pattern from A.3, safe today because `N` is `BLOCK_N_A3`-aligned for
  Qwen3.5 shapes. Replace with lane-out-guarded variant before broadening
  model support. Not blocking Stage 2C completion.

## Non-goals for Stage 2C

- FlashAttention-V100. Self-attention is 7.6% of decode; cu128 -> cu129
  toolchain bump is not justified yet.
- Grouped-WMMA. Route counts at decode (`<= 32` slots) do not fill WMMA
  fragments; current A.3 CUDA-core path is the right shape.
- CUDA graphs. V100 runs `--enforce-eager`.
- DeepSeek V4 / FP4 experts. Separate project.
- FLA Mamba kernel work for GDN (17% of decode). Secondary lever; revisit
  only after Stage 2C lands.

## Environment variables added in Stage 2C

| Var | Default | Purpose |
|---|---|---|
| `VLLM_V100_FP8_MOE_DECODE_M_MAX` | 8 | Upper M for the `decode` phase bucket. Combined with `route_slots <= MOE_GROUPED_MAX_ROUTE_SLOTS`. |
| `VLLM_V100_FP8_MOE_PROFILE_WARMUP_CALLS` | 200 | MoE calls per rank discarded before stats recording begins. |
| `VLLM_V100_FP8_MOE_PROFILE_ACTIVE_STAT` | 0 | Enables the `torch.unique(local_expert_ids)` instrumentation. Off by default; ~6% of MoE call when on. |
| `VLLM_V100_FP8_MOE_FAST_ROUTE_PREP` | 0 | Enables the Step C reshape-based fast path. Set 1 to opt in. Default-off after A/B showed -1.3% production wall on Qwen3.5-122B-A10B-FP8. Code retained for experimentation on shapes where the CPU/GPU pipeline does not absorb the saved CUDA work. |

## Open questions deferred past Stage 2C

- Per-tuple stats granularity: if many shape combinations show up under
  load, fall back to `(phase, grouped)` and per-layer offenders only.
  Watch on the validation run.
- PWAL-time vs lazy first-call caching of `expert_map` and uint8 weight
  views. Lazy is the v1; PWAL-time is cleaner but requires more plumbing.
- Deferred-sync profile mode under
  `VLLM_V100_FP8_MOE_PROFILE_DEFER_SYNC=1`. Added only if the unattributed
  bucket remains a question after Step B/C/D land.

## Outcome (2026-05-25)

Warmed profile-off A/B on Qwen3.5-122B-A10B-FP8 TP=8, standard 5-token
prompt, `max_tokens=200`, 4 timed curls per leg:

| Config | tok/s (warmed mean of 4 curls) |
|---|---:|
| v0.2.0 baseline | 4.866 |
| Step C + Step D, `fast_route_prep=1` | 5.027 |
| Step C + Step D, `fast_route_prep=0` | **5.093** (default) |

**Default flipped to `fast_route_prep=0`** because the fast path measured
~1.3% slower than the conventional `nonzero`-based path in production.

Attribution: the +5-7% over v0.2.0 comes entirely from the **Step D
caches** (`_get_layer_uint8_weights`, `_get_layer_expert_map_dev`), which
run on *both* the fast and the conventional path. The Step C fast path's
CUDA-event savings (~0.6 ms of GPU-stream work per MoE call) did not
translate to wall savings because the CPU/GPU pipeline was already
overlapping that work with subsequent kernels; the per-call sync at the
end of each MoE call charged the wait to wall regardless of when the
removed kernels would have run.

**What stays in tree:**

- All Step A profile machinery (warmup-skip, phase tagging, finer wall
  timers, gated `active_experts_stat`).
- All Step D layer caches.
- The Step C fast path code, gated behind
  `VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1`. Defaults off. Retained because
  the code is correct, simpler, and may be useful on future models with
  non-dense topk or different decode shapes where the pipeline can't
  absorb the removed work.

**Stage 2D direction (per GPT after Stage 2C closeout):**

The next-biggest lever is the gap between the full
`Qwen3NextSparseMoeBlock` (145 ms/token in profile-on) and the
`_our_moe_apply` portion we already instrument (48 × 1.74 = ~84 ms). That
~60 ms/token of wrapper overhead (topk gating, combine, shared experts)
is larger than any remaining target inside the FP8 fallback. Stage 2D
should start by instrumenting that wrapper. GDN/FLA Mamba (60+ ms/token,
17-25% of decode) is the second lever.
