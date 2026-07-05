# Stage F — Loader-wiring design note (TurboMind SM70 FP8 engine → the loader)

Status: **IMPLEMENTED (native Fp8 path), awaiting the loader smoke.** Branch
`fp8-engine-stage-f-loader-wiring` off `fp8-engine-stage-f-build` (@33885e7). Codex confirmed both
findings + all open decisions (2026-07-05); wiring targets the native Fp8 path as recommended.

## IMPLEMENTATION STATUS (2026-07-05)
Code written (uncommitted; both files `py_compile` clean, adapter `_selftest` PASS):
- **Adapter** `src/fp8_w8a16_sm70/turbomind_fp8_backend.py` — FINDING 2 fixed: ops resolve under
  `torch.ops.turbomind_fp8_sm70` via a new memoized `ensure_engine()` (presence-check by default;
  JIT-build only under `VLLM_V100_FP8_ENGINE_JIT=1`, never silently under a service). `ops_available`
  + `prepare/gemm_out/moe_gemm_out/moe_build_strided_ptrs` route through it.
- **Loader** `src/fp8_w8a16_sm70/vllm_serve.py` — native Fp8 path wired:
  dense `_our_process_weights_after_loading`→`_tm_dense_prepare`, `_our_apply`→`_tm_dense_apply`;
  MoE `_our_moe_process_weights_after_loading`→`_tm_moe_prepare`,
  `_our_moe_apply`→`_our_moe_apply_turbomind` (moe_permute→w13→SwiGLU→w2→moe_unpermute, topk_weights
  fp32). `select_backend()` gates each weight (quiet); MoE all-or-nothing (w13 AND w2), EP excluded.
  Engine resolved ONCE at startup in `_patch_vllm_for_v100`. `_TM_FREE_RAW` flag (default OFF =
  keep raw ≈FP16 footprint until smoke+serving validate; flip to 1 for the memory win).
- **Smoke** `third_party/turbomind_gemm_sm70/loader_smoke.py` — drives the WIRED helpers on real
  Qwen3.5-35B-A3B-FP8 weights (dense M=1/4/16 + full grouped MoE with the unpermute combine
  build_and_gate_moe.py skipped).

### VALIDATION RESULTS (2026-07-05)
- **Loader smoke: PASS** (`vllm-v100:vllm021-cu126`, JIT engine). Real Qwen3.5-35B-A3B-FP8 layer 0:
  dense M=1/4/16 **cos=1.0000**; full grouped MoE permute→w13→SwiGLU→w2→**unpermute cos=1.0000**
  (the combine the lower-level gate skipped). `ops_available=True` confirms the namespace fix.
- **TP=2 eager serve: PASS** (`tools/turbomind_ab/tp_serve_validate.sh`, path A = prewarm→persisted
  `torch_extensions` cache→JIT). Dense Qwen3.5-27B-FP8 + MoE Qwen3.5-35B-A3B-FP8: `TurboMind
  DENSE/MoE engaged` on **both** TP workers, coherent output (rep≈0.09), 0 fallback lines.
- **`_TM_FREE_RAW` now defaults ON** — validated (smoke packed cos=1.0 + both TP serves) and REQUIRED:
  with it off, a turbomind MoE layer keeps raw+packed (~2× experts) and OOMs at TP2. apply() only reads
  the packed weight, so freeing raw is safe.
- **NEXT:** TP=4, then bake the engine into the image (`docker/Dockerfile.prod`) + rerun without JIT,
  then cudagraph (non-eager) + throughput. TP8 stays on ours (I/tp=64 breaks block-128) — deferred.

Safety: with the engine absent (baseline image, default `auto`), every FP8 weight resolves to
`ours` → **zero behaviour change**. Verified: adapter returns `ours` when ops absent.

---

(Original design note follows — kept as the rationale record.)

Branch `fp8-engine-stage-f-loader-wiring` off `fp8-engine-stage-f-build` (@33885e7). Written before
code so the seams + the two findings below (⚠ FINDING 1 = wrong wiring target in the plan;
⚠ FINDING 2 = adapter namespace bug) get reviewed first.

Goal (narrow): make `VLLM_V100_FP8_BACKEND=auto` route eligible **block-128 FP8** weights to the
vendored engine (`torch.ops.turbomind_fp8_sm70.*`, proven dense cos=1.0000 @4e90a83, grouped MoE
cos≥0.99 @33885e7), while channel/tensor and TP-broken shards stay on **ours**. No behaviour
change unless the engine ops are present AND the checkpoint is block-128.

---

## ⚠ FINDING 1 — the plan's wiring target is the wrong loader

The Codex RESUME card says "wire `select_backend()` into `compressed_tensors_v100.py`." Evidence
says that is the wrong file for every block-FP8 checkpoint we actually serve.

**Disk scan (all 5 block-FP8 [128,128] checkpoints on `/mnt/models`):**

| kind | arch | checkpoint | quant_method |
|------|------|-----------|--------------|
| MoE | Qwen3_5MoeForConditionalGeneration | Qwen3.5-122B-A10B-FP8 | **fp8 (native)** |
| MoE | Qwen3_5MoeForConditionalGeneration | Qwen3.5-35B-A3B-FP8 | **fp8 (native)** |
| MoE | Qwen3_5MoeForConditionalGeneration | Qwen3.6-35B-A3B-FP8 | **fp8 (native)** |
| dense | Qwen3_5ForConditionalGeneration | Qwen3.5-27B-FP8 | **fp8 (native)** |
| dense | Qwen3_5ForConditionalGeneration | Qwen3.6-27B-FP8 | **fp8 (native)** |

Every one is `quant_method: fp8` → vLLM's **native** `Fp8Config`/`Fp8LinearMethod`/`Fp8MoEMethod`,
handled by the patch family in **`vllm_serve.py`**, NOT compressed-tensors. Conversely, the
compressed-tensors path (`compressed_tensors_v100.py`) only ever sees CHANNEL/TENSOR checkpoints
today (GLM-4.5-Air channel W8A8, gemma CT-MoE) — its block-FP8 branches are **dead consumers**:
dense block → FP16 fallback (`:415-434`), MoE block → `raise NotImplementedError` (`:959-962`),
and no checkpoint on disk exercises either.

So the vendored engine (block-128 only) belongs in the **native Fp8 path**, where block-FP8 is the
live path and Stage D's own gate loaded "Qwen3.5-35B-A3B-FP8 block-FP8 expert off disk." Recommend
retargeting to `vllm_serve.py`; keep the CT hooks as a *documented secondary* (wire the identical
`select_backend()` there only if/when a block-FP8 compressed-tensors checkpoint appears).

**This reverses the plan's stated file — confirm before coding.** Everything below is written for
the native Fp8 path (with CT noted where it mirrors).

---

## 0. The seams (native Fp8 path in `vllm_serve.py`)

Installed at serve startup in the Fp8 patch block (`:2384-2509`): min_cap→70, and swaps on
`Fp8LinearMethod` + `Fp8MoEMethod` (+ `Fp8OnlineMoEMethod`).

| Path | process-weights (load) | apply (forward) | today's block-FP8 behaviour |
|------|------------------------|-----------------|------------------------------|
| Dense Linear | `_our_process_weights_after_loading` (`:1282`) | `_our_apply` (`:1207`) | keep FP8[N,K]+FP16 block scale → `_v100_fp8_gemm` |
| MoE experts | `_our_moe_init` (`:1341`) / `_our_moe_process_weights_after_loading` (`:1366`) | `_our_moe_apply` (`:1970`) → `_our_moe_apply_grouped` (`:1542`) | keep FP8[E,2I,H]/[E,H,I]+FP16 block scale → grouped coalesced GEMV |

These already do exactly what turbomind needs as scaffolding: keep the FP8 weight, cast block
scale to FP16, store both on the layer, and dispatch to our kernel in apply. The turbomind branch
is a **strict add** at each of the two decision points (dense PWAL, MoE PWAL) + a branch in each
apply. Decision is made **once per weight at load** (dims are post-TP-shard, per-rank), recorded on
the layer, read in apply. No per-forward decision, no per-forward prepare.

CT path mirror (secondary, only if a block CT ckpt appears): dense
`compressed_tensors_v100.py:353/442`, MoE `:955/1069`.

---

## Q(a) Where `select_backend()` is called

**Dense** — in `_our_process_weights_after_loading` (`:1282`), right after `self.block_quant` is
confirmed (`:1314`) and `weight`/`weight_scale_inv` are normalized (`:1321`). `N,K =
weight.shape`. Call `select_backend(strategy="BLOCK", weight_block_size=self.weight_block_size,
local_n=N, local_k=K, need_moe=False)`. Only on `"turbomind"` take the new prepare branch; else the
existing store-FP8+FP16-scale path is unchanged.

**MoE** — in `_our_moe_process_weights_after_loading` (`:1366`), after the block-quant guard
(`:1377`). Local MoE dims:
- w13: `K = H` (hidden 2048, 128-aligned), `N = 2·I/tp`
- w2 : `K = I/tp` — **the TP trip**: Qwen I=512 → TP4 gives K=128 (eligible), TP8 gives K=64
  (`local_k%128!=0` → predicate returns ours), `N = H`

Call for **both** w13 and w2; the layer is turbomind-MoE only if *both* eligible (else the whole
fused-MoE layer stays on the existing grouped-GEMV/FP16 path — no mixed engines within one layer).

---

## Q(b) When `prepare` runs + where meta(k_ld,q_ld) lives

**prepare is load-time only** (in process-weights), once per weight. Never in apply.

**Dense** (new turbomind branch of `_our_process_weights_after_loading`):
```
tm_w, tm_s, meta = backend.prepare(weight_fp8[N,K], weight_scale_inv.float()[N/128,K/128], 128)
layer._v100_tm = {"w": tm_w, "s": tm_s, "k_ld": int(meta[0]), "q_ld": int(meta[1]), "N": N, "K": K}
layer._v100_fp8_backend = "turbomind"
replace_parameter(layer, "weight", <tiny placeholder>)   # raw FP8 no longer needed post-pack
```
Apply — `_our_apply` (`:1207`) gets a branch above the `_v100_fp8_gemm` dispatch:
```
if getattr(layer, "_v100_fp8_backend", None) == "turbomind":
    out = empty[M, N]; backend.gemm_out(out, x2d, tm.w, tm.s, tm.k_ld, tm.q_ld); reshape; +bias; return
```

**MoE** (new turbomind branch of `_our_moe_process_weights_after_loading`) — sequence already proven
in `third_party/turbomind_gemm_sm70/build_and_gate_moe.py`; reuse it:
```
per expert e: r13 = prepare(w13[e], w13_scale[e], 128); r2 = prepare(w2[e], w2_scale[e], 128)
ptrs13 = awq_moe_build_strided_ptrs(stack(tm_w13), stack(tm_s13), k13, q13, E)
ptrs2  = awq_moe_build_strided_ptrs(stack(tm_w2),  stack(tm_s2),  k2,  q2,  E)
layer._v100_tm_moe = {"w13": {ptrs13, K13, N13}, "w2": {ptrs2, K2, N2}, "E": E}
layer._v100_fp8_backend = "turbomind"
```
Apply — `_our_moe_apply` (`:1970`) branches to a turbomind grouped path parallel to
`_our_moe_apply_grouped`: `moe_permute` (stock 0.21 11-arg — the verified call) →
`fp8_moe_gemm_sm70_out(gate_up, …ptrs13…)` → silu → `fp8_moe_gemm_sm70_out(down, …ptrs2…)` →
`moe_unpermute` (**topk_weights MUST be fp32** — Stage D gotcha). Persistent buffers, no per-call alloc.

Meta lives on the layer object (survives forwards); the *only* place k_ld/q_ld are kept — never
recomputed, never passed to `_auto`.

---

## Q(c) Avoiding repeated prepare / JIT during serving

- **prepare**: load-time only (above). Apply reads packed weights + meta off the layer. Zero
  prepare/pack in the hot path.
- **JIT/build**: ops live in namespace `torch.ops.turbomind_fp8_sm70` (TORCH_LIBRARY,
  `binding/fp8_sm70_bindings.cpp:37`). Register **once at process init** via a single
  `ensure_engine()` memo, placed next to the Fp8 patch install (`vllm_serve.py:2384`): baked image
  → ops already registered, return the handle; dev → call `_ext_build.build_ops()` exactly once
  (module-level guard). Every layer then resolves the same global ops. Prepare/apply never build.

### ⚠ FINDING 2 — adapter uses the wrong op namespace (hard blocker, fix in this change)
`turbomind_fp8_backend.py` probes `torch.ops._C.fp8_sm70_prepare` (`:56-57`) and its wrappers call
`vllm._custom_ops.fp8_sm70_prepare` (`:143-160`). The real engine registers under
`torch.ops.turbomind_fp8_sm70` (per the binding). **Neither probe nor call resolves.** Fix:
1. `ops_available()` → check `torch.ops.turbomind_fp8_sm70.{fp8_sm70_prepare, fp8_gemm_sm70_out
   [, fp8_moe_gemm_sm70_out, awq_moe_build_strided_ptrs]}`.
2. `prepare/gemm_out/moe_gemm_out` → route through the `ensure_engine()` handle
   (`torch.ops.turbomind_fp8_sm70`), not `vllm._custom_ops`.
3. `ensure_engine()` owns baked-vs-JIT resolution (Q(e)) so the loader stays agnostic.

---

## Q(d) auto / ours / turbomind behaviour (per weight kind)

Encoded in `select_backend()`; restated for the loader:
- **auto** (default): turbomind iff eligible (block-128, local dims 128-aligned, ops present), else
  ours + one `info_once` reason. Only mode that ever silently uses ours.
- **ours**: always ours — control/compat/fallback (channel W8A8, TP-broken shards, any model the
  engine can't consume). Unchanged from today.
- **turbomind**: turbomind if eligible, else **HARD-RAISE** (`turbomind_fp8_backend.py:122-127`) —
  an explicit request that can't serve this weight is a config error, not a silent downgrade.
  Per-weight: a TP8 w2 (K=64) under `mode=turbomind` raises at load, loudly.

Log the choice **once per (layer-kind, backend, reason)** via `info_once`, not per layer (hundreds).

---

## Q(e) Deployment — baked image vs runtime JIT

**Production = engine baked into the image** (Codex + agreed). `docker/Dockerfile.prod` gains a
build step that compiles the `third_party/` ext to a `.so` (same source list as `_ext_build.py`),
so `import` pre-registers `torch.ops.turbomind_fp8_sm70` with **no runtime compile** — no
first-request stall, no torch-extensions cache perms surprise under systemd, no "works in my shell,
not in the service." `ensure_engine()` in prod is just a presence check.

**Dev = runtime JIT**, behind a flag (`VLLM_V100_FP8_ENGINE_JIT=1`): `ensure_engine()` calls
`_ext_build.build_ops()` once at startup. Never the systemd path.

`ops_available()` returns True in both once registered; the loader gates purely on it, agnostic to
how the ops arrived.

---

## Validation order (each gates the next)

1. **Adapter unit** — namespace fix (FINDING 2) + `_selftest()` PASS; `ops_available()` True/False correct.
2. **Loader smoke** (in-process, cu126 image, no server): one block-FP8 dense Linear + one block-FP8
   MoE layer through the patched native-Fp8 PWAL+apply, cos vs FP16 ref (reuse Stage-D gates).
   Confirms prepare-at-load + meta-on-layer + apply wire correctly, engine baked/JIT-loaded once.
3. **TP≤4 real serve** — Qwen3.5-27B-FP8 (dense) + Qwen3.5-35B-A3B-FP8 (MoE), `=auto`, eager.
   At TP4 w2 K=I/tp=128 aligned → MoE goes turbomind; dense qkvo where 128-aligned goes turbomind.
   ⚠ these are `…ForConditionalGeneration` (VL + linear/Mamba-attn hybrid): the project already
   serves them on V100 today via our path, so the architecture runs — but confirm the *serve command*
   we already use (which excludes the visual tower / handles linear_attn) is the smoke harness, so a
   turbomind regression is isolated from architecture-support noise.
4. **TP8 sharding question** — I/tp=64 → predicate → ours fallback (graceful: existing grouped-GEMV,
   not a crash). Decide *separately* whether to solve real TP8 turbomind MoE (expert-parallel vs pad).
5. **cudagraph + real prompts** — throughput vs the FP16/ours baseline; confirm FP8 mem win + no
   capture regression.

---

## Open decisions to confirm before coding

1. **FINDING 1 (wiring target)** — retarget to the native Fp8 path (`vllm_serve.py`), CT as secondary?
   This reverses the Codex card's stated file. (Recommend yes — it's where the checkpoints live.)
2. **FINDING 2 (namespace)** — fix `turbomind_fp8_backend.py` op namespace in this change (hard blocker).
3. **Dense block consumer is real** — Qwen3.5-27B-FP8 is dense block-FP8; and in the MoE models the
   *full-attention* layers' qkvo are block-FP8 (linear_attn/gate/visual are `modules_to_not_convert`).
   So the dense turbomind branch is NOT speculative.
4. **MoE all-or-nothing per layer** — turbomind only if w13 AND w2 both eligible; else the whole
   fused-MoE layer stays on the existing path. Agree?
5. **Free raw FP8 after prepare** — dense frees [N,K] FP8 once packed; MoE frees per-expert raw after
   strided ptrs built. Preserves the half-memory win. (Stage A/D used prepare output standalone.)
6. **TP8 deferred, not solved** — TP≤4 is the Stage-F serving target; TP8 turbomind MoE is follow-on.
   Graceful fallback means TP8 still serves (via ours), just without the engine.
