# Stage E (2/2) — `VLLM_V100_FP8_BACKEND` adapter (scaffolding)

Branch `fp8-engine-stage-e-adapter`. Module: `src/fp8_w8a16_sm70/turbomind_fp8_backend.py`.

## What this is
The backend-selection shell + TurboMind call wrappers that let our vLLM choose, per FP8 weight,
between our coalesced GEMV (`ours`) and the upstream-lmdeploy SM70 s884 FP8 engine (`turbomind`).
Our path stays the compat/fallback/control path; TurboMind is the fast path where the checkpoint
format matches (Stage C). Engine is sourced from upstream lmdeploy (Stage E source audit).

`VLLM_V100_FP8_BACKEND = ours | turbomind | auto`  (default: auto).

## Eligibility (auto picks turbomind ONLY IF ALL hold; else ours + one-line reason)
1. block-FP8, group_size 128 (`weight_block_size == (128,128)`);
2. scales representable as fp32 block scales (implied by BLOCK);
3. **LOCAL (post-TP-shard) dims block-128-aligned** (`N%128==0 and K%128==0`) — rules out TP8-on-Qwen
   (I/tp=64), the Stage-D finding;
4. prepare metadata (k_ld,q_ld) obtainable;
5. SM70 FP8 ops actually built into the image;
6. NOT channel/tensor scale;
7. call path never uses `_auto`.
`mode=turbomind` on an ineligible weight **raises** (no silent fallback — silent wrong backend is the
hazard). `mode=auto` falls back to `ours` with the reason logged/observable.

## Validated now (no engine required)
`python3 src/fp8_w8a16_sm70/turbomind_fp8_backend.py` — self-test PASS across: block-128 dense/MoE →
turbomind; channel/tensor → ours; TP8 I/tp=64 → ours; ops-absent → ours; `mode=ours` forced; and
`mode=turbomind` on ineligible raises. Real-image check (`vllm-v100:vllm021-cu126`): `ops_available()
== False` today, so every block-128 weight correctly resolves to `ours` with reason "SM70 FP8 ops not
present in this image". The `ours` path is unchanged; nothing switches to turbomind until the engine is
built in (Stage F infra).

## TurboMind call wrappers (the required contract)
`prepare(qweight, scales)` → `(tm_weight, tm_scales, meta[k_ld,q_ld])` (casts scales→fp32);
`gemm_out(out, x, tm_w, tm_s, k_ld, q_ld)` — dense, explicit-ld entry (**never `_auto`**);
`moe_gemm_out(out, sorted_x, expert_offsets, ptrs_w, ptrs_s, E, k, n)` — grouped (strided ptrs carry ld).
These raise if the ops are absent; callers gate on `select_backend(...) == "turbomind"`.

## Remaining (Stage F)
- [ ] Build upstream lmdeploy SM70 gemm (+ the FP8 prepare/gemm/moe_gemm bindings) into our vLLM image
  so `ops_available()` becomes True and the turbomind path goes live.
- [ ] Wire `select_backend()` into `compressed_tensors_v100.py` create_weights/process_weights so each
  FP8 layer records + acts on its backend (turbomind prepare at load; ours otherwise).
- [ ] Serving validation: TP≤4 clean, TP8 sharding decision, eager + cudagraph, real-prompt sanity.
