# FP8 Engine — Stage H: wire compressed-tensors BLOCK-128 → TurboMind (design note)

Status: **DESIGN — for review before code.** Scope of this cut: **DENSE Linear only.** CT-block MoE
deferred to a follow-up (Stage H2). Fresh branch: `fp8-ct-block-turbomind-wiring`.

## 1. Motivation (a coverage/memory finding, not perf polish)

Measured 2026-07-07 (`results/gemma4_functional_20260707_221319/`), all on the baked prod image
`vllm-v100:vllm021-cu126-fp8engine`, ours path:

| Gemma-4 FP8 variant | scheme | path on ours | weights/wkr | result |
|---|---|---|---|---|
| gemma-4-31B-it-FP8-Dynamic | CT channel | FP8-resident | 16.93 GiB | ✅ TP2 |
| gemma-4-26B-A4B-it-FP8-Dynamic | CT channel MoE | FP8-resident | fit | ✅ TP2 |
| **gemma-4-31B-it-FP8-block** | **CT block-128** | **FP16-dequant at load** | **30.38 GiB** | ❌ **OOM at TP2** (KV −2.2 GiB); needs TP4 |

A checkpoint *labeled* FP8-block should not require FP16-resident memory. Today it does, because of a
two-layer backend boundary:

1. **Scheme layer** — TurboMind is block-128 only; channel/dynamic → ours. (Correct, unavoidable.)
2. **Format layer** — TurboMind is wired **only into the native-fp8 loader** (`quant_method=fp8`;
   `select_backend` is called at `vllm_serve.py:~1502` dense / `~1577,1582` MoE). The
   compressed-tensors loader (`compressed_tensors_v100.py`) has **zero** turbomind references, so *every*
   CT weight — including block-128 — runs on ours. And ours' FP8-resident path is **CHANNEL/TENSOR only**
   (`compressed_tensors_v100.py:370-373` requires `"CHANNEL" in strat`), so CT-block falls to
   `_dequant_ct_weight_to_fp16` (line 423) → 2× weight VRAM → the OOM above.

**Fixing layer 2 for block strategy gives both a memory win (stay FP8-resident, no FP16 dequant → fits
TP2) and TurboMind acceleration** — the same story Qwen native-fp8 already gets. It's buyer-relevant:
a large share of aftermarket FP8 checkpoints are RedHatAI compressed-tensors.

## 2. The key enabler (why this is plumbing, not new kernel math)

The block dequant is identical on both loaders:
- CT block (`_dequant_ct_weight_to_fp16`, line 232-235): `dequant = fp8_weight * weight_scale`, where
  `weight_scale` is `[N/128, K/128]` fp32, expanded 128×128.
- Native block (`_tm_dense_prepare`, `vllm_serve.py:1239`): `tb.prepare(weight, weight_scale_inv, gs=128)`
  where `weight_scale_inv` is `[N/128, K/128]` — despite the "inv" name, it is the *multiplicative* per-block
  dequant scale.

⇒ **CT `weight_scale` (block) ≡ native `weight_scale_inv` semantically.** So
`tb.prepare(ct_weight.contiguous(), ct_weight_scale.float().contiguous(), 128)` is a drop-in. This is the
central assumption and MUST be gate-checked by a loader-cos self-check (§5) before trusting any layer.

## 3. Interception points (both in `patch_compressed_tensors_for_v100`)

**(a) Load — `patched_process_weights_after_loading` (line 353).** Add a BLOCK-turbomind branch, evaluated
before the existing channel-resident / FP16-fallback decision:

```
strat = str(self.strategy).upper(); wbs = self.weight_block_size   # [128,128] for block ckpts
w = layer.weight.data                                              # FP8 e4m3 [N,K]  (local, post-TP-shard)
N, K = w.shape
if ("BLOCK" in strat) and wbs is not None and not excluded:
    tb = _tm_backend()                                            # from fp8_w8a16_sm70.vllm_serve import
    backend, why = tb.select_backend(strategy="BLOCK", weight_block_size=tuple(wbs),
                                     local_n=N, local_k=K, need_moe=False, mode=None)   # auto/ours/turbomind
    if backend == "turbomind" and _ct_block_selfcheck_ok(w, layer.weight_scale, N, K):  # §5
        _tm_dense_prepare(self, layer, w.contiguous(), layer.weight_scale, (128,128), f"ct-block: {why}")
        layer._v100_ct_tm = True
        _ct_log_decision(layer, "resident", f"tm-block N={N},K={K}")
        return
    # not eligible (TP-shard misalign / engine absent / self-check fail) -> existing FP16 dequant fallback
```

Notes:
- `select_backend` already enforces `local_n%128==0 and local_k%128==0` and ops-present, and honors
  `VLLM_V100_FP8_BACKEND` (`auto` default engages tm; `ours` forces skip; `turbomind` raises if ineligible).
  So mixed eligible/fallback within one model (some Gemma layers whose TP2 shard isn't 256-aligned) is
  handled exactly like the 122B mixed path — those layers fall to FP16 dequant on ours.
- `_tm_dense_prepare` frees the raw FP8 weight when `_TM_FREE_RAW` (default ON) → the memory win. Its
  `weight_scale_inv`-param free branch won't match CT's `weight_scale` param; free CT's `weight_scale`
  explicitly in the CT branch after prepare (small tensor; avoids a dangling scale param).

**(b) Apply — `patched_apply_weights` (line 442).** Add the turbomind branch first:

```
if getattr(layer, "_v100_ct_tm", False):
    from fp8_w8a16_sm70.vllm_serve import _tm_dense_apply
    return _tm_dense_apply(layer, x, bias)
if getattr(layer, "_v100_ct_resident", False):   # existing channel path
    ...
# existing FP16 fallback
```

This **reuses the proven native `_tm_dense_prepare`/`_tm_dense_apply`** (one turbomind apply path for both
loaders) — the recommended approach over duplicating `tb.prepare`/`tb.gemm_out` inside the CT file.

## 4. Eligibility & expected coverage (Gemma-4-31B-FP8-block, TP2)

Block-128 quant ⇒ per-tensor dims are 128-multiples at TP1. At TP2 a weight is eligible iff its *local*
(sharded) N and K stay %128==0 (i.e. original %256==0 on the sharded axis). Layers that don't (e.g. an
o_proj with an awkward head split) fall back to FP16-dequant on ours — same mixed behavior as 122B TP8.
Expected: most Linears go turbomind-resident ⇒ per-worker weights drop from **30.38 GiB → ~15–16 GiB**,
leaving KV headroom ⇒ **fits at TP2**.

## 5. Safety net — per-layer loader-cos self-check (mandatory)

Mirror the existing channel self-check (`compressed_tensors_v100.py:384-405`): for each candidate
turbomind block layer, at load, compare `tb.gemm_out` on a random probe against the FP16-dequant reference;
if `L2rel ≥ 1e-2`, DO NOT commit turbomind — fall back to FP16 for that layer and log it. Guards the §2
scale-semantics assumption per-layer, in one load, exactly like the channel path already does.

```
def _ct_block_selfcheck_ok(w, wscale, N, K):     # returns True to allow turbomind
    tb = _tm_backend()
    probe = torch.randn(4, K, device=w.device, dtype=torch.float16) * 0.1
    scale_exp = wscale.to(torch.float16).repeat_interleave(128,0).repeat_interleave(128,1)[:N,:K]
    ref = F.linear(probe, (w.to(torch.float16) * scale_exp))
    tm_w, tm_s, meta = tb.prepare(w.contiguous(), wscale.float().contiguous(), 128)
    out = torch.empty((4, N), dtype=torch.float16, device=w.device)
    tb.gemm_out(out, probe.contiguous(), tm_w, tm_s, int(meta[0]), int(meta[1]), 128)
    return (out.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12) < 1e-2
```
(Prepare is done twice — once in the check, once in `_tm_dense_prepare`; acceptable at load, or thread the
packed tensors through to avoid the repeat. Optimize only if load time matters.)

## 6. Scope boundary — dense now, MoE later (Stage H2)

CT-block MoE (`patch_compressed_tensors_moe_for_v100`, the `_ct_moe_*` machinery) is out of scope for this
cut. No common CT-block MoE checkpoint is on disk today (Gemma-4-26B-A4B is *channel* MoE → ours). When one
appears, mirror `_tm_moe_prepare`/`_tm_moe_apply` (`vllm_serve.py:1285,1362`) into the CT MoE loader with the
same select_backend gate + w13/w2 per-shard eligibility (w2 K=I/TP is the alignment risk, as in native).

## 7. Gate ladder (same rigor as Stage F)

1. **Loader-cos** — per-layer self-check green on gemma-4-31B-FP8-block (no BAD→FP16 for aligned layers).
2. **Serving agreement** — gemma-4-31B-FP8-block ours-vs-`auto`(turbomind), eager/greedy, via
   `tools/turbomind_ab/exactness_ab.sh` (now a REAL A/B — this is the model that couldn't A/B before).
   Expect dense agreement like Qwen-27B (near-100%, benign ties).
3. **Memory** — confirm TP2 fit: per-worker weights ~15–16 GiB (not 30.38), KV headroom > 0. The OOM is gone.
4. **Perf** — dense C1/C8 + TTFT vs the FP16-dequant baseline (should match the Qwen native-fp8 story).

## 8. Rollout / risk

- Gated by the existing `VLLM_V100_FP8_BACKEND` (`auto` default). Engine-absent image → `select_backend`
  returns ours → unchanged FP16 dequant. So the change is **inert without the baked engine**, and the
  self-check makes it inert per-layer on any scale-semantics surprise.
- Add a CT-specific kill switch `VLLM_V100_CT_BLOCK_TM=0` (default on) for a fast revert without touching
  the global backend selector.
- Primary risk = the §2 scale assumption; fully covered by §5. Secondary = load-time cost of the extra
  prepare (mitigate by threading packed tensors from the self-check if it matters).

## 9. Deliverable checklist for the code PR

- [ ] `_ct_block_selfcheck_ok` helper + BLOCK-turbomind branch in `patched_process_weights_after_loading`.
- [ ] turbomind branch in `patched_apply_weights` (reuse `_tm_dense_apply`).
- [ ] explicit free of CT `weight_scale` param after prepare.
- [ ] `VLLM_V100_CT_BLOCK_TM` kill switch.
- [ ] gate-ladder run on gemma-4-31B-FP8-block; record TP2-fit + agreement in `results/`.
