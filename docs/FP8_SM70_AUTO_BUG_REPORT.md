# Bug report — `fp8_gemm_sm70_out_auto` returns garbage on 1Cat-vLLM SM70 FP8

**Project:** 1CatAI / 1Cat-vLLM (SM70 FP8 W8A16 TurboMind s884 path)
**Repo HEAD tested:** `f1a64a76f` — built for `TORCH_CUDA_ARCH_LIST=7.0`, image `1catai-vllm-v100:cu128-fp8sm70`
**Hardware:** NVIDIA Tesla V100-SXM2-32GB (sm_70), CUDA 12.8, torch 2.9.1+cu128
**Severity:** correctness — silent wrong output (no error, no warning), cos ≈ 0 vs reference
**Date:** 2026-07-03

> This is a correctness/contract bug report only. It contains **no** performance
> comparison — the fixed path is fast and correct; this is purely about the broken
> convenience entry point that an external integrator will reach for first.

---

## Summary

`torch.ops._C.fp8_gemm_sm70_out_auto(out, x, tm_weight, tm_scales)` produces
**numerically garbage output** (cosine similarity ≈ 0 vs an FP32 dequant reference)
for every problem shape and every M we tested. The two sibling entry points against
the **same prepared weights** are correct (cos = 1.0000):

- `fp8_gemm_sm70_out(out, x, tm_w, tm_s, group_size, k_ld, q_ld)` — explicit ld ✅
- `fp8_gemm_sm70_out_meta(out, x, tm_w, tm_s, meta)` — ld read from the `meta` tensor ✅
- `fp8_gemm_sm70_out_auto(out, x, tm_w, tm_s)` — ld reconstructed internally ❌

Because `_auto` fails silently (no `TORCH_CHECK`, no NaN), an integrator who calls it
gets plausible-looking fp16 output that is entirely wrong.

## Raw evidence (self-contained repro `repro_fp8_auto_bug.py`)

```
# shape  N=256 K=256 block=128
# meta (PACKED ld from prepare/Convert):  k_ld=8192  q_ld=256
# nominal (unpacked) geometry the _auto path reconstructs from: k=256 n=256
# -> packed k_ld != nominal k(256); packed q_ld == nominal n(256)
   M path                  cos      max_abs
   1 fixed              1.0000    3.212e-04
   1 meta               1.0000    3.212e-04
   1 auto               0.2039    1.085e+00
   4 auto               0.1282    1.014e+00
  16 auto               0.1659    1.153e+00

# shape  N=1024 K=512 block=128 :  packed k_ld=16384 (= 512*32)   auto cos ~0.02-0.04
# shape  N=3072 K=4096 block=128:  packed k_ld=131072 (= 4096*32) auto cos ~ -0.01..0.01
```

Key observation: the **packed weight leading dimension is exactly `32 × K`** in every
case (8192=256·32, 16384=512·32, 131072=4096·32). The scale leading dimension `q_ld`
happens to equal the nominal `N` here (so it is coincidentally recoverable), but the
weight ld is not — and `_auto` uses the nominal `K` (32× too small), so the GEMM
strides through the packed weight buffer incorrectly.

## Root cause (source read)

The packed leading dimension is produced *during packing* and is not recoverable from
geometry alone:

1. `LayoutConverter::Convert` takes its destination descriptor by **non-const reference**
   and overwrites `.ld` with the swizzled/packed value:
   - `lmdeploy/src/turbomind/kernels/gemm/convert.h:15-19` — `MatrixLayout& Ddesc`
   - `lmdeploy/src/turbomind/kernels/gemm/convert_v3.cu:48` —
     `Ddesc.ld = mk2cs<order_>(Packing_v2<pack, order_>::apply({Sdesc.rows, Sdesc.cols})).x;`

2. `fp8_sm70_prepare` runs `Convert` for weight and scales, then stores the **post-Convert
   packed ld** into `meta`:
   - `csrc/quantization/awq/awq_sm70_gemm.cu:838-844` (weight Convert),
     `:879-885` (scale Convert), `:887-889` — `meta = {k_desc.ld, q_desc.ld}`

3. The **fixed** path takes `k_ld/q_ld` as arguments and writes them straight into the
   GEMM descriptors — correct:
   - `csrc/quantization/awq/awq_sm70_gemm.cu:1149` `desc_B.ld = k_ld`,
     `:1173` `desc_V.ld = q_ld`

4. The **`_auto`** path rebuilds `desc_B`/`desc_V` from `(n, k, order, pack)` via the
   `MatrixLayout` constructor + `transpose`, but **never runs `Convert`**, so those
   descriptors keep their *nominal* (unpacked) ld. It then passes those nominal ld values
   as `k_ld/q_ld`:
   - `csrc/quantization/awq/awq_sm70_gemm.cu:1215-1279` (whole function),
     `:1277-1278` — `fp8_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, group_size,
     desc_B.ld, desc_V.ld);`

The nominal `desc_B.ld` differs from the packed ld by the pack factor (32× for this
FP8 s884 converter), so the weight is read with the wrong stride → garbage.

## Suggested fix (either is fine)

- **Preferred:** make `_auto` compute the packed ld the same way `Convert` does —
  `mk2cs<order>(Packing_v2<pack, order>::apply({rows, cols})).x` — instead of leaving
  the constructor default. The packed ld is a pure function of `(rows, cols, pack, order)`;
  no data or GPU work is needed to derive it.
- **Or:** remove/deprecate `_auto` and route external callers to `_meta` (which already
  works) or the explicit-ld form. 1Cat-vLLM's own call sites already thread the packed ld
  from `meta` (`vllm/.../fp8_sm70_moe.py:166-173`), so `_auto` is effectively an
  unexercised trap for outside integrators.
- **Minimum:** add a `TORCH_CHECK`/warning so `_auto` cannot fail silently.

> **Integrator note (our side).** Any serving wrapper over this engine must treat
> `prepare` → `meta` (`k_ld`, `q_ld`) as a **required contract**: thread the packed ld
> explicitly (dense) or via `awq_moe_build_strided_ptrs` (MoE), and **fail loudly** if it is
> missing or a shape is unsupported. `_auto` must never be on a serving path. The silent
> low-cos failure — not the bug itself — is the hazard.

## How to reproduce

```bash
# in the 1catai SM70 FP8 image (torch.ops._C.fp8_sm70_prepare present)
python3 repro_fp8_auto_bug.py --N 256  --K 256  --Ms 1 4 16
python3 repro_fp8_auto_bug.py --N 1024 --K 512  --Ms 1 4 16
python3 repro_fp8_auto_bug.py --N 3072 --K 4096 --Ms 1 4 16
```

Repro script: `tools/turbomind_ab/repro_fp8_auto_bug.py` (self-contained; synthesizes an
FP8 block-128 weight + activations, runs all three entry points against a shared FP32
dequant reference). Raw log: `tools/turbomind_ab/repro_auto_bug_20260703_052923.log`.
