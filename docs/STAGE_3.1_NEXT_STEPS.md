# Stage 3.1 — next-session action plan (drafted 2026-05-26 end-of-session)

This file is the handoff into the next session. It encodes two diagnostic
plans that came out of the dense FP8 / GPTQ-Int4 / FP16 three-way
comparison on Qwen3.6-27B (see SESSION_LOG.md Stage 3 + Stage 3.1
sections):

A. A small microbenchmark to attribute the dense-FP8 slowness to a
   specific cause (kernel vs dispatch vs fusion).
B. A controlled upgrade ladder (driver → CUDA runtime → torch → vLLM)
   for chasing further perf gains beyond the v0.4.0 baseline without
   regressing it.

The deployment rule is already settled by the data:

| Workload                        | Use                                                 |
|---------------------------------|-----------------------------------------------------|
| 122B / 35B-A3B (MoE+GDN, prod)  | our FP8 (34.76 / 52.87) OR GPTQ-Int4 (63.62 on 122B)|
| Dense+GDN (27B-class)           | GPTQ-Int4 (47.61) or FP16 (39.60); NOT our FP8      |
| Small dense                     | FP16 or Int4; not our FP8                           |

Neither plan below changes this rule — they only justify whether
investing engineering in dense FP8 is ever worth it. Today, both
Claude and GPT independently concluded it isn't, given Int4 already
beats best-plausible FP8 by ~4× on dense V100.

---

## Plan A — three-step dense FP8 microbench

Goal: explain the 76 ms/token gap between our FP8 (11.44 tok/s,
~87 ms/token) and GPTQ-Int4 (47.61 tok/s, ~21 ms/token) on Qwen3.6-27B
at TP=4 on V100. Cheapest probes first, hardest last.

### Step 1 — direct C++ entry-point timing

Invoke each of our four kernel variants in a tight loop on
representative dense 27B shapes, no Python wrapper, no model context.
Compare to cuBLAS FP16 `torch.matmul` and (if accessible) GPTQ
exllama's Int4 GEMM on the same shapes.

Representative shapes (Qwen3.6-27B, hidden=5120, intermediate=17408,
TP=4 so per-rank hidden_per_tp=5120, intermediate_per_tp=4352):

| Linear in the model              | M  | N    | K    |
|----------------------------------|----|------|------|
| Q proj / KV proj (attention)     | 1  | 5120 | 5120 |
| GDN in_proj_qkvz                 | 1  | ~10k | 5120 |
| MLP gate+up (fused)              | 1  | 8704 | 5120 |
| MLP down                         | 1  | 5120 | 4352 |
| RowParallel attn out / down_proj | 1  | 5120 | 5120 |

For each shape, time:
1. `_ext.fp8_w8a16_gemm_a1(x, w, scale, N, K, 128, 128)`
2. `_ext.fp8_w8a16_gemm_a2(...)`
3. `_ext.fp8_w8a16_gemm_a3(...)`
4. `_ext.fp8_w8a16_gemm_wmma_poc(...)` (only when M is 64-aligned)
5. cuBLAS FP16 baseline: `torch.matmul(x_fp16, w_fp16)`

The right tool is a small benches/bench_fp8_gemm_kernels.py script
that warms up, then runs each variant in a `torch.cuda.synchronize()`
fenced loop with `torch.cuda.Event` timing. Output a table.

### Step 2 — Python wrapper overhead probe

Wrap `_v100_fp8_gemm` with a profiling shim that records per-call
Python overhead separately from the actual `_ext.*` call. Or simpler:
replace `_v100_fp8_gemm` with a passthrough that returns a precomputed
output tensor of the right shape (no GPU work at all), and measure
how much of the 87 ms/token wall disappears.

If 20-40 ms/token disappears with the passthrough, Python wrapper
dispatch is real and accounts for ~25-45% of the cost. That matches
the "5-10 μs × 4096 calls/token" rough estimate.

### Step 3 — three-branch attribution

After Steps 1+2 we can answer:

| Observation                                              | Root cause              | Easy fix?                                                            |
|----------------------------------------------------------|-------------------------|-----------------------------------------------------------------------|
| Direct `_ext.*` matches cuBLAS FP16 within 1.5×          | Wrapper or fusion       | Bypass `_v100_fp8_gemm`; try `torch.library.custom_op`+FULL_AND_PIECEWISE |
| Direct `_ext.*` is 3×+ slower than cuBLAS FP16           | Kernel/dequant itself   | Hard — would need .cu refactor; not worth vs Int4                     |
| Direct fine and wrapper fine but model-level still slow  | Graph/launch/fusion     | `torch.library.custom_op` refactor is the right call                  |

### What we'll NOT do without microbench data

- Pre-dequantize at load time (wastes 2× memory, lands at FP16; useful
  only if microbench rules out the cheap fix).
- Inline dequant into GEMM kernel (substantial CUDA work; only after
  Steps 1-2 prove this is the bottleneck).

---

## Plan B — controlled upgrade ladder beyond v0.4.0

Goal: find low-risk perf gains without regressing the v0.4.0 production
baseline. One variable at a time, smoke-tested before promoting.

### Golden baseline (current v0.4.0)

- Python 3.12
- vLLM 0.18.0
- torch 2.10.0+cu128
- CUDA 12.8 runtime
- ubuntu 24.04 base
- NVIDIA driver: **TBD — check with `nvidia-smi --query-gpu=driver_version --format=csv,noheader`**
- mode=0, FULL_DECODE_ONLY
- VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1
- VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=128

### Smoke matrix (apply at every step)

1. Qwen3.5-122B-A10B-FP8 TP=8 → must stay ≥34 tok/s
2. Qwen3.6-35B-A3B-FP8 TP=4 → must stay ≥52 tok/s
3. Qwen3.6-27B FP16 or GPTQ-Int4 TP=4 → sanity check dense path
   (≥39 or ≥47 tok/s respectively)

### Rung 1 — Driver upgrade (safest, do this first)

Current likely 535 (per GPT's read of the cu128 stack). Candidates:
R570 / R575 / R580. Must NOT pull in a driver/toolkit combo that
implies CUDA 13 (drops sm_70).

Expected gain: modest. Possibly improves NCCL all-reduce on the DGX-1
hypercube (which Stage 2D measured as ~41% of decode wall on the old
eager stack — that fraction is now even higher on cudagraph since
compute compressed 6.83×).

After driver bump, run the smoke matrix. If any rung regresses,
revert driver.

### Rung 2 — CUDA runtime image

Try cu129 if torch/vLLM wheels are available cleanly. Do NOT try
CUDA 13 (no Volta).

Same image rebuild: `docker/Dockerfile.vllm018_py312` with the
base bumped to `nvidia/cuda:12.9.x-devel-ubuntu24.04`. Re-test
smoke matrix.

Expected gain: low. The cu128/cu129 delta on Volta is small.

### Rung 3 — torch upgrade

Only after Rung 1+2 are stable.

Before any bump, confirm:
```python
import torch
print(torch.cuda.get_arch_list())  # must contain '7.0' or 'sm_70'
```

If a candidate torch (2.11 / 2.12 / nightly) doesn't include sm_70 in
its built kernel set, it's dead on arrival for V100.

If sm_70 is included, install in a sibling image
(`vllm-v100-py312-torch211:cu128`) and re-smoke. Keep the v0.4.0 image
buildable.

### Rung 4 — vLLM upgrade (highest risk)

vLLM 0.19 changelog likely has internal API changes that could break
our monkey-patches (specifically: `Fp8Config.get_min_capability`,
`Fp8LinearMethod.apply`, `FusedMoEMethodBase`, the V1 engine init
flow, breakdown hooks). The cu128 image worked because vLLM 0.18 was
exactly the API our patches target.

Pre-check before any bump: grep vLLM 0.19+ release notes for any of:
- `quantization.fp8` API changes
- `FusedMoEMethodBase` changes
- `gpu_model_runner.py` changes around `_update_hybrid_attention_mamba_layout`
- `compilation_config` API changes

Treat as a controlled branch, not the new default. Plan ~1 day of
compatibility testing before promotion.

---

## Open Stage 3.5+ items (orthogonal to A and B)

These exist independently of the dense-FP8 question and the upgrade
ladder:

1. **`torch.library.custom_op` refactor** of the 5 `_ext.*` entry points
   with fake/meta impls. Unlocks mode=3 + FULL_AND_PIECEWISE. Could
   add 20-40% to v0.4.0 numbers. ~1-2 days.
2. **`is_current_stream_capturing()` guard** in breakdown hooks
   (vllm_serve.py:492, 502, plus Stage 2D Step 2D.3 row_parallel_ar
   timer). Makes Stage 2D profiling safe under cudagraph. ~10 LOC.
3. **Stage 2D re-profile on v0.4.0 stack** (currently needs
   `--enforce-eager`). Under cudagraph the AR fraction of decode is
   likely much higher relative to compute since compute compressed
   6.83×. If AR is now >60% of decode, MTP becomes a natural lever.
4. **V100 MoE config autotune**. vLLM ships no
   `E=128/N=192/Tesla_V100-SXM2-32GB.json`. Could add 20-40% on
   MoE path. Use `vllm/model_executor/layers/fused_moe/benchmark_moe.py`.
5. **MTP via 1catai** (after their `language_model.` loader patch +
   FP8 gate bypass + wheel rebuild). Potential 2-3× on top of v0.4.0
   if speculative decode amortizes AR cost. ~1-2 days.
6. **Quality benchmark suite** comparing 122B FP8 vs Int4 outputs
   side-by-side on standard prompts. Currently only validated on
   anecdotal Rayleigh probe.
7. **Determinism investigation** — observed minor `\n\n` vs `\n`
   variance between curls 1-2 and 3-4 at temp=0 on 122B-FP8. Doesn't
   affect semantic quality but worth understanding before any A/B
   that depends on bit-level reproducibility.
8. **README + REQUIREMENTS.md pivot** — currently lead with py3.10
   cu128 stack; should lead with py3.12 v0.4.0.
