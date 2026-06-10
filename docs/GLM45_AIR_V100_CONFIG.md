# GLM-4.5-Air-FP8 on 8×V100 — validated serving config

The promoted, validated envelope for serving **GLM-4.5-Air-FP8**
(`zai-org/GLM-4.5-Air-FP8`, `Glm4MoeForCausalLM`, compressed-tensors channel
W8A8-FP8, 46 layers / 128 routed experts / 1 shared) on **8×V100-32GB, TP=8**,
under stock vLLM 0.21 + the `fp8_w8a16_sm70` patches.

All optimizations below are **on by default** — no extra env needed beyond the CT
residency flags. Numbers are file-verified (2026-06-09).

## Run it

```bash
VLLM_V100_CT_FP8_RESIDENT=1 \
VLLM_V100_CT_MOE_W13_RESIDENT=1 \
VLLM_V100_CT_MOE_W13_FREE_FP16=1 \
./tools/glm45_air_fp8_load_vllm021.sh
```

`MODE=cudagraph` is the default (the validated best path). It expands to
`--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'` +
`VLLM_ATTENTION_BACKEND=TRITON_ATTN`. **`mode=0` is mandatory** — vLLM 0.21's
default `-O3` runs a TorchDynamo fullgraph trace that fatally rejects our
`torch._dynamo.disable`'d pybind kernels; `mode=0` skips Dynamo so they run eager
inside the captured graph. TRITON_ATTN is `AttentionCGSupport.ALWAYS`, so
`FULL_DECODE_ONLY` captures without downgrade. Use `MODE=eager` for profiling.

## Measured numbers

**Decode (single stream, cudagraph):**

| context | decode tok/s | KV cache | concurrency |
|---|---:|---:|---:|
| 2k (shallow) | **30.7** | 563k tok | 275× |
| 6k depth | 27.0 | — | — |
| 26k depth | 18.6 | — | — |

Decode-at-depth falloff is the attention cost over a deeper KV (inherent to any
model), not our FP8 kernels (O(1)/token). vs the FP16-fused-MoE cudagraph
alternative (4.81 tok/s / 78× concurrency), mixed-FP8 wins **6.4× speed AND 3.5×
concurrency** — it is the best GLM-Air config on this stack, not a memory-vs-speed
tradeoff.

**Prefill TTFT (26k-token prompt):** 60.2s (was 169s before Phase 4, **2.8×**).
The remaining cost is ~70% self-attention + TP all-reduce (vLLM internals); the
FP8/MoE host-loop bottleneck is retired (our code is GPU-bound). Further prefill
gains need an attention project (V100 FlashAttention / TP-overlap), not MoE work.

## The optimization stack (all default-on; each has a kill switch)

| flag (default) | what it does | kill switch |
|---|---|---|
| `VLLM_V100_CT_FP8_RESIDENT=1` | channel Linears FP8-resident (our W8A16 kernel) vs dequant→FP16 | `=0` |
| `VLLM_V100_CT_MOE_W13_RESIDENT=1` | MoE w13 (gate_up) FP8-resident via grouped kernel | `=0` |
| `VLLM_V100_CT_MOE_W13_FREE_FP16=1` | free the transient FP16 w13 → ~8.3 GB/GPU → KV 168k→577k (3.4× concurrency) | `=0` |
| `VLLM_V100_CT_MOE_W2_GROUPED=1` | decode w2 via one grouped kernel launch (not the per-expert Python loop) | `=0` |
| `VLLM_V100_CT_MOE_PREFILL_TILED=1` | at R≥256 (prefill), per-expert tiled GEMMs (weight reuse) | `=0` |
| `VLLM_V100_CT_MOE_PREFILL_FUSED=1` | fused w13 (one launch, GPU-side offsets, no `.tolist()`) + cuBLAS w2 | `=0` → per-expert loop |
| `VLLM_V100_CT_CHANNEL_WMMA=1` | channel Linears use WMMA (tensor cores) at prefill M | `=0` → A.2 |
| `MODE=cudagraph` | mode=0 + FULL_DECODE_ONLY + TRITON_ATTN | `MODE=eager` |
| `VLLM_V100_CT_PROFILE=0` | per-section prefill GPU+wall timers (use with `MODE=eager`) | `=1` to enable |

Decode (R<256) always uses the per-row grouped kernels + cudagraph — the prefill
tiled/fused paths never touch it.

## Validation tooling (all in `tools/`)

- `ct_fp16_w2_grouped_numtest_vllm021.sh` — decode w2 grouped kernel
- `ct_fp8_tiled_prefill_numtest_vllm021.sh` — tiled prefill vs per-row + FP32
- `ct_fp8_fused_tiled_numtest_vllm021.sh` — fused grouped-tiled kernel (bit-identical to a2)
- `ct_fp8_channel_wmma_numtest_vllm021.sh` — channel-WMMA vs FP32 + block_h=128 no-regression
- `prefill_wmma_microbench_vllm021.sh` — WMMA-vs-A.2 + grouped-MoE prefill cost
- `ct_mixed_moe_e2e_diff_vllm021.sh` — e2e token-diff vs FP16 baseline

## Open / future (separate projects)

- **Attention prefill** (the ~42s TTFT floor at 26k): V100 FlashAttention / paged
  prefill, TP all-reduce overlap.
- **Stage 1.5b**: partial-N (N=352) WMMA to fold w13's 12s prefill GPU into the
  fused kernel (~10s off TTFT) — polish only; attention is the real floor.
- **Gemma-4 FP8** — direct transfer of this stack (no MLA blocker).
