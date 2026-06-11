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

**Decode (single stream, cudagraph, coalesced GEMV default):**

| context | decode tok/s | KV cache | concurrency |
|---|---:|---:|---:|
| 2k (shallow) | **56.6** (30.7 A.3 → 45.4 coalesced-attn → 56.6 +coalesced-MoE-w13) | 563k tok | 275× |
| 6k depth | ~27 + coalesced lift | — | — |
| 26k depth | ~18.6 + coalesced lift | — | — |

The **coalesced FP8 GEMV** (default on) accelerates the attention/dense Linears:
GLM-Air TP=8 A/B = a3 (old A.3) 30.81 → **coalesced 45.37 tok/s (1.47×)**, which is
**95% of the FP16-attn ceiling (47.82)** at *full* FP8 residency (max KV). The
1.47× (vs ~1.9× on lower-TP models) is tempered by TP=8 all-reduce being a larger
fixed fraction. Decode-at-depth falloff is the attention-over-KV cost (inherent to
any model), not our FP8 kernels (O(1)/token). vs the FP16-fused-MoE cudagraph
alternative (4.81 tok/s / 78× concurrency), this config wins decisively on both
speed and concurrency. See `docs/COALESCED_FP8_GEMV.md` for the kernel.

**Prefill TTFT (26k-token prompt):** 60.2s (was 169s before Phase 4, **2.8×**).
The remaining cost was ~70% self-attention — now addressed by FlashAttention-V100
(below): **TTFT@24k 51.8s → 19.45s (2.66×)**, total journey 169s → ~19.5s (~8.7×).

## FlashAttention-V100 prefill (opt-in; recommended for long context)

`VLLM_V100_FLASH_ATTN=1` routes prefill attention batches from TRITON_ATTN's
`unified_attention` to the ai-bond flash-attention-v100 kernel (8.4× vs Triton at
26k: 112.6 vs 945.4 ms/layer — the Triton prefill kernel runs ~2.2 TFLOP/s on
Volta, tensor-core-less). Decode keeps the validated Triton+cudagraph path **by
construction** (decode-only graphs never see the FA path; e2e decode is
digit-identical with the flag on/off).

**Measured (GLM-Air TP8, 24,040-token prompt):** TTFT 51.8 → **19.45s (2.66×)**,
output coherent and tracking the baseline greedy trace. Under 4-user concurrent
load: worst-case TTFT 93.7 → 45.0s (2.1×), 13/13 coherent, decode parity, zero
preemptions.

**Requirements:** `--block-size 256` (multiple-of-256; empirically neutral vs
block-16 — identical KV pool and behavior), fp16 KV cache, and the ai-bond
extension built for this image (torch 2.11+cu126; 3 source patches in the local
fork, see `docs/FA_V100_AUDIT.md`). Expose ONLY `flash_attn_v100_cuda*.so` to the
container PYTHONPATH — never ai-bond's `flash_attn` python shim (it breaks vLLM's
optional flash-attn probe). The adapter falls back to Triton per-call for anything
ungated (non-fp16 KV, sliding window, head_dim ∉ {128,256}, fp8-KV, alibi/sinks).
Kill switch: `VLLM_V100_FLASH_ATTN=0` (default).

The four integration invariants + full audit trail: `docs/FA_V100_AUDIT.md`.
Model applicability: D=128/256 full-attention layers (GLM-Air = headline 2.66×;
GDN-hybrids like Qwen3.5-122B get a bounded win — only their full-attn layers
route; Gemma-4 sliding layers pend a window gate).

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
| `VLLM_V100_FP8_COALESCED_GEMV=1` | coalesced decode GEMV for attn/dense Linears (30.81→45.37, 1.47×) | `=0` → A.3 |
| `VLLM_V100_FP8_COALESCED_UNROLL=4` | K-unroll depth (4 = the measured knee) | — |
| `VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8` | max batch on the coalesced path (8 = batched decode) | `=1` single-stream |
| `VLLM_V100_CT_MOE_W13_COALESCED=1` | grouped coalesced GEMV for MoE w13 decode (45.4→56.6, +25%) | `=0` → grouped a3 |
| `MODE=cudagraph` | mode=0 + FULL_DECODE_ONLY + TRITON_ATTN | `MODE=eager` |
| `VLLM_V100_CT_PROFILE=0` | per-section prefill GPU+wall timers (use with `MODE=eager`) | `=1` to enable |
| `VLLM_V100_FLASH_ATTN=0` | **opt-in**: FA-V100 prefill (TTFT@24k 2.66×; needs `--block-size 256`) | default off |

Decode (R<256) always uses the per-row grouped kernels + cudagraph — the prefill
tiled/fused paths never touch it.

## Validation tooling (all in `tools/`)

- `ct_fp16_w2_grouped_numtest_vllm021.sh` — decode w2 grouped kernel
- `ct_fp8_tiled_prefill_numtest_vllm021.sh` — tiled prefill vs per-row + FP32
- `ct_fp8_fused_tiled_numtest_vllm021.sh` — fused grouped-tiled kernel (bit-identical to a2)
- `ct_fp8_channel_wmma_numtest_vllm021.sh` — channel-WMMA vs FP32 + block_h=128 no-regression
- `prefill_wmma_microbench_vllm021.sh` — WMMA-vs-A.2 + grouped-MoE prefill cost
- `ct_mixed_moe_e2e_diff_vllm021.sh` — e2e token-diff vs FP16 baseline
- `fa_v100_build_microbench.sh` — FA: build + test.py + paged smoke + longseq/layout gates + microbench
- `fa_v100_longseq_check.py` — FA: 512→24k paged sweep, interleaved KV, Sq<Sk/mixed-batch, strided-q probe (`--d/--hq/--hk`)
- `fa_prefill_ttft_ab_vllm021.sh` — FA: e2e TTFT A/B (triton vs fa arms)
- `fa_concurrency_soak_vllm021.sh` — FA: 3-arm multi-user soak (block-size + FA decomposed)

## Open / future (separate projects)

- ~~Attention prefill~~ **DONE** — FlashAttention-V100 section above (169s→~19.5s total).
- **Stage 1.5b**: partial-N (N=352) WMMA to fold w13's 12s prefill GPU into the
  fused kernel (~10s off TTFT) — polish only.
- **Gemma-4 FP8** — direct transfer of this stack (no MLA blocker); FA sliding-window
  gate (P3) pends for its sliding layers.
- **FA upstream**: 3 ai-bond fixes ready as issue/PR (fa repo commit `ccb6557`).
