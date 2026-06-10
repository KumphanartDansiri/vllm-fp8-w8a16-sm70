# Coalesced FP8 W8A16 decode GEMV for V100 (sm_70)

A drop-in CUDA kernel that makes **FP8-resident decode on Volta nearly as fast as
FP16** — closing the gap that previously forced a choice between *fits in memory*
(FP8) and *decodes fast* (FP16). On V100 this is the difference between a model
loading-but-crawling and loading-and-serving.

**Who this is for:** V100 owners who **can't fit FP16** — 16 GB cards, low GPU
count / low tensor-parallel degree, or models simply too large for FP16 on the
hardware you have. If a model fits FP16 comfortably and you don't need the freed
memory, plain FP16 is marginally faster and simpler (see the decision matrix).

---

## The problem it fixes

At single-token decode (M=1), a Linear is **memory-bandwidth-bound** — time is
dominated by reading the weight matrix once. FP8 weights are *half the bytes* of
FP16, so an FP8 decode kernel *should* be ~2× faster than FP16 cuBLAS. The old
V100 path (`A.3`) was the opposite — **2–9× slower** — because its memory access
was badly coalesced: each thread owned one output column and strided across rows
by `K` bytes, so a warp scattered its loads across ~25 cache sectors instead of ~2.

Measured with Nsight Compute (M=1, N=K=5120):

| kernel | time | sectors/request | DRAM throughput |
|---|---:|---:|---:|
| `A.3` (old) | 219 µs | **25.3** | 17.8% |
| coalesced (this kernel) | 108 µs | **2.4** | 29.4% |
| FP16 cuBLAS (reference) | 82 µs | 2.0 | 77.4% |

The fix re-maps the kernel: **one warp per output column, lanes stride consecutive
K, warp-reduce the partial dot products** — so consecutive threads read consecutive
bytes (`sectors/request` 25.3 → 2.4, matching cuBLAS). A 4-deep K-unroll
(`UNROLL=4`) keeps enough loads in flight to hide HBM latency.

---

## Measured end-to-end results (V100-32GB, TP=4, cudagraph)

All numbers are **steady-state decode tok/s** (inter-token rate, excluding TTFT;
see "How we measured"). `coalesced` = `VLLM_V100_FP8_COALESCED_GEMV=1 UNROLL=4`.

### Dense — Gemma-4-31B-it-FP8 (compressed-tensors channel)

| path | decode tok/s | vs A.3 | vs FP16 |
|---|---:|---:|---:|
| A.3 (old FP8-resident) | 6.54 | — | 23% |
| **coalesced FP8-resident** | **28.24** | **4.3×** | **97.5%** |
| FP16 (dequant → cuBLAS) | 28.97 | — | 100% |

**FP8-resident at 97.5% of FP16 speed, using half the weight memory.**

### Dense/hybrid — Qwen3.6-27B-FP8 (block-FP8)

| path | decode tok/s | vs A.3 |
|---|---:|---:|
| A.3 (old) | 11.75 | — |
| **coalesced FP8-resident** | **37.24** | **3.17×** |

### MoE — Gemma-4-26B-A4B-it-FP8 (full FP8 residency)

The kernel accelerates the MoE's **attention / dense projections** (the experts
use a separate grouped kernel). Single-stream and 4-concurrent:

| path | 1 stream | 4 concurrent (per-stream / aggregate) |
|---|---:|---:|
| A.3 attention | 19.93 | — |
| **coalesced attention** | **38.14** (1.9×) | **30.4 / ~121** |

Coalesced + full residency **retires the older "FP16-attention-Linears" workaround**:
you get ~93% of that config's speed while keeping *full* FP8 residency (more KV
cache headroom), and it scales to ~121 tok/s aggregate at 4 concurrent users.

> Every transformer — dense, MoE, or hybrid — runs a dense projection backbone
> per token. This kernel pays down that universal cost, so the win is not
> model-specific.

---

## When to use it — decision matrix

| your situation | recommendation |
|---|---|
| Model **doesn't fit FP16** (16 GB cards, low TP, or too large) | **FP8-resident + coalesced** — the only path that loads, and now it's fast (~FP16-class) |
| Model fits FP16, **but you want max KV / concurrency** | **FP8-resident + coalesced** — ≈equal speed at half the weight memory → more KV cache |
| Model fits FP16 with memory to spare | **FP16** — marginally faster (coalesced is 92–97.5% of FP16) and simpler |

The coalesced kernel never makes FP8-resident *slower* — it strictly replaces the
old A.3 decode path (3–4× faster). The only question is FP8-resident vs FP16, and
that's a memory decision.

---

## Flags

| flag | default | meaning |
|---|---|---|
| `VLLM_V100_FP8_COALESCED_GEMV` | `0` (off) | master gate; set `1` to enable |
| `VLLM_V100_FP8_COALESCED_UNROLL` | `2` | K-unroll depth. **Use `4`** (the measured knee; `8` regresses on register pressure) |
| `VLLM_V100_FP8_COALESCED_GEMV_M_MAX` | `1` | max batch routed to the coalesced path. `1` = single-stream only (proven); **set `8`** for multi-user/batched decode (validated, beats A.3 at concurrency) |

**Engagement conditions** (else it falls back to A.3/A.1/A.2/WMMA, fully safe):
`M ≤ M_MAX`, `block_w == 128`, `block_h ∈ {1, 128}` (block *and* channel scale),
`K % 128 == 0`. Correctness validated `cos ≈ 1.0` vs FP32 for M=1/2/8, both scales.

Recommended production config:
```bash
VLLM_V100_FP8_COALESCED_GEMV=1
VLLM_V100_FP8_COALESCED_UNROLL=4
VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8   # 1 if you only ever serve single-stream
```

---

## How we measured (and a trap to avoid)

**Always measure steady-state per-stream decode rate** — `(tokens − 1) /
(t_last − t_first)`, excluding TTFT — and report aggregate as the *sum of
per-stream rates*. **Never use `total_tokens / wall_time`**: wall time folds in the
prefill + first-decode cudagraph-capture bubble, which divides into the rate and
**manufactures phantom "cliffs"** at low concurrency. During this work a tokens/wall
measurement produced an apparent concurrency collapse that fully evaporated under
steady-state measurement — there was no real regression, only the artifact.

Tooling (in `tools/`):
- `coalesced_gemv_e2e_ab_vllm021.sh` — 3-arm A/B (A.3 / coalesced / FP16) single-stream
- `coalesced_concurrency_sweep_vllm021.sh` — concurrency sweep, steady-state metric
- `qwen27b_fp8_gemm_microbench.{py,sh}`, `qwen27b_fp8_a3_ncu_probe.{py,sh}` — kernel microbench + NCU
- `qwen27b_fp8_coalesced_numtest.py` — correctness (cos vs FP32)

## Scope and limits

- **V100 / sm_70 only.** This is a CUDA-core kernel (Volta has no FP8 tensor cores,
  no `cp.async`). It is *not* a Marlin port — Marlin is an sm_80+ tensor-core GEMM,
  the wrong tool for M=1 bandwidth-bound decode.
- Accelerates **dense/attention Linears** (the per-token backbone). MoE experts use
  the separate grouped kernel; the GatedDeltaNet recurrence in hybrid models is its
  own path. All validated to scale at concurrency.
- Remaining headroom: the kernel is at ~29% DRAM vs cuBLAS's ~77% (it reads half the
  bytes, hence still competitive) — deeper bandwidth tuning could push it *past* FP16;
  and batched decode is sub-linear (normal). Neither is a correctness or stability gap.
