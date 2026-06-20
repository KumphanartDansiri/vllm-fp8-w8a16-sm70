# Performance Experiment v2 — design

Redesign of the V100 benchmark after the FP8 dequant cliff (2026-06-20): dense-C8
parity now needs a custom integrated FP8×FP16 matmul (tensor-core/WMMA), out of
scope here. v2 measures what actually ships, across engine and concurrency, with a
correctness battery so no number is published without a "real answer, not garbage"
check.

## Axes
- **Model fleet (6):** Qwen3.6-27B (dense), Qwen3.6-35B-A3B (MoE), Qwen3.5-122B-A10B
  (large MoE, VL), GLM-4.5-Air (MoE, compressed-tensors), Gemma-4-31B (dense, VL),
  Gemma-4-26B-A4B (MoE, VL).
- **Precision:** fp8 (our W8A16 plugin, primary) + the reference that fits
  (fp16/bf16 where VRAM allows; GPTQ-Int4 for the 122B). fp8 uses coalesced +
  vectorized dequant (the shipping kernel).
- **Engine:** **vLLM 0.19 AND 0.21**, every feasible cell on both (quantifies the
  0.21 regression). Transformers per model (below).
- **Concurrency:** C1 / C2 / C4 / C8.

## Engine × transformers × VL requirement table
| model | VL (skip-mm) | min TP | 0.19 image | 0.21 image | transformers |
|---|---|---|---|---|---|
| Qwen3.6-27B | no | 2 | vllm019-cu126 | vllm021-cu126 | stock 4.57 |
| Qwen3.6-35B-A3B | no | 4 | vllm019-cu126 | vllm021-cu126 | stock 4.57 |
| Qwen3.5-122B-A10B | **yes** | 8 | vllm019-cu126 | vllm021-cu126 | stock 4.57 |
| GLM-4.5-Air | no | 8 | vllm019-cu126 | vllm021-cu126 | stock 4.57 |
| Gemma-4-31B | **yes** | 4 | **vllm019-tf5** | vllm021-cu126 (`--max-num-batched-tokens>=2496`) | **tf-5.x** |
| Gemma-4-26B-A4B | **yes** | 4 | **vllm019-tf5** | vllm021-cu126 | **tf-5.x** |

"Whether an upgraded transformers is needed" is answered here: stock 4.57 for
Qwen/GLM on both engines; **tf-5.x required for Gemma-4** (0.19 needs the vllm019-tf5
image; 0.21 needs `--max-num-batched-tokens>=2496`). VL models get
`--skip-mm-profiling` so vision-encoder dummy profiling (~45× cold-start tax) is
excluded with zero decode impact.

## Per-cell protocol
Cell = (model, precision, engine). For each feasible cell, one serve
(cudagraph FULL_DECODE_ONLY, coalesced FP8 on for fp8 cells, skip-mm for VL):

### A. Correctness battery — 5 tests, ALL monitored for "rightness"
Greedy (temp=0). Every output is auto-gated (non-empty, >=20 tokens, repetition
ratio < 0.35, + category check below) AND captured verbatim. Each test gets a
`quality_status` ∈ {`pass`, `suspect`, `fail`}; the cell's status is the worst of
its tests. **Auto-gates are first-pass only** — a model can emit parseable JSON
that is useless or a function that is wrong — so `suspect`/`pass` near the boundary
is left for human review, and **speed rows from a `fail` cell are withheld**.
1. **Exactness** — one fixed prompt run **5×**; compare output hashes →
   `Exact` (all 5 identical) / `Stable` (coherent, varies — FP8-vs-FP16 tops out here
   by construction) / `Coherent` / `Fail`.
2. **Reasoning** — a multi-step word/logic problem (numeric/reasoning intactness).
3. **Code** — "write function X"; gate also checks it contains a definition.
4. **Long-form** — detailed multi-paragraph explanation (coherence + on-topic). **Also
   the performance prompt (B).**
5. **JSON / instruction-following** — "return exactly 3 items as JSON"; gate also
   checks parseable JSON with the right shape.
(Tests 2/3/5 categories are easy to swap in the harness.)

### B. Performance — driven entirely by Test 4 (long-form)
Same task family as Test 4; throughput fixes *generated* tokens, TTFT fixes *input*
length.
- **Decode C1/C2/C4/C8** — Test-4 prompt, `ignore_eos=true`, `max_tokens=256`,
  **topic rotated per concurrent stream**, per-user + aggregate tok/s, **median of 5 reps**.
- **TTFT** — Test-4-style prompt padded with a coherent reference document to a fixed
  *input* length: **short ~2k** and **long ~24k** tokens. `max_tokens=32-64`, `temp=0`,
  `stream=true`, **no `ignore_eos`**; measure send→first-content-token, steady-min
  (drop the first request's warmup). The **long** prompt is also run with **FA-V100 on**
  (`VLLM_V100_FLASH_ATTN=1`, `--block-size 256`) for the off/on prefill A/B — the 2.66×
  lever. FA arm is FA-eligible models only (MHA/GQA: 27B/35B/122B/GLM-Air; **not** Gemma
  head_dim-512, **not** MLA).

## Outputs
Durable `results/perf_v2_<model>_<prec>_<engine>_<stamp>/`: SUMMARY.txt, the 5
captured test outputs, and one CSV row per (cell × metric) feeding the SSOT
`data/benchmark_matrix.csv` rebuild. **Every row carries `quality_status`**
(pass/suspect/fail) + the exactness label, so "fast but garbage" can be filtered out
of the matrix.

## Cost note
6 models × ~2 precisions × 2 engines ≈ 24 serves, each with the battery + TTFT(×2+FA)
+ C1–C8(×5). This is a multi-hour fleet pass → run as a clean-box batch, infeasible
cells (OOM at min TP) auto-skipped and logged, not zeroed.
