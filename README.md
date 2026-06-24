# FP8 W8A16 on V100 for vLLM

Volta-native fallback kernels and vLLM monkey-patches for serving
DeepSeek-style block-FP8 W8A16 models on NVIDIA Tesla V100 (`sm_70`).

Upstream vLLM dropped practical V100 support; this package is the **V100 vLLM
compatibility + performance layer** that brings it back on **vLLM 0.19 and 0.21**
(source-built on CUDA 12.6). It is more than FP8 kernels: it adds the FP8 W8A16
linear path (Volta WMMA prefill + grouped/coalesced CUDA-core MoE decode), the
**FP16-MoE `sm_70` config fix**, a **FlashAttention-V100 prefill + MLA bridge**,
and the benchmark harness behind the companion **V100 vLLM in 2026** write-up
(7-model dual-engine matrix, frozen at tag `fp8-v100-2026-matrix`).

> **This is not native FP8 execution.** V100 (`sm_70`) has no FP8 tensor-core
> path. Weights are stored as block-scaled FP8 and dequantized to FP16 *inside*
> the GEMM kernel — dequantized weights never round-trip through HBM — then
> executed as FP16: WMMA tensor cores for prefill, CUDA-core GEMM for MoE
> decode. See **How it works** below.

## Why this exists

V100 was the flagship datacenter GPU of its generation. What dates it now isn't
raw compute — it's memory: 16/32 GB of capacity and ~900 GB/s of bandwidth that
modern models outgrow, plus the absence of the native low-precision paths (FP8
tensor cores, FlashAttention) newer GPUs added.

Meanwhile, FP8 has become a format model creators ship *first*. Large MoE models
in particular are released as block-scaled FP8 because that's what modern
hardware trains and serves in — so the FP8 checkpoint is often the canonical
release, and GPTQ builds are later re-quants. On V100, vLLM already runs that
re-quantized GPTQ-Int4 today; its FP8 path was the one piece that refused
`sm_70`. **This project fills that gap** — it adds the missing FP8
capability so a V100 can consume those releases directly, instead of waiting for
someone to re-quantize them.

The catch: V100 has no native FP8, so weights must be dequantized in software,
on **CUDA cores** (there is no FP8/tensor-core path). The original verdict was
"yes for sparse MoE, no for dense." A branchless E4M3 dequant rewrite overturned
half of that: because CUDA-core dequant is bandwidth-favorable at low batch,
**dense FP8 now beats FP16 at 1–2 users, is on par at 4, and loses only at 8**
(the compute-bound regime that needs a tensor-core kernel we haven't built — no
timeline). **Sparse MoE wins at every concurrency.** The rest of this README is
the honest accounting of where it lands.

## Status

| Tier | What |
|---|---|
| **Known good** | **vLLM 0.19 + 0.21, source-built on CUDA 12.6** (0.21 = newest models; 0.19 = faster decode; both run the plugin), torch 2.10–2.11, Python 3.12, NVIDIA V100 (`sm_70`); 7 model families — dense + MoE + MLA — across FP8 / FP16 / GPTQ-Int4. The frozen matrix is tag `fp8-v100-2026-matrix`. |
| **Validated** | dense-FP8 (coalesced + branchless dequant → *faster than FP16* at low concurrency); FlashAttention-V100 prefill + MLA bridge (GLM-4.7-Flash on both engines). |
| **Optional** | MTP speculative decoding (`ENABLE_QWEN_MTP=1`, default OFF). |
| **In flight** | warm/chunked-prefill TTFT columns in the matrix; a tensor-core (WMMA) decode kernel to close the dense C8 gap (no timeline). |
| **Unsupported here** | CUDA 13; native FP8 hardware compute (V100 has none); non-Volta GPUs (A100/H100/etc. — use upstream vLLM, which has native FP8). |

## Is this for me?

**Use this if** you have V100s (`sm_70`) and want to serve **sparse MoE**
block-FP8 checkpoints (DeepSeek-style W8A16) that upstream vLLM refuses to run
on this hardware.

**Also useful for dense** models at low concurrency: with the branchless-dequant
rewrite, dense FP8 now **beats FP16 at 1–2 users and ties at 4** — though FP16 /
GPTQ-Int4 still win at 8 users (the CUDA-core dequant is compute-bound there).

**Do not use this if** you have H100/A100 or newer (use upstream vLLM with
native FP8) or want generic FP8 tensor-core acceleration (V100 has none). See
**Performance notes** below for the per-concurrency numbers.

## How it works

**Mechanism.** Weights are stored in block-scaled FP8 (DeepSeek-style E4M3),
activations stay FP16 (W8A16). Custom Volta kernels apply the per-block scale
and dequantize FP8→FP16 *inside* the kernel, immediately before the FP16
computation, so dequantized weights never round-trip through HBM. The two paths
differ in how: the **prefill** path stages the dequantized FP16 weight tiles in
shared memory (double-buffered) and feeds them to an FP16 WMMA tensor-core
matmul; the **MoE decode** path dequantizes inline inside a CUDA-core grouped
GEMM.

**Where this sits.** This is a **fused dequant→compute** design. FP8 weights are
dequantized to FP16 *inside* the compute kernel, immediately before the FP16
math; dequantized weights are never materialized as an FP16 matrix in HBM, and
are not handed to a separate GEMM launch.

What it is **not** is a natively-integrated quantized kernel like Marlin or
GPTQ-Marlin, where the quantized format itself is woven into the tensor-core
pipeline — format-aware packing, async-pipelined loads, dequantization
interleaved directly with MMA execution. We are far from that level of
integration, and on Volta (`sm_70`) there is no `cp.async` and no native
FP8/INT4 tensor-core support to integrate against — so dequant-feeds-FP16 is
the pragmatic structure here, not the optimal one.

The payoff is enabling block-FP8 MoE on hardware upstream dropped, where
per-token sparsity (~3B of weights active) makes "read fewer bytes" matter more
than peak GEMM efficiency.

**The landscape.** Quantized inference kernels generally fall into three
categories:

1. **Dequantize to an FP16 matrix, then launch a stock GEMM** (e.g. FP8 → FP16
   buffer → cuBLAS). *Not this project.*
2. **Dequantization fused inside the compute kernel**, immediately before the
   FP16 math — dequantized weights are never materialized as an FP16 matrix in
   HBM. **← this project.**
3. **Quantized format integrated directly into the MMA / tensor-core pipeline**
   (e.g. Marlin, GPTQ-Marlin). *Not this project.*

If you arrived assuming Marlin/GPTQ-Marlin (category 3), or assuming "FP8" means
Hopper FP8 tensor cores, or just "half the memory" — this is category 2 on
hardware that has neither native FP8 nor the pipeline features category 3 needs.

## Current Baseline

The package version is `0.5.0`. The two serving stacks are **vLLM 0.19 and 0.21,
both source-built on CUDA 12.6** (see [vLLM 0.19 compatibility](#vllm-019-compatibility)
and [vLLM 0.21 sm_70 base](#vllm-021-sm_70-base)):

| Component | Baseline |
|---|---|
| Python | 3.12 (**required for cudagraph** — see below) |
| vLLM | 0.19.x and 0.21.x (source-built; **not** pip wheels) |
| torch / CUDA | 2.11/cu126 (0.21) · 2.10/cu128 (0.19-tf5); CUDA 12.x — never CUDA 13 on V100 |
| Docker images | `vllm-v100:vllm021-cu126`, `vllm-v100-py312:vllm019-cu126` (+ a transformers-5 / cu128 variant for Gemma-4 / GLM-4.7) |
| Launcher | `python3 -m fp8_w8a16_sm70.vllm_serve …` (mount the package on `PYTHONPATH`; kernels JIT-cache) |

> **Python 3.10 breaks cudagraph.** On Python ≤3.10 the cudagraph path hits a
> FakeTensorMode bug in vLLM 0.18, so the legacy 3.10 stack is forced into
> `--enforce-eager` — a correctness fallback that runs **~6.8–8× slower** (see
> [Performance notes](#performance-notes)). Use Python 3.12 for the baseline
> numbers; treat 3.10 as a debugging/correctness fallback only.

## vLLM 0.19 compatibility

vLLM 0.19 is a **primary serving baseline on V100** (alongside 0.21), first
validated for the Qwen 3.5 / 3.6 family at parity with the original 0.18 build.
One important build difference (shared with 0.21):

**You must build vLLM 0.19 from source for `sm_70`.** Unlike 0.18, the official
0.19 PyPI wheel is compiled without `7.0` in its CUDA arch list (the release
build uses `8.7 8.9 9.0 10.0+PTX 12.0`), so `pip install vllm==0.19.0` dies on
V100 with *"no kernel image is available for execution on the device."* The
vLLM **source** still supports `7.0` (`CMakeLists.txt` keeps it), so we compile
from a local checkout with `TORCH_CUDA_ARCH_LIST=7.0`. Same Python 3.12 + torch
2.10.0+cu128 ABI as the 0.18 image, so the monkey-patches and the JIT-compiled
`fp8_dequant.cu` stay binary-compatible.

| Component | 0.19 baseline |
|---|---|
| Dockerfile | `docker/Dockerfile.vllm019_py312` (source build, `sm_70` only) |
| Launcher | `docker/run_docker_vllm019_py312.sh` (`build` / `shell` / `serve` / `serve-fp8`) |
| Smoke driver | `tools/smoke_vllm019.sh` (clean-box-guarded load matrix → `/tmp/v100_smoke019/`) |
| Image tag | `vllm-v100-py312:vllm019` |

**Validation (correctness, `--enforce-eager`): 6/6 Qwen models load and generate
coherent text.** The FP8 W8A16 monkey-patches port to 0.19 with no code changes
— every patch target is signature-identical, the patches engage in every TP
worker (`volta=True`, `min_cap=70`, MoE fallback on), and no fallback fault
(`get_fused_moe_quant_config` / `NotImplementedError`) appears in any log.

| Model | Mode | TP | Result |
|---|---|---|---|
| Qwen3.6-35B-A3B-FP8 | `serve-fp8` | 4 | PASS |
| Qwen3.6-35B-A3B (FP16) | `serve` | 4 | PASS |
| Qwen3.6-27B-FP8 (dense) | `serve-fp8` | 4 | PASS |
| Qwen3.6-27B (FP16) | `serve` | 4 | PASS |
| Qwen3.5-122B-A10B-FP8 | `serve-fp8` | 8 | PASS |
| Qwen3.5-122B-A10B-GPTQ-Int4 | `serve` | 8 | PASS |

**Performance (cudagraph, `mode=0`+`FULL_DECODE_ONLY`, ns=8): parity with the
0.18 baseline, MTP included.** Measured on Qwen3.6-35B-A3B-FP8 (FP16 activations,
TP=4, steady-state):

| Config | vLLM 0.19 tok/s | 0.18 baseline |
|---|---|---|
| cudagraph | **52.27** | 52.4 |
| cudagraph + MTP (`ENABLE_QWEN_MTP=1`) | **62.54** (1.20×) | — |

So the FP8 W8A16 + cudagraph + MTP stack ports to 0.19 with no regression.

Build, verify, and measure:

```bash
./tools/smoke_vllm019.sh build              # one-time source build (~30–90 min)
./tools/smoke_vllm019.sh qwen               # full Qwen load-matrix → /tmp/v100_smoke019/
./tools/measure_qwen35b_vllm019.sh          # cudagraph tok/s
MTP=1 ./tools/measure_qwen35b_vllm019.sh    # cudagraph + MTP
```

## vLLM 0.21 sm_70 base

vLLM 0.21.0 is **validated as a stock source-build base on V100** for FP16 and
GPTQ-Int4 workloads. This does not yet mean the FP8 W8A16 wrapper is ported;
it means the new engine base, cudagraph path, and model registry are viable on
`sm_70`.

The current 0.21 lane uses a CUDA 12.6 / torch 2.11.0+cu126 stack:

| Component | 0.21 stock base |
|---|---|
| Dockerfile | `docker/Dockerfile.vllm021_cu126` (source build, `sm_70`) |
| Smoke drivers | `tools/smoke_vllm021.sh`, `tools/smoke_vllm021_modes.sh` |
| Image tag | `vllm-v100:vllm021-cu126` |
| Artifacts | `/tmp/v100_smoke021/` |

Why cu126: vLLM 0.21 pins torch 2.11.0, and the cu128 torch 2.11 wheel no
longer includes Volta support. The cu126 torch 2.11 wheel keeps `sm_70` in
`torch.cuda.get_arch_list()`.

Validation summary, stock vLLM 0.21.0 with no FP8 patches:

| Model | eager tok/s | cudagraph tok/s | cudagraph + MTP |
|---|---:|---:|---|
| Qwen3.6-27B FP16 dense GDN | 6.81 | 36.28 | 42.32, exact-match/lossless, 82.6% accept |
| Qwen3.6-35B-A3B FP16 MoE | 6.61 | 15.34 | coherent but exact-diff, 86.0% accept |
| gemma-4-31B-it FP16 dense | 8.11 | 28.50 | no MTP head |
| Qwen3.5-122B-A10B GPTQ-Int4 TP8 | 5.31 | 52.01 | worker-init crash |

Interpretation:

- Gate 4 passed: vLLM 0.21.0 runs on V100 in eager and cudagraph across dense,
  MoE, GPTQ-Int4, and Gemma 4 workloads.
- MTP is validated on the dense 27B canary. The 35B-A3B MTP exact-diff appears
  coherent and likely reflects MoE+FP16 nondeterminism, but exactness remains a
  useful guardrail.
- 122B-Int4 MTP is a separate GPTQ-on-Volta issue and is not on the FP8 path.
- The FP8 insertion points were source-checked and appear intact in 0.21:
  `Fp8Config.get_min_capability`, `Fp8LinearMethod` init/PWAL/apply, and
  `Fp8MoEMethod` / `Fp8OnlineMoEMethod` init/PWAL/apply.

## Build

```bash
./docker/run_docker_vllm018_py312.sh build
```

The image uses `nvidia/cuda:12.8.1-devel-ubuntu24.04` so `nvcc` is available
for the first-run JIT build of `src/fp8_w8a16_sm70/fp8_dequant.cu`.

## Serve FP8

```bash
PORT=8002 GPUS=all ./docker/run_docker_vllm018_py312.sh serve-fp8 \
    --model /mnt/models/Qwen3.5-122B-A10B-FP8 \
    --served-model-name qwen-v100 \
    --quantization fp8 \
    --dtype float16 \
    --attention-backend TRITON_ATTN \
    --tensor-parallel-size 8 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 32768 \
    --no-enable-chunked-prefill \
    --disable-custom-all-reduce \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 0.0.0.0 \
    --port 8002 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

The launcher defaults the v0.4.0 MoE knobs:

- `VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1`
- `VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=128`
- `VLLM_V100_FP8_MOE_GROUPED_K_SPLIT=auto`
- `VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1`

> **Use `--max-num-seqs 8`, not `1`.** On these hybrid (attention + GDN/mamba)
> models, `--max-num-seqs 1` under cudagraph crashes at init — an upstream vLLM
> 0.18 issue, not this package. See **Known limitations** below.

## Serve a plain FP16 / GPTQ model (no wrapper)

FP16/BF16 and GPTQ-Int4 models do **not** need this package — they run on
stock vLLM 0.18. Use the `serve` subcommand (plain `vllm serve`, no
monkey-patches):

```bash
PORT=8002 GPUS=all ./docker/run_docker_vllm018_py312.sh serve \
    --model /mnt/models/<your-fp16-or-gptq-model> \
    --dtype float16 \
    --attention-backend TRITON_ATTN \
    --tensor-parallel-size 4 \
    --no-enable-chunked-prefill \
    --disable-custom-all-reduce \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 0.0.0.0 --port 8002
```

Note the differences from the FP8 launch:

- **`serve`, not `serve-fp8`** — no wrapper is loaded.
- **No `--quantization fp8`** — let vLLM read the checkpoint's own config
  (GPTQ-Int4 is auto-detected; FP16 needs no quant flag).
- **Use `cudagraph_mode: FULL_DECODE_ONLY`** (same as the FP8 path). Do **not**
  use `FULL_AND_PIECEWISE` with `mode:0` — vLLM treats that combination as
  incompatible and silently overrides cudagraph to `NONE`, so you get
  eager speed (~7 tok/s on 27B instead of ~40). Piecewise graphs would require
  a non-zero compilation mode, outside this project's tested envelope.

The sm_70 viability flags (TRITON_ATTN, no chunked prefill, disabled custom
all-reduce, cudagraph) still apply — those are V100 constraints, not FP8 ones.

**Do not run FP16 through `serve-fp8`.** The monkey-patches only intercept the
FP8 linear path, so on a non-FP8 checkpoint they are inert — you gain nothing
and only add the FP8 capability-gate bypass you don't need. Rule: `serve` for
FP16/GPTQ, `serve-fp8` for block-FP8.

## Known Deployment Rule

| Workload | Preferred path |
|---|---|
| Qwen3.5/Qwen3.6 MoE+GDN FP8 | This package on the py3.12 cudagraph baseline |
| Dense+GDN 27B-class | GPTQ-Int4 if available; otherwise FP16 |
| Small dense | FP16 or Int4 |

Dense FP8 on V100 is currently slower than FP16 and GPTQ-Int4. See
**Performance notes** below for the numbers and why.

## Performance notes

*These effects come from the vLLM/V100 serving path and from model
architecture — not from FP8 accuracy. They explain the headline numbers, and
where this path does **not** help.*

**cudagraph is mandatory for the headline numbers.** `mode=0 +
FULL_DECODE_ONLY` (in the serve command above) is the performance path.
`--enforce-eager` is a correctness fallback only and runs **~6.8–8× slower**
(122B-A10B-FP8: 5.09 → 34.76 tok/s; 35B-A3B-FP8: 6.57 → 52.87 tok/s).
Requires Python 3.12 — the 3.10 path hits a FakeTensorMode cudagraph bug.

**Decode is communication-bound on V100.** With `TRITON_ATTN` (no
FlashAttention on `sm_70`) and custom all-reduce disabled, cross-GPU
all-reduce is a large fraction of per-token time at TP=8. That is the V100
tax, and the reason MTP is the highest-leverage optional knob.

**MTP (optional) amortizes that cost.** Speculative decoding spreads per-token
overhead — notably all-reduce — across multiple tokens, so decode-heavy
traffic gains most (up to 1.36×) and long-prompt/short-output can regress.

**FP8 decode runs on CUDA cores — so it's a low-concurrency win, by design.**
V100 has no FP8 (nor any) tensor-core path for this format, so the fused
dequant→FP16 GEMV executes on **CUDA cores**. That's bandwidth-favorable at low
batch and compute-bound at high batch — which sets a clean, honest concurrency
pattern that splits by model type:

- **Dense models:** FP8 **wins at C1/C2**, is **roughly parity at C4**, and
  **falls behind FP16 at C8** — because FP16 decode runs through cuBLAS, which
  uses tensor cores more effectively at batch, while our FP8 path is CUDA-core
  dequant→matmul (Qwen3.6-27B TP4, 0.21: FP8 46 vs FP16 35 at C1; 28 vs 28 at C4;
  20 vs 27 at C8). Closing the dense C8 gap likely requires a proper **tensor-core
  / WMMA decode kernel that consumes FP8 weights and scales directly** — future
  work, no timeline. (Two rewrites got dense this far: a coalesced GEMV and a
  **branchless E4M3 dequant** — the *converter*, not `sm_70`, was the real
  limiter — together flipping dense FP8 from ~0.31× FP16 to faster-than-FP16 at
  low concurrency.)
- **Sparse MoE — the dense limit does not invalidate this.** Sparse activation
  moves only ~3B params/token, so FP8's reduced weight traffic stays valuable
  **across all concurrency** (Qwen3.6-35B-A3B TP4, 0.21: FP8 75 / 48 vs FP16
  56 / 21 tok/s/user at C1 / C8). FP8 wins broadly here — and for the largest
  models (122B, GLM) it's **often the only format that fits.** This is the
  flagship V100 case.

Full per-engine (0.19 / 0.21), per-concurrency numbers are the frozen matrix
(tag `fp8-v100-2026-matrix`), rendered in the companion **V100 vLLM in 2026**
write-up. Decision matrix + flags: **[`docs/COALESCED_FP8_GEMV.md`](docs/COALESCED_FP8_GEMV.md)**.

## Known limitations

**`--max-num-seqs 1` crashes on hybrid (attention + GDN/mamba) models under
cudagraph — this is upstream vLLM 0.18, not this package.** The Qwen3.5/3.6-A\*B
checkpoints (and the 27B) are hybrid models. At `--max-num-seqs 1`, vLLM's
cudagraph memory profiler allocates a minimal 2-block KV cache, and its
attention/mamba layout check can't disambiguate the resulting `[2, 2, …]`
tensor (`assert shape[1] != 2`) — so the engine aborts at init. This reproduces
on **stock `vllm serve` with an FP16 checkpoint**, so neither the wrapper nor
FP8 is involved.

- **Supported config: `--max-num-seqs 8`** — more cudagraph capture blocks, so
  the profiling cache is wider than 2 and the ambiguity never arises. It's also
  faster than `ns=1`.
- Non-hybrid / dense-attention models are unaffected at `ns=1`.
- If you specifically need `ns=1` (e.g. lowest-latency streaming), run
  `--enforce-eager` (no cudagraph) — correct, but ~7× slower.

## Optional: MTP Speculative Decoding (v0.4.1)

Stock vLLM 0.18.0 ships `qwen3_5_mtp.py` and registers `Qwen3_5MoeMTP` as a
spec-decode architecture. The production Qwen3.5/3.6-A*B-FP8 checkpoints
ship MTP head weights baked in (`mtp_num_hidden_layers=1`, ~1.5k `mtp.*`
tensors). v0.4.1 exposes this as an opt-in launcher knob; the underlying
code path is upstream.

Enable with `ENABLE_QWEN_MTP=1`:

```bash
PORT=8002 GPUS=all ENABLE_QWEN_MTP=1 \
  ./docker/run_docker_vllm018_py312.sh serve-fp8 \
    --model /mnt/models/Qwen3.5-122B-A10B-FP8 \
    <...the rest of the Serve FP8 args above...>
```

*(Abbreviated — copy the complete argument list from **Serve FP8** above; the
only change is adding the `ENABLE_QWEN_MTP=1` prefix.)*

The launcher appends:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Default is OFF. Everything else in the v0.4.0 launch is unchanged.

### What v0.4.1 claims (measured)

- Adds optional Qwen3.5 MTP speculative decoding support.
- Improves decode-heavy workloads in measured tests.
  - 122B-A10B-FP8 TP=8: `34.76 → 47.32 tok/s` (1.36×), from the dedicated
    bench (steady-state decode, prod-config launch).
  - 35B-A3B-FP8 TP=4: `52.87 → 65.14 tok/s` (1.23×).
  - 27B-Dense-FP16 TP=4: `1.09×` (smaller because Dense baseline already
    well-amortized).
- MoE/FP8 outputs are not guaranteed bit-identical to baseline.
- Observed divergences were quality-equivalent in the validation suite
  (inspected by hand; alternate phrasings, same-distribution continuations).

### What v0.4.1 does NOT claim

- Bit-exactness on MoE or FP8.
- Universal speedup.
- Long-prompt-short-output safety as a general guarantee.
- Default-on validation.
- Acceptance rate alone as proof of correctness.

### Workload guidance

- **Decode-heavy traffic** (chat, code generation, reasoning, sustained
  generation): MTP wins on every model tested. Recommended.
- **Long-prompt-short-output traffic** (summarization, retrieval QA):
  - On 35B-A3B-FP8 we observed a slight regression (~0.94×–0.97×).
  - On 122B-A10B-FP8 we did NOT observe that regression in the test suite.
  - **Operators should benchmark their workload before enabling globally.**
- **Streaming and tool-calling**: not formally validated in v0.4.1; treat
  as untested. v0.4.2 will cover these.

### Validation matrix (exactness)

| Model | Baseline self-stable | MTP within baseline noise envelope |
|---|:---:|:---:|
| Qwen3.6-27B Dense FP16 | (assumed) | ✓ bit-exact (11/11) |
| Qwen3.6-35B-A3B FP16 | ✓ 11/11 | MTP changes 5/11 outputs |
| Qwen3.6-35B-A3B-FP8 | 8/11 | ✓ MTP 7/11 vs baseline (within noise) |
| Qwen3.5-122B-A10B-FP8 (prod target) | 6/11 | ✓ MTP 7/11 vs baseline (within noise) |

Method: 11-prompt suite, temperature=0, token-string list equality. Baseline
self-test (same serve called twice) distinguishes MTP-introduced divergence
from intrinsic FP8 path nondeterminism. Full data in
`/tmp/v100_bench/exactness_*.json`; see `docs/SESSION_LOG.md` Stage 4 for
the investigation arc.

### Path to default-on (v0.4.2)

`ENABLE_QWEN_MTP=1` will become default after:

1. Multi-sample self-tests at higher N to estimate baseline noise rate
   statistically (current data is single-sample).
2. More prompts, including production chat/code/tool-use traffic.
3. Streaming and chat-completions endpoint sanity checks.
4. Per-workload routing decision (per-request enable vs two-service split).
5. Long-prompt-short-gen confirmation at higher N on the production target.

## Benchmarking

Measured on the **Serve FP8** cudagraph launch above (py3.12, TRITON_ATTN,
`mode=0 + FULL_DECODE_ONLY`, the v0.4.0 MoE knobs) at the supported
`--max-num-seqs 8`, single-stream, temperature=0, steady-state decode (5 timed
requests, `ignore_eos`, 200 tokens each):

| Model | Hardware | tok/s | Notes |
|---|---|---|---|
| Qwen3.6-35B-A3B-FP8 | 4× V100, TP=4 | **52.4** | |
| Qwen3.5-122B-A10B-FP8 | 8× V100, TP=8 | **34.6** | |
| Qwen3.5-122B-A10B-FP8 | 8× V100, TP=8 | **45–47** | + `ENABLE_QWEN_MTP=1`; workload-dependent (1.3–1.37×), see MTP section |

**Reproduce with `tools/bench_v100.sh`** — it launches that exact cudagraph
`serve-fp8` config by default and captures the resolved config, git commit,
serve console, and a timed curl loop into one timestamped directory under
`/tmp/v100_bench/`:

```bash
# Terminal 1: launch the cudagraph serve-fp8 baseline
./tools/bench_v100.sh serve prod
#   ENABLE_QWEN_MTP=1 ./tools/bench_v100.sh serve prod   # for the MTP row
#   ENFORCE_EAGER=1   ./tools/bench_v100.sh serve prod   # legacy eager fallback

# Terminal 2: 1 warmup + N timed requests against the running serve
./tools/bench_v100.sh curls prod 8
```

Or measure manually: launch the **Serve FP8** command and drive it with any
OpenAI-compatible client (1 warmup, then N timed requests at temperature=0,
completion tokens ÷ elapsed). Either way, a tok/s figure is only meaningful
next to the launch config that produced it — always report the serve command
with it.

## AI-Assisted Remote Setup

This project runs well from a remote V100 server over SSH with an agentic
coding assistant in the repo. The setup has many small environment details —
CUDA version, Docker GPU access, mounted model paths, vLLM flags, JIT caches —
and an agent can check them directly on the machine instead of forcing you to
copy terminal output back and forth.

Recommended workflow:

1. SSH to the server with your normal remote editor or terminal.
2. Clone this repo on the server.
3. Start your coding agent from the repo root.
4. Give it the prompt below, edited for your model path and GPU count.
5. Review commands before approving anything that installs packages, starts a
   long serve, or pushes to a remote.

Suggested first prompt:

```text
I am setting up fp8-w8a16-sm70 on a remote V100 server.

Please inspect the repo first, then verify:
- nvidia-smi shows V100 GPUs
- Docker can see the GPUs
- /mnt/models contains my FP8 model
- CUDA is 12.x, not CUDA 13
- the README serve command matches this host

Build the vLLM 0.18 Python 3.12 Docker image if needed:
./docker/run_docker_vllm018_py312.sh build

Then help me launch this model with serve-fp8:
/mnt/models/Qwen3.5-122B-A10B-FP8

Use port 8002, tensor parallel size 8, TRITON_ATTN, no chunked prefill,
disable custom all-reduce, and quantization fp8.

Do not modify files outside this repo. Do not delete models or caches.
Explain each command before running anything destructive or long-running.
```

Useful things to ask the agent after launch:

- "Check the logs and tell me whether the FP8 monkey-patch loaded."
- "Confirm the CUDA extension compiled and is now cached."
- "Send one OpenAI-compatible test request to the local server."
- "Summarize throughput and any warnings from the serve log."
- "Turn this exact launch into a systemd service or Docker runbook."

Keep secrets out of prompts and repo files. API keys, private model tokens,
SSH keys, and internal hostnames should stay in your shell environment,
secret manager, or deployment system.

## Profiling & diagnostics

The package ships extensive opt-in instrumentation — per-section decode timing
breakdown, MoE GEMM profiling, and cross-cutting all-reduce attribution — so
you can see where decode time actually goes and decide what's worth
optimizing. **All of it is OFF by default and costs nothing in normal
serving.**

Two things to know before enabling it:

- Treat the profiling hooks as **eager-only**. Some capture-sensitive paths
  are guarded, but CUDA timing events do not produce useful decode attribution
  under cudagraph capture; add `--enforce-eager` when profiling.
- Eager is ~7× slower, so profiling numbers are for **relative attribution**
  ("the all-reduce is 90% of `out_proj`"), never headline throughput.

Quick start:

```bash
VLLM_V100_FP8_DECODE_BREAKDOWN=1 ... serve-fp8 ... --enforce-eager
```

Full knob reference, output guide, and a worked "find the bottleneck" workflow
are in [`docs/PROFILING.md`](docs/PROFILING.md).

## Why monkey-patches, not a vLLM fork

This package patches vLLM at runtime — replacing the FP8 linear method and a
few MoE/attention paths on import — rather than forking and editing vLLM's
source. That's deliberate:

- **Stock vLLM stays unmodified.** You install the official `vllm==0.18.0`
  wheel; the patches apply when you run `python -m fp8_w8a16_sm70.vllm_serve`.
  There is no fork to rebase against upstream and no patched vLLM to rebuild.
- **The delta from upstream is explicit and small.** Everything we change
  lives in `src/fp8_w8a16_sm70/`, so it's easy to audit exactly what differs
  from stock vLLM.
- **Trivially removable.** Drop the wrapper and you're back to stock vLLM —
  that is literally the `serve` subcommand. The FP16/GPTQ paths are unmodified
  upstream code.

Honest limitation: the patches propagate to TP workers via vLLM 0.18's worker
re-exec, which is verified here but not guaranteed robust across other vLLM
launch styles (Ray executor, forkserver). If a future vLLM breaks that, the
migration path is a vLLM plugin or `sitecustomize.py` — not a fork. This is
part of why the project is pinned to 0.18 (see `REQUIREMENTS.md`).

## Development

Useful entry points:

- `python -m fp8_w8a16_sm70.vllm_serve` applies the vLLM monkey-patches.
- `src/fp8_w8a16_sm70/fp8_dequant.cu` contains the CUDA kernels.
- `src/fp8_w8a16_sm70/module.py` contains `FP8W8A16Linear`.
- `src/fp8_w8a16_sm70/ext_loader.py` centralizes JIT compilation.

Start every new session by reading `docs/SESSION_LOG.md`; it is the project
source of truth.

## Acknowledgements

The FlashAttention-V100 prefill and MLA-prefill bridges (`VLLM_V100_FLASH_ATTN=1`)
wrap the [**ai-bond/flash-attention-v100**](https://github.com/ai-bond/flash-attention-v100)
kernel by **D. Skryabin** ([@ai_bond007](https://t.me/ai_bond007)) — a from-scratch
Volta (`sm_70`) reimplementation of FlashAttention / FlashAttention-2, released
under the BSD 3-Clause License. This project does **not** modify that kernel: it
loads the compiled `flash_attn_v100_cuda.so` and routes vLLM's prefill through it
(`src/fp8_w8a16_sm70/fa_v100_prefill.py` for MHA/GQA,
`fa_v100_mla_prefill.py` for MLA). The underlying FlashAttention algorithm is the
work of Tri Dao et al.

The FP8 W8A16 kernels (Volta WMMA prefill + grouped/coalesced CUDA-core MoE
decode), the FP16-MoE `sm_70` config fix, and the benchmark harness are original
to this project.
