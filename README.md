# FP8 W8A16 on V100 for vLLM

Volta-native fallback kernels and vLLM monkey-patches for serving
DeepSeek-style block-FP8 W8A16 models on NVIDIA Tesla V100 (`sm_70`).

Upstream vLLM dropped practical V100 support; this package brings it back on
**vLLM 0.19 and 0.21** (source-built on CUDA 12.6). Beyond the FP8 W8A16 linear
path (Volta WMMA prefill + grouped/coalesced CUDA-core MoE decode) it bundles a
couple of `sm_70` enablers that ride along in the same plugin: the **FP16-MoE
config fix** and a **FlashAttention-V100 prefill + MLA bridge**.

The *story* — why V100 still works in 2026, the full per-engine benchmark
matrix, the methodology, and deployment field-notes — lives in the companion
**V100 vLLM in 2026** write-up (the `v100-vllm-2026` repo). **This README is the
artifact**: what the plugin is, how it works, and how to build, run, and extend
it.

> **This is not native FP8 execution.** V100 (`sm_70`) has no FP8 tensor-core
> path. Weights are stored as block-scaled FP8 and dequantized to FP16 *inside*
> the GEMM kernel — dequantized weights never round-trip through HBM — then
> executed as FP16: WMMA tensor cores for prefill, CUDA-core GEMM for MoE
> decode. See **How it works** below.

## Is this for me?

**Use this if** you have V100s (`sm_70`) and want to serve **sparse MoE**
block-FP8 checkpoints (DeepSeek-style W8A16) that upstream vLLM refuses to run
on this hardware.

**Also useful for dense** models at low concurrency: with the branchless-dequant
rewrite, dense FP8 now **beats FP16 at 1–2 users and ties at 4** — though FP16 /
GPTQ-Int4 still win at 8 users (the CUDA-core dequant is compute-bound there).

**Do not use this if** you have H100/A100 or newer (use upstream vLLM with
native FP8) or want generic FP8 tensor-core acceleration (V100 has none). See
**Performance** below for the per-concurrency pattern.

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
interleaved directly with MMA execution. On Volta (`sm_70`) there is no
`cp.async` and no native FP8/INT4 tensor-core support to integrate against — so
dequant-feeds-FP16 is the pragmatic structure here, not the optimal one. The
payoff is enabling block-FP8 MoE on hardware upstream dropped, where per-token
sparsity (~3B of weights active) makes "read fewer bytes" matter more than peak
GEMM efficiency.

If you arrived assuming Marlin/GPTQ-Marlin, or assuming "FP8" means Hopper FP8
tensor cores, or just "half the memory" — this is fused-dequant on hardware that
has neither native FP8 nor the pipeline features a format-integrated kernel
needs.

## Status

| Tier | What |
|---|---|
| **Known good** | **vLLM 0.19 + 0.21, source-built on CUDA 12.6** (0.21 = newest models; 0.19 = faster decode; both run the FP8 plugin), torch 2.10–2.11, Python 3.12, NVIDIA V100 (`sm_70`); 7 model families — dense + MoE + MLA — across FP8 / FP16 / GPTQ-Int4. The frozen matrix is tag `fp8-v100-2026-matrix`. |
| **Validated** | dense-FP8 (coalesced + branchless dequant → *faster than FP16* at low concurrency); FlashAttention-V100 prefill + MLA bridge (GLM-4.7-Flash on both engines). |
| **Optional** | MTP speculative decoding (`ENABLE_QWEN_MTP=1`, default OFF). |
| **In flight** | warm/chunked-prefill TTFT columns in the matrix; a tensor-core (WMMA) decode kernel to close the dense C8 gap (no timeline). |
| **Unsupported here** | CUDA 13; native FP8 hardware compute (V100 has none); non-Volta GPUs (A100/H100/etc. — use upstream vLLM, which has native FP8). |

## Supported engines

Two source-built serving stacks, **both running the FP8 W8A16 plugin** on
`sm_70`. The package version is `0.6.0`.

| Engine | Image | Dockerfile | Use |
|---|---|---|---|
| **vLLM 0.19** | `vllm-v100-py312:vllm019` | `docker/Dockerfile.vllm019_py312` | faster decode |
| **vLLM 0.21** | `vllm-v100:vllm021-cu126` | `docker/Dockerfile.vllm021_cu126` | newest models (Gemma-4, GLM-4.7) |

| Component | Baseline |
|---|---|
| Python | 3.12 (**required for cudagraph** — Python ≤3.10 hits a FakeTensorMode bug and is forced into `--enforce-eager`, ~7× slower) |
| torch / CUDA | 2.11/cu126 (0.21) · 2.10/cu128 (0.19); **CUDA 12.x — never CUDA 13 on V100** |
| Launcher | `python3 -m fp8_w8a16_sm70.vllm_serve …` (mount the package on `PYTHONPATH`; kernels JIT-cache on first run) |

**Build from source for `sm_70`.** The official 0.19/0.21 PyPI wheels are
compiled without arch `7.0`, so `pip install` dies on V100 with *"no kernel
image is available for execution on the device."* The vLLM **source** still
keeps `7.0` in `CMakeLists.txt`, so we compile from a local checkout with
`TORCH_CUDA_ARCH_LIST=7.0`. The FP8 monkey-patches and the JIT-compiled
`fp8_dequant.cu` are signature-compatible across both engines (ported with no
code changes).

> vLLM **0.18** was the original FP8 baseline and still runs
> (`docker/run_docker_vllm018_py312.sh`), but it is **legacy** — 0.19/0.21 are
> the supported lanes. Full per-engine validation tables and the 0.18→0.21
> history are in the **V100 vLLM in 2026** write-up.

## Build

```bash
./docker/run_docker_vllm019_py312.sh build      # one-time source build (~30–90 min)
```

The image carries `nvcc` for the first-run JIT build of
`src/fp8_w8a16_sm70/fp8_dequant.cu`. (For the 0.21 lane, build
`docker/Dockerfile.vllm021_cu126` instead.)

## Serve FP8

```bash
PORT=8002 GPUS=all ./docker/run_docker_vllm019_py312.sh serve-fp8 \
    --model /mnt/models/Qwen3.5-122B-A10B-FP8 \
    --served-model-name qwen-v100 \
    --quantization fp8 \
    --dtype float16 \
    --attention-backend TRITON_ATTN \
    --tensor-parallel-size 8 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 32768 \
    --disable-custom-all-reduce \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 0.0.0.0 \
    --port 8002 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

The launcher defaults the MoE knobs (`VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1`,
`…_MAX_ROUTE_SLOTS=128`, `…_K_SPLIT=auto`, `…_FAST_ROUTE_PREP=1`). For the 0.21
lane, run `python3 -m fp8_w8a16_sm70.vllm_serve <same args>` inside the
`vllm-v100:vllm021-cu126` image.

> **Use `--max-num-seqs 8`, not `1`.** On hybrid (attention + GDN/mamba) models,
> `--max-num-seqs 1` under cudagraph crashes at init — an upstream vLLM issue,
> not this package. See **Known limitations**.
>
> **Leave chunked prefill ON** (the default). Disabling it (`--no-enable-chunked-prefill`)
> is a known crash-causer on some large hybrid configs; if a specific benchmark
> needs it for comparability, set it there, not in your first-run config.

## Serve a plain FP16 / GPTQ model (no wrapper)

FP16/BF16 and GPTQ-Int4 checkpoints don't need this package — they run on stock
vLLM. Use the `serve` subcommand (plain `vllm serve`, no monkey-patches), and
let vLLM read the checkpoint's own quant config (no `--quantization fp8`):

```bash
PORT=8002 GPUS=all ./docker/run_docker_vllm019_py312.sh serve \
    --model /mnt/models/<your-fp16-or-gptq-model> \
    --dtype float16 --attention-backend TRITON_ATTN \
    --tensor-parallel-size 4 --disable-custom-all-reduce \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 0.0.0.0 --port 8002
```

- **`serve`, not `serve-fp8`** — no wrapper loaded. Running FP16 through
  `serve-fp8` gains nothing (the patches only intercept the FP8 linear path).
- Keep `cudagraph_mode: FULL_DECODE_ONLY` with `mode:0`. **Do not** pair `mode:0`
  with `FULL_AND_PIECEWISE` — vLLM treats it as incompatible and silently drops
  to eager (~7 tok/s on 27B instead of ~40).

The `sm_70` viability flags (TRITON_ATTN, disabled custom all-reduce, cudagraph)
are V100 constraints, not FP8 ones.

## Known Deployment Rule

| Workload | Preferred path |
|---|---|
| Sparse MoE (Qwen3.5/3.6-A\*B, 122B, GLM) FP8 | This package — the flagship V100 case |
| Dense 27B-class | FP8 at 1–4 users; FP16 / GPTQ-Int4 at 8+ |
| Small dense | FP16 or Int4 |

## Performance

*These effects come from the vLLM/V100 serving path and model architecture, not
from FP8 accuracy.*

**cudagraph is mandatory for the headline numbers.** `mode=0 + FULL_DECODE_ONLY`
is the performance path; `--enforce-eager` is a correctness fallback only and
runs **~7–8× slower** (needs Python 3.12).

**FP8 decode runs on CUDA cores, so the concurrency pattern splits by model
type:**

- **Dense:** FP8 **wins at 1–2 users, ~ties at 4, falls behind FP16 at 8**
  (FP16 decode uses tensor cores more effectively at batch; ours is CUDA-core
  dequant→matmul). Closing the C8 gap needs a WMMA decode kernel that consumes
  FP8 directly — future work, no timeline.
- **Sparse MoE:** sparse activation (~3B params/token) keeps FP8's reduced weight
  traffic valuable **at every concurrency**, and for the largest models (122B,
  GLM) FP8 is **often the only format that fits**.

Full per-engine (0.19 / 0.21), per-concurrency numbers are the frozen matrix
(tag `fp8-v100-2026-matrix`) in the **V100 vLLM in 2026** write-up. Decision
matrix + the complete flag list: [`docs/COALESCED_FP8_GEMV.md`](docs/COALESCED_FP8_GEMV.md).

## Known limitations

**`--max-num-seqs 1` crashes on hybrid (attention + GDN/mamba) models under
cudagraph — upstream vLLM, not this package.** The Qwen3.5/3.6-A\*B checkpoints
(and the 27B) are hybrid. At `ns=1`, vLLM's cudagraph profiler allocates a
2-block KV cache whose `[2, 2, …]` tensor trips an attention/mamba layout
assert. It reproduces on **stock `vllm serve` with an FP16 checkpoint**, so
neither the wrapper nor FP8 is involved.

- **Supported config: `--max-num-seqs 8`** — wider capture cache, ambiguity gone,
  and faster than `ns=1` anyway.
- Non-hybrid / dense-attention models are unaffected at `ns=1`.
- If you truly need `ns=1`, run `--enforce-eager` (correct, ~7× slower).

## Optional: MTP speculative decoding

The production Qwen3.5/3.6-A\*B-FP8 checkpoints ship MTP head weights, and the
spec-decode path is stock vLLM. This package exposes it as an opt-in launcher
knob — **default OFF**:

```bash
ENABLE_QWEN_MTP=1 ./docker/run_docker_vllm019_py312.sh serve-fp8 <…Serve FP8 args…>
# appends: --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Decode-heavy traffic gains most (122B-A10B-FP8 TP8: ~1.36×; deeper `k` helps the
comm-bound flagship more); long-prompt/short-output can regress. **Not
bit-identical on MoE** (divergences were quality-equivalent in validation), and
acceptance rate alone is never proof — benchmark your own workload before
enabling globally. Full claims, per-model `k` curves, and the exactness analysis:
the **V100 vLLM in 2026** write-up, MTP chapter (`docs/04_mtp.md`).

## Profiling & diagnostics

The package ships extensive opt-in instrumentation — per-section decode timing,
MoE GEMM profiling, all-reduce attribution — **all OFF by default, zero cost in
normal serving**. Treat the hooks as **eager-only** (CUDA timing events give no
useful attribution under cudagraph capture; add `--enforce-eager` when
profiling, so numbers are relative attribution, not throughput):

```bash
VLLM_V100_FP8_DECODE_BREAKDOWN=1 ... serve-fp8 ... --enforce-eager
```

Full knob reference and a worked "find the bottleneck" workflow:
[`docs/PROFILING.md`](docs/PROFILING.md).

## Why monkey-patches, not a vLLM fork

This package patches vLLM at runtime — replacing the FP8 linear method and a few
MoE/attention paths on import — rather than forking and editing vLLM's source:

- **Stock vLLM stays unmodified.** You build the official vLLM source for
  `sm_70`; the patches apply when you run `python -m fp8_w8a16_sm70.vllm_serve`.
  No fork to rebase, no patched vLLM to rebuild.
- **The delta is explicit and small.** Everything we change lives in
  `src/fp8_w8a16_sm70/`, easy to audit against stock vLLM.
- **Trivially removable.** Drop the wrapper and you're back to stock vLLM — that
  is literally the `serve` subcommand; the FP16/GPTQ paths are unmodified.

Honest limitation: the patches propagate to TP workers via vLLM's worker
re-exec, verified on the 0.19/0.21 lanes but not guaranteed across other launch
styles (Ray executor, forkserver). If a future vLLM breaks that, the migration
path is a vLLM plugin or `sitecustomize.py` — not a fork.

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
