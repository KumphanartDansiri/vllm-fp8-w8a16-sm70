# FP8 W8A16 on V100 for vLLM

A vendored TurboMind FP8 engine (default for block-128) plus Volta-native dequant
kernels and vLLM monkey-patches, for serving block-FP8 W8A16 models — dense and
MoE — on NVIDIA Tesla V100 (`sm_70`).

Upstream vLLM dropped practical V100 support; this package brings it back on
**vLLM 0.19 and 0.21** (source-built on CUDA 12.6). Beyond the FP8 W8A16 linear
path (Volta WMMA prefill + grouped/coalesced CUDA-core MoE decode) it bundles a
couple of `sm_70` enablers that ride along in the same plugin: the **FP16-MoE
config fix** and a **FlashAttention-V100 prefill + MLA bridge**.

The *story* — why V100 still works in 2026, the full per-engine benchmark
matrix, the methodology, and deployment field-notes — lives in the companion
**V100 vLLM in 2026** write-up (the [`v100-vllm-2026`](https://github.com/KumphanartDansiri/v100-vllm-2026) repo). **This README is the
artifact**: what the plugin is, how it works, and how to build, run, and extend
it.

> **This is not native FP8 execution.** V100 (`sm_70`) has no FP8 tensor-core
> path. Weights are stored as block-scaled FP8 and dequantized to FP16 before the
> math — interleaved in registers inside the TurboMind engine's WMMA kernel, or
> *inside* our fallback GEMM kernels — never round-tripping an FP16 weight matrix
> through HBM, then executed as FP16. See **How it works** below.

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

**Mechanism.** Weights are stored in block-scaled FP8 (E4M3); activations stay
FP16 (W8A16). Volta has **no FP8 tensor cores**, so FP8 here is a
*memory-bandwidth* play — read half the weight bytes, compute in FP16. **Two
engines** implement it, picked **per weight at load**: **our dequant kernels
(CUDA cores)** and the vendored **TurboMind engine (Volta tensor cores)** —
see [FP8 backends](#fp8-backends).

**TurboMind engine — the default for block-128 (tensor cores).** A vendored
LMDeploy `s884` SM70 FP8 GEMM — we found it through the **1catai-vLLM** project's
V100 work and vendored it (Apache-2.0). It's a *format-integrated* kernel: the
block-scaled FP8 weight is unpacked and dequantized in registers *interleaved
with* the Volta `m8n8k4` **HMMA tensor-core** matmul — the way Marlin/GPTQ-Marlin
weave the quantized format into the pipeline, rather than pre-dequantizing to an
FP16 matrix. This is the real serving backend for block-128 weights: native
`quant_method=fp8` (Qwen/DeepSeek) **and** compressed-tensors block (e.g.
`gemma-4-31B-FP8-block`).

**Our dequant path — universal fallback + correctness reference.** A
*fused dequant→compute* design: custom Volta kernels apply the per-block scale
and dequantize FP8→FP16 *inside* the compute kernel, immediately before the FP16
math, so dequantized weights never round-trip through HBM. The **decode** hot path
runs on **CUDA cores** (a warp-per-output coalesced GEMV / grouped MoE GEMM);
prefill uses a WMMA matmul on FP16 tiles staged in shared memory. It carries
everything TurboMind can't: channel/tensor scale (RedHatAI CT-channel,
GLM-4.5-Air), non-128-aligned TP shards, and any image without the engine baked
in. It is also the bit-exact reference the engine is validated against.

**The Volta honesty.** Our path is fused-dequant, *not* a format-integrated
kernel — on `sm_70` there is no `cp.async`, and for our CUDA-core decode the
dequant-feeds-FP16 structure is pragmatic, not optimal. TurboMind closes that gap
for the block-128 case (a real format-integrated kernel); ours remains the
coverage path where one doesn't apply. If you arrived assuming Hopper FP8 tensor
cores or "half the memory for free" — this is block-FP8 on hardware with neither
native FP8 nor `cp.async`, made to serve dense **and** MoE where per-token
sparsity makes "read fewer bytes" matter more than peak GEMM efficiency.

## FP8 backends

`VLLM_V100_FP8_BACKEND` selects the FP8 GEMM backend per weight:

| Value | Behavior |
|---|---|
| `auto` *(default)* | TurboMind where **eligible**, our dequant path everywhere else |
| `ours` | force our dequant path (the pre-engine behavior) |
| `turbomind` | force TurboMind; **raise** if a weight is ineligible (never a silent wrong backend) |

**"FP8" is not one thing.** What a checkpoint actually ships decides which engine
serves it. TurboMind is **eligible** only for block-128 scale
(`weight_block_size == (128,128)`) with per-rank (post-TP-shard) `N`,`K` both
`% 128 == 0` and the engine present; ours (CUDA-core) carries everything else:

| FP8 type | Config signature | Ours (CUDA-core) | TurboMind (tensor-core) |
|---|---|:--:|:--:|
| Native block | `quant_method=fp8`, `weight_block_size=[128,128]` | ✓ fallback/ref | ✓ **default** |
| CT block (dense) | `compressed-tensors`, `strategy=block`, `[128,128]` | ✓ (FP16-dequant if forced) | ✓ **default** |
| CT channel / dynamic | `compressed-tensors`, `strategy=channel` | ✓ | ✗ |
| CT tensor | `compressed-tensors`, `strategy=tensor` | ✓ | ✗ |
| Native block **MoE** (Qwen) | native block expert scales | ✓ fallback/ref | ✓ **default** (per-shard; TP8 `w2 K=I/tp` → ours) |
| CT block **MoE** | `compressed-tensors` block MoE | ✗ raises¹ | ✗ deferred² |
| CT channel/dynamic **MoE** | RedHatAI / Gemma / GLM | ✓ | ✗ |

Rule of thumb: **block-128 → TurboMind; channel / tensor / dynamic / unaligned
shards → ours.** With the engine absent, behavior is byte-for-byte the pre-engine
(ours) path. ¹ Ours refuses block-strategy CT MoE (needs block-scale expansion;
CHANNEL/TENSOR only). ² TurboMind wiring is scoped but deferred — no CT-block-MoE
checkpoint exists to validate against; it's the only format on **neither** path.

**Coverage/memory fix — compressed-tensors block.** A checkpoint *labeled*
FP8-block used to fall onto our FP16-dequant path purely because it's
compressed-tensors instead of native `quant_method=fp8` — doubling weight VRAM.
Wiring the compressed-tensors loader to the engine, `gemma-4-31B-FP8-block` went
from **OOM at TP2** (30.4 GiB/worker, FP16-dequant) to **fitting at TP2**
(TurboMind FP8-resident, ~17 GiB/worker), output **bit-identical** to the
FP16-dequant reference (517/517 tokens), single-user decode **1.41×**. Guarded by
a mandatory per-layer loader self-check and the `VLLM_V100_CT_BLOCK_TM` kill
switch; falls back to FP16-dequant on any divergence.

**Tested models.** TurboMind-vs-ours agreement is measured eager/greedy (temp 0);
"benign ties" = every divergence is a logit-tie (worst Δlogp ≤ 0.08 across all
runs, no systematic divergence — dense native-block tends bit-identical, MoE and
hybrid land on more ties):

| Model | FP8 format | D/MoE | Backend · result |
|---|---|---|---|
| Qwen3.5-27B-FP8 | native block | dense | **TurboMind** · 100% bit-identical |
| Qwen3.6-27B-FP8 | native block | dense (hybrid) | **TurboMind** · 93.3%, benign ties |
| Qwen3.5-35B-A3B-FP8 | native block | MoE | **TurboMind** · 87–94%, benign ties |
| Qwen3.6-35B-A3B-FP8 | native block | MoE | **TurboMind** · 75.0%, benign ties |
| Qwen3.5-122B-A10B-FP8 | native block | MoE | **mixed** (tm where shard-eligible, ours on TP8) · 80.6%, benign |
| gemma-4-31B-it-FP8-block | CT block | dense | **TurboMind** · 100% bit-identical; fits TP2, 1.41× C1 |
| gemma-4-31B-it-FP8-Dynamic | CT channel | dense | **ours** · coherent |
| gemma-4-26B-A4B-it-FP8-Dynamic | CT channel | MoE | **ours** · coherent |
| GLM-4.5-Air-FP8 | CT channel | MoE | **ours** · validated separately |

**Both engines (0.19 + 0.21).** The engine and its loader wiring are identical on
vLLM 0.19 and 0.21 — the compatibility table above is **engine-invariant** (verified
by the engine engaging on both loaders). Only *performance* differs: 0.19 is the
faster decode lane (~10–20%), for the engine too.

**Performance** (cudagraph decode, TP2, per-user tok/s at C1 / C8):

| Model | ours (0.19) | TurboMind (0.19) | ours (0.21) | TurboMind (0.21) |
|---|---|---|---|---|
| Qwen3.5-27B (dense) | 36.6 / 12.8 | **39.9 / 34.3** | 33.5 / 12.3 | 36.1 / 31.0 |
| Qwen3.5-35B-A3B (MoE) | 95.0 / 46.6 | 95.2 / **69.5** | 78.9 / 42.0 | 78.1 / 63.1 |

Reading it: **dense** — TurboMind is modest at C1 (~1.09×) but **decisive at C8
(~2.7×)**, where ours' CUDA-core dense decode falls behind; **MoE** — a **tie at C1**
(ours' grouped GEMV matches the engine at single-user), TurboMind ~1.5× at C8. Both
backends cudagraph on both engines (ours' MoE cudagraph needs the launcher's default
`VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1` — a capture-safe route-prep). And separately,
on `gemma-4-31B-FP8-block` (TP4) TurboMind is 1.41× at C1 and **fits at TP2** — half
the GPUs — where ours OOMs. More: [`docs/FP8_ENGINE_STAGE_G_PERF.md`](docs/FP8_ENGINE_STAGE_G_PERF.md).

The engine is compiled **ahead-of-time into the image** (`docker/Dockerfile.fp8engine`
→ `vllm-v100:vllm021-cu126-fp8engine`, no runtime JIT). Vendored from
[LMDeploy](https://github.com/InternLM/lmdeploy) `v0.14.0` (Apache-2.0) with three
documented V100 deltas, in `third_party/turbomind_gemm_sm70/`. Design + validation:
[`docs/FP8_ENGINE_STAGE_F_LOADER_WIRING.md`](docs/FP8_ENGINE_STAGE_F_LOADER_WIRING.md),
[`docs/FP8_ENGINE_STAGE_G_PERF.md`](docs/FP8_ENGINE_STAGE_G_PERF.md),
[`docs/FP8_ENGINE_STAGE_H_CT_BLOCK_WIRING.md`](docs/FP8_ENGINE_STAGE_H_CT_BLOCK_WIRING.md).

## Status

| Tier | What |
|---|---|
| **Known good** | **vLLM 0.19 + 0.21, source-built on CUDA 12.6** (0.21 = newest models; 0.19 = faster decode; both run the FP8 plugin), torch 2.10–2.11, Python 3.12, NVIDIA V100 (`sm_70`); 7 model families — dense + MoE + MLA — across FP8 / FP16 / GPTQ-Int4. The frozen matrix is tag `fp8-v100-2026-matrix`. |
| **Validated** | dense-FP8 (coalesced + branchless dequant → *faster than FP16* at low concurrency); FlashAttention-V100 prefill + MLA bridge (GLM-4.7-Flash on both engines); **TurboMind FP8 engine** (default via `auto`) — serving-exactness vs ours: dense **bit-identical**, MoE benign logit-ties, 122B-A10B TP8 flagship agreement; compressed-tensors block wired in (`gemma-4-31B-FP8-block`: fits TP2, bit-identical, 1.41× C1). |
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

> **For the TurboMind engine (default for block-128), serve from the
> engine-baked image** `vllm-v100:vllm021-cu126-fp8engine`; `VLLM_V100_FP8_BACKEND`
> defaults to `auto` (TurboMind where eligible, ours elsewhere). The base
> `vllm-v100:vllm021-cu126` image has no engine, so it runs the dequant path
> unchanged. See [FP8 backends](#fp8-backends).

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

- **Dense:** on **our dequant path**, FP8 **wins at 1–2 users, ~ties at 4, falls
  behind FP16 at 8** (FP16 batch decode uses tensor cores more effectively; ours
  is CUDA-core dequant→matmul). The **TurboMind engine** (default) consumes FP8
  directly through its `s884` WMMA kernel — the "WMMA decode kernel that consumes
  FP8 directly" this section used to file under future work is now the shipped
  default, so the C8 ceiling is engine- and model-dependent, not a fixed CUDA-core
  limit (e.g. `gemma-4-31B-FP8-block`: TurboMind leads ours **1.41× at C1** and
  fits at **half the TP**).
- **Sparse MoE:** sparse activation (~3B params/token) keeps FP8's reduced weight
  traffic valuable **at every concurrency**, and for the largest models (122B,
  GLM) FP8 is **often the only format that fits**.

Full per-engine (0.19 / 0.21), per-concurrency numbers are the frozen matrix
(tag `fp8-v100-2026-matrix`) in the **V100 vLLM in 2026** write-up. TurboMind-vs-ours
serving perf (concurrency, TTFT, MoE cudagraph): [`docs/FP8_ENGINE_STAGE_G_PERF.md`](docs/FP8_ENGINE_STAGE_G_PERF.md).
Decision matrix + the complete flag list: [`docs/COALESCED_FP8_GEMV.md`](docs/COALESCED_FP8_GEMV.md).

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

The default FP8 GEMM backend for block-128 weights is the `s884` SM70 kernel
vendored from [**LMDeploy / TurboMind**](https://github.com/InternLM/lmdeploy)
`v0.14.0` (Apache-2.0), in `third_party/turbomind_gemm_sm70/` with three
documented V100 deltas. This project does **not** rewrite that kernel: it wires
the compiled engine into vLLM's FP8 loaders behind `VLLM_V100_FP8_BACKEND`
(`src/fp8_w8a16_sm70/turbomind_fp8_backend.py`), and validates every eligible
weight against our own dequant path as the bit-exact reference.

Our FP8 W8A16 dequant kernels (Volta WMMA prefill + grouped/coalesced CUDA-core
MoE decode — the universal fallback and correctness reference), the
compressed-tensors → TurboMind loader wiring, the FP16-MoE `sm_70` config fix,
and the benchmark harness are original to this project.
