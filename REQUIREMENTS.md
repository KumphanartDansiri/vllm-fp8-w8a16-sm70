# Requirements

This document captures *what* this project depends on, *why*, and *which parts
are real constraints vs. deployment choices*. The discipline is to keep the
"hard requirements" layer minimal and explicit, so future upgrades (e.g.
moving from CUDA 12.8 to 12.9, picking up flash-attention-v100) are obvious
*movements within an envelope* rather than freelancing.

## Layer 1 — Hard requirements (cannot move without breaking the project)

| Constraint | Value | Source / Why |
|---|---|---|
| Python | `>=3.10, <3.14` | vLLM 0.18 `pyproject.toml` |
| vllm | `==0.18.0` (or 0.18.x) | **FP8 production anchor.** vLLM 0.18 remains the validated FP8 W8A16 baseline. vLLM 0.19 also works via source build for Qwen 3.5/3.6 FP8; vLLM 0.21 is a validated stock FP16/GPTQ engine base, but the FP8 wrapper port is still in flight. |
| torch | `==2.10.0` (any `+cuXX` wheel variant) | vLLM 0.18 `requirements/cuda.txt` exact pin for the current FP8 production baseline. The experimental vLLM 0.21 lane uses torch 2.11.0+cu126. |
| torchaudio | `==2.10.0` | vLLM 0.18 |
| torchvision | `==0.25.0` | vLLM 0.18 |
| flashinfer-python | `==0.6.6` | vLLM 0.18 (pure-Python wheel; cuXX-agnostic) |
| nvidia-cudnn-frontend | `>=1.13.0, <1.19.0` | vLLM 0.18 |
| transformers | `>=4.56.0, <5` | vLLM 0.18 |
| Hardware | NVIDIA V100 (sm_70 / compute capability 7.0) | Whole project exists because vLLM officially rejects sm_70 for FP8; this repo is the patch. |
| CUDA toolkit | `12.x` (any minor) | CUDA 13.x dropped sm_70. Stay in 12-series. |

## Layer 2 — sm_70 viability (must not consume features that require sm_75+)

vLLM 0.18 still has code paths that *would* use sm_80+ features if enabled. We
explicitly disable those to keep V100 working. Each item below is a hard "must
do" when invoking the serve:

| Setting | Reason |
|---|---|
| `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'` | Current v0.4.0 performance baseline. Python 3.12 avoids the Python <=3.10 FakeTensorMode cudagraph bug seen in vLLM 0.18. |
| `--enforce-eager` | Legacy correctness fallback for Python 3.10 or profiling/debugging paths that are not cudagraph-safe. Not the performance baseline. |
| `--attention-backend TRITON_ATTN` | FlashAttention v2+ needs sm_80+; Triton attention is the only V100-compatible backend in 0.18 |
| `--no-enable-chunked-prefill` | Chunked prefill has known instability on V100 in this vllm version |
| `--disable-custom-all-reduce` | vLLM's custom all-reduce uses sm_75+ features in some paths |
| `--quantization fp8` | Requires the monkey-patches in `fp8_w8a16_sm70.vllm_serve` (vLLM 0.18 would reject sm_70 otherwise) |
| `--max-num-seqs 8` (≥2; **not 1**) | On hybrid attention+GDN models (Qwen3.5/3.6-A\*B, 27B), `--max-num-seqs 1` under cudagraph crashes at init: vLLM 0.18's minimal cudagraph-profiling KV cache is 2 blocks wide and its attention/mamba layout check can't disambiguate the `[2,2,…]` shape (`assert shape[1] != 2`). Upstream vLLM bug — reproduces on stock `vllm serve` with FP16. `ns=8` sidesteps it (wider profiling cache) and is faster. For `ns=1` (low-latency streaming) use `--enforce-eager`. |

If a future workload needs a feature that *requires* sm_80+ (e.g. FP8
activation compute, native FA2), that capability is simply unavailable to us
on this hardware — not a project bug.

## Layer 3 — Deployment choices (we picked these; could pick differently)

| Choice | Current value | Why we picked it, alternatives |
|---|---|---|
| torch wheel CUDA variant | `+cu128` for the 0.18/0.19 FP8 baseline; `+cu126` for the 0.21 experiment | `+cu128` mirrors aiagent's verified production baseline. For vLLM 0.21, torch 2.11.0+cu128 dropped Volta from the wheel, while torch 2.11.0+cu126 keeps `sm_70`; that makes cu126 mandatory for the 0.21 lane. |
| Base docker image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` for 0.18/0.19; `nvidia/cuda:12.6.3-devel-ubuntu24.04` for 0.21 | Devel images are required because `torch.utils.cpp_extension.load` JIT-compiles `fp8_dequant.cu`. The 0.21 source build intentionally stays on CUDA 12.6 so the torch/vLLM stack preserves Volta support. Avoid CUDA 13. |
| transformers version | `4.57.6` | Matches aiagent. Any version in `[4.56.0, 5)` would satisfy vLLM. |
| numpy, safetensors, etc. | Aiagent-mirrored | Convenience; not load-bearing. |
| Python distribution | system Python 3.12 in Ubuntu 24.04 | Current v0.4.0 baseline. Python 3.10 remains a fallback only. |
| NVIDIA driver | Host-provided; previously observed `535.288.01` | Whatever the host has. cu128/cu129 wheels both run via NVIDIA forward-compat shim. Driver upgrade is the first rung in the post-v0.4.0 ladder. |
| Qwen3.5 MTP speculative decoding | Opt-in via `ENABLE_QWEN_MTP=1` (default OFF); v0.4.1 | Upstream-supported path in vLLM 0.18.0 (`qwen3_5_mtp.py`, `Qwen3_5MoeMTP` arch). Qwen3.5/3.6-A*B-FP8 checkpoints ship MTP head weights baked in. Default off pending v0.4.2 validation (multi-sample, production prompt mix, streaming). See README.md "Optional: MTP Speculative Decoding (v0.4.1)" and `docs/SESSION_LOG.md` Stage 4. |

## Layer 4 — Observed-but-not-enforced (informational)

Versions seen in our actual runs that satisfy the Layer 1-3 constraints. If
something breaks after an upgrade, suspect *these* before suspecting the hard
pins.

| Package | Observed |
|---|---|
| triton | `3.6.0` (transitive dep of torch 2.10.0; not vllm-pinned) |
| nvidia-cuda-runtime-cu12 | `12.8.x` (in cu128 wheels; would be `12.9.x` in cu129) |
| NCCL | bundled with torch; works on V100 without special config |

## Forward-looking notes

- **CUDA 13.0 drops sm_70.** If we ever move to CUDA 13 toolchain or runtime, V100 stops working. Stay in 12-series indefinitely.
- **PyTorch 2.11 wheel choice matters.** torch 2.11.0+cu128 dropped Volta, but torch 2.11.0+cu126 still includes `sm_70` and is validated for the vLLM 0.21 stock engine lane.
- **vLLM 0.21 is viable from source for stock FP16/GPTQ on V100.** Eager and cudagraph are validated across Qwen3.6-27B FP16, Qwen3.6-35B-A3B FP16, Gemma 4 FP16, and Qwen3.5-122B-A10B GPTQ-Int4. FP8 W8A16 is not validated on 0.21 yet; treat that as the next port target.
- **flash-attention-v100** is a candidate for accelerating attention beyond `TRITON_ATTN`. Wants cu129; that's a Layer-3 move, not blocked by Layer 1.

## How to update this document

When bumping anything, classify the change:

1. **Layer 1 change** — needs evidence that vllm 0.18 (or its declared deps) actually changed. Almost never moves.
2. **Layer 2 change** — needs evidence that the sm_70 viability concern has been retired. Rare; usually means newer hardware.
3. **Layer 3 change** — deliberate deployment decision. Just update the value and note why.
4. **Layer 4 change** — automatic; reflects whatever pip resolved.

The point is to never accidentally promote a Layer 4 observation into a Layer 1 requirement.
