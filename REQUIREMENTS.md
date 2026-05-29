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
| vllm | `==0.18.0` (or 0.18.x) | **Project anchor.** vLLM `>=0.20` drops sm_70 support entirely. We stay on 0.18 because that's the last version that still tolerates V100. |
| torch | `==2.10.0` (any `+cuXX` wheel variant) | vLLM 0.18 `requirements/cuda.txt` exact pin |
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

If a future workload needs a feature that *requires* sm_80+ (e.g. FP8
activation compute, native FA2), that capability is simply unavailable to us
on this hardware — not a project bug.

## Layer 3 — Deployment choices (we picked these; could pick differently)

| Choice | Current value | Why we picked it, alternatives |
|---|---|---|
| torch wheel CUDA variant | `+cu128` | Mirrors aiagent's verified production baseline. `+cu129` is also valid (newer minor, same sm_70 retention, FA-V100 prefers it). Switching cu128↔cu129 is a Layer-3 move, not a Layer-1 change. |
| Base docker image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` | Current py3.12 baseline image. Devel image is required because `torch.utils.cpp_extension.load` JIT-compiles `fp8_dequant.cu`. Could try 12.9 as a Layer-3 move; avoid CUDA 13. |
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
- **PyTorch 2.11+ may drop sm_70.** Currently 2.10.0 still includes sm_70 in `torch.cuda.get_arch_list()`. Watch the release notes.
- **vLLM 0.19 → 0.20** is the next break point. 0.19 likely still tolerates sm_70 with capability fallbacks (untested by us); 0.20 explicitly drops it.
- **flash-attention-v100** is a candidate for accelerating attention beyond `TRITON_ATTN`. Wants cu129; that's a Layer-3 move, not blocked by Layer 1.

## How to update this document

When bumping anything, classify the change:

1. **Layer 1 change** — needs evidence that vllm 0.18 (or its declared deps) actually changed. Almost never moves.
2. **Layer 2 change** — needs evidence that the sm_70 viability concern has been retired. Rare; usually means newer hardware.
3. **Layer 3 change** — deliberate deployment decision. Just update the value and note why.
4. **Layer 4 change** — automatic; reflects whatever pip resolved.

The point is to never accidentally promote a Layer 4 observation into a Layer 1 requirement.
