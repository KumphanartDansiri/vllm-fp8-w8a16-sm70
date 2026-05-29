# FP8 W8A16 on V100 for vLLM

Volta-native fallback kernels and vLLM monkey-patches for serving
DeepSeek-style block-FP8 W8A16 models on NVIDIA Tesla V100 (`sm_70`).

Upstream vLLM rejects FP8 on this architecture. This package keeps vLLM 0.18.0
usable on V100 by replacing the FP8 linear path with custom CUDA kernels,
including a Volta WMMA prefill path and grouped routed MoE decode kernels.

## Current Baseline

The performance baseline is `v0.4.0`:

| Component | Baseline |
|---|---|
| Python | 3.12 |
| vLLM | 0.18.0 |
| torch | 2.10.0+cu128 |
| CUDA | 12.x runtime/toolkit; do not use CUDA 13 on V100 |
| Docker image | `vllm-v100-py312-test:cu128` |
| Launcher | `docker/run_docker_vllm018_py312.sh` |

The legacy Python 3.10 / `--enforce-eager` stack remains a correctness fallback,
not the default performance path.

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
    --max-num-seqs 1 \
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

## Known Deployment Rule

| Workload | Preferred path |
|---|---|
| Qwen3.5/Qwen3.6 MoE+GDN FP8 | This package on the py3.12 cudagraph baseline |
| Dense+GDN 27B-class | GPTQ-Int4 if available; otherwise FP16 |
| Small dense | FP16 or Int4 |

Dense FP8 on V100 is currently slower than FP16 and GPTQ-Int4 because the
custom dequant/GEMM path outweighs the bandwidth savings. See
`docs/STAGE_3.1_NEXT_STEPS.md` before investing in dense-FP8 work.

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
    ...  # same args as the v0.4.0 launch
```

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

## Development

Useful entry points:

- `python -m fp8_w8a16_sm70.vllm_serve` applies the vLLM monkey-patches.
- `src/fp8_w8a16_sm70/fp8_dequant.cu` contains the CUDA kernels.
- `src/fp8_w8a16_sm70/module.py` contains `FP8W8A16Linear`.
- `src/fp8_w8a16_sm70/ext_loader.py` centralizes JIT compilation.

Start every new session by reading `docs/SESSION_LOG.md`; it is the project
source of truth.
