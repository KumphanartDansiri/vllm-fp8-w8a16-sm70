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

## Development

Useful entry points:

- `python -m fp8_w8a16_sm70.vllm_serve` applies the vLLM monkey-patches.
- `src/fp8_w8a16_sm70/fp8_dequant.cu` contains the CUDA kernels.
- `src/fp8_w8a16_sm70/module.py` contains `FP8W8A16Linear`.
- `src/fp8_w8a16_sm70/ext_loader.py` centralizes JIT compilation.

Start every new session by reading `docs/SESSION_LOG.md`; it is the project
source of truth.
