# Aiagent's Verified Working Environment (Production Baseline)

> **Frozen snapshot captured 2026-05-23.** This is the known-good combination that runs vLLM
> on this hardware. Treat as read-only reference — do **not** modify aiagent's setup. Our
> dev environment in `Dockerfile.dev` mirrors these versions exactly.

---

## Hardware

| | |
|---|---|
| Host | `llm` (Ubuntu 22.04.5 LTS, kernel 5.15.0-177-generic) |
| CPU | 80 cores |
| RAM | 502 GiB |
| GPUs | **8× Tesla V100-SXM2-32GB** (compute capability **7.0** / sm_70) |
| Total GPU memory | 256 GB across all 8 |
| Driver | **535.288.01** (driver-native CUDA ≤ 12.2; runs cu128 wheels via NVIDIA forward-compat) |
| CUDA toolkit on host | 12.4 at `/usr/local/cuda-12.4/` |

## Software stack inside `/home/aiagent/vllm-env/`

| Package | Version |
|---|---|
| Python | **3.10.12** |
| torch | **2.10.0+cu128** |
| torchvision | 0.25.0 |
| torchaudio | 2.10.0 |
| vllm | **0.18.0** |
| transformers | 4.57.6 |
| triton | 3.6.0 |
| flashinfer-python | 0.6.6 |
| safetensors | 0.7.0 |
| numpy | 2.2.6 |

`torch.cuda.get_arch_list()` → `['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']` — Volta IS included.

## Verified working vLLM command

The command aiagent uses to serve **Qwen3.5-122B-A10B-GPTQ-Int4** on 8 V100s:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 \
/home/aiagent/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model /mnt/models/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 32768 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-num-seqs 1 \
    --quantization gptq \
    --dtype float16 \
    --enforce-eager \
    --disable-custom-all-reduce \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --no-enable-chunked-prefill \
    --attention-backend TRITON_ATTN
```

### Why each flag is set (decoded)

| Flag / env var | Reason on V100 |
|---|---|
| `NCCL_P2P_DISABLE=1` | NCCL peer-to-peer over NVLink/PCIe has known issues on this configuration; fall back to host-staged transfers |
| `NCCL_IB_DISABLE=1` | No InfiniBand on this host; explicit disable to suppress noisy probes |
| `NCCL_SHM_DISABLE=1` | Shared-memory transport disabled (workaround for a NCCL+container issue) |
| `--quantization gptq` | GPTQ INT4 W4A16 — supported on sm_60+ via `gptq_gemm` (CUDA-core kernel) |
| `--dtype float16` | FP16 activations — V100 has FP16 tensor cores (`HMMA.884`) |
| `--enforce-eager` | Disable `torch.compile` / CUDA graphs — some compiled paths assume sm_80+ |
| `--disable-custom-all-reduce` | Use NCCL all-reduce; vLLM's custom all-reduce requires sm_75+ for some paths |
| `--attention-backend TRITON_ATTN` | FlashAttention v2 requires sm_80+; Triton backend is the fallback that supports Volta |
| `--no-enable-chunked-prefill` | Chunked prefill has had instability on V100 in this vllm version |
| `--max-num-seqs 1` | Single-stream serving (decode-focused workload) |
| `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` | Application-level: function-calling format for the Qwen3 Coder model |

### What's NOT supported on this stack

These would fail on V100 with the current vLLM 0.18 code:

- **FP8 W8A16 quantization** (`--quantization fp8`) — requires sm_75+ via Marlin fallback. This is **the gap our integration fills**.
- **FlashAttention v2** (`--attention-backend FLASH_ATTN`) — needs sm_80+
- **`--enable-prefix-caching`** in some configurations (chunked prefill dependency)
- **Some `torch.compile` paths** — hence `--enforce-eager`
- **Native FP8 compute** (`activation_scheme: "dynamic"` matmuls) — needs sm_89+

## Models on `/mnt/models/`

| Path | Format | Notes |
|---|---|---|
| `/mnt/models/Qwen3.5-122B-A10B-GPTQ-Int4` | GPTQ INT4 W4A16 | aiagent serves this; works on V100 today |
| `/mnt/models/Qwen3.5-4B-FP8` | DeepSeek-style block-FP8 (W8A8 designed) | Target of our V100 W8A16 work |
| `/mnt/models/Qwen3.6-27B-FP8` | DeepSeek-style block-FP8 | Larger FP8 model |
| `/mnt/models/Qwen3.5-122B-A10B-FP8` | DeepSeek-style block-FP8 | (downloading at time of capture) |

All FP8 models use:
- `quant_method: "fp8"` in config.json (not `compressed-tensors`)
- `weight_block_size: [128, 128]` (2D block scales)
- `weight_scale_inv` named scale tensors (BF16)
- `activation_scheme: "dynamic"` (designed for W8A8)

## Driver / CUDA compatibility note

The aiagent stack runs **torch 2.10.0+cu128** on a host with **driver 535.288.01** (which only natively supports up to CUDA 12.2). This works because:

1. PyTorch's `cu128` wheels bundle the CUDA 12.8 runtime libraries (`nvidia_cublas_cu12`, `nvidia_cuda_runtime_cu12`, etc.) inside the package
2. NVIDIA's userspace driver maintains **forward compatibility** for minor versions within a major (drivers ≥ R525 support up to CUDA 12.x via compat)
3. The host's `libcuda.so` (driver 535) is loaded; the userspace CUDA runtime libs from torch's wheel are what the application links to

Downsides: some CUDA 12.8 features that require driver functions added after R535 will silently degrade or warn. So far this hasn't caused observable problems for GPTQ serving.

The driver wasn't upgraded to R560+ (which would natively support cu128) because of compatibility requirements with other software on this host.

## What this means for our dev environment

Our `Dockerfile.dev` mirrors **exactly** this combination:
- Same Python version (3.10)
- Same torch+CUDA combo (2.10.0+cu128)
- Same vllm version (0.18.0)
- Same key deps

This guarantees: if aiagent's GPTQ command works on bare metal, our patched FP8 command should work inside our Docker container on the same hardware. Any version drift would introduce uncertainty about whether failures are due to our changes or env differences.

If we later find a workaround for some limitation (e.g., chunked prefill on V100), we can deviate from this baseline — but ONLY after the baseline is proven working, and document the deviation explicitly.
