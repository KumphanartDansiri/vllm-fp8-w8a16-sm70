# Volta (sm_70) fused-MoE config fix — upstream artifact

**Status:** validated on 8×V100-SXM2-32GB, vLLM 0.21.0 + CUDA 12.6, Triton 3.6.

**Report to BOTH engines, framed differently** (same `fused_moe.py` logic lives in both trees):
- **vLLM** (our main engine — we run vLLM 0.21 on V100, having learned the sm_70 rewire approach
  from aphrodite): file as an **issue/finding**, not an sm_70-support PR (that's declined by policy).
  Two parts that stand on their own merit: (1) the **diagnostic** — `get_default_config`'s decode
  branch picks `BLOCK_SIZE_K=128` for M≤64, which is pathological without `cp.async`; *worth checking
  whether the small-M default is even optimal on cp.async arches* (we only have V100 data — frame as a
  question, not a claim). (2) the **V100 tuned config JSONs** as a device-config data contribution
  (vLLM ships ~317 such files). Realistic outcome: the finding documents the issue for the many V100
  users still out there; configs *may* be accepted, the heuristic code likely not — file it anyway,
  it keeps our main-engine foundation honest and may surface a cross-arch default question.
- **aphrodite-engine** (the conceptual source — broad-arch culture): file as a **PR** with the full
  fix incl. the `get_default_config` sm<80 heuristic, credited as building on their sm_70 approach.
  Most likely to actually merge.

## Problem

Stock unquantized (fp16/bf16) MoE decode runs pathologically slow on V100 — *slower than a
same-class dense model*, which is backwards for a sparse MoE:

| model (TP4, fp16, cudagraph, ns=8) | stock decode | same-class dense |
|---|---|---|
| Qwen3.6-35B-A3B (MoE) | 15.6 tok/s | 27B dense = 37–41 |
| gemma-4-26B-A4B (MoE) | 10.9 tok/s | 31B dense = 17.6 |

## Root cause (code-traced + measured)

`fused_moe.py::get_default_config` has no sm_70 case. Its decode branch (M ≤ 64) picks
`BLOCK_SIZE_K = 128` (and `num_stages = 4`). On Volta — which has no `cp.async` and a smaller
96 KB shared-memory budget — `BLOCK_K=128` register-spills Triton's codegen; spill traffic
contends on HBM and the per-call cost grows ~linearly with batch. The only platform special-case
in that function is ROCm.

**What is NOT the cause:** `num_stages`. A 4/3/2 e2e sweep with the tile held fixed was flat
(15.57 tok/s on all arms, bit-identical output) — disproving the intuitive "no cp.async →
fewer stages" hypothesis. The dominant lever is **`BLOCK_SIZE_K`**: kernel time scales 64→632 µs,
128→1450 µs, 256→2300 µs at M=1. (Prefill, M>64, already gets `BLOCK_K=64` — only decode is hit.)

Precedent for a Volta case already exists in-tree: `moe_fused_mul_sum.py::_heuristic_config`
sets `num_stages = 2` for `is_sm80_before`. The main GEMM default was simply never given the
same treatment.

## Two-part contribution

### 1. Heuristic (covers every model/TP, no per-shape files)

Add an sm<80 small-M branch to `get_default_config` (unquantized path only):

```python
# in get_default_config, fp16/bf16 (dtype is None) branch:
if current_platform.has_device_capability(70) and not current_platform.has_device_capability(80) \
        and M <= 64:
    # Volta: BLOCK_K=128 register-spills Triton (no cp.async, 96KB smem). BLOCK_K=64 is
    # 2-9x faster; fat-N helps at the M~=8 knee. Mirrors moe_fused_mul_sum.py's sm<80 case.
    if M <= 4:
        return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
    return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2}
```

Reference impl (env-gated, monkey-patch form) shipped in this repo:
`src/fp8_w8a16_sm70/vllm_serve.py::_patch_volta_moe_default_config`
(`VLLM_V100_MOE_FP16_TUNED`, default ON).

### 2. Tuned configs (precise, headline shapes)

Per-M autotuned JSONs for the two reference shapes, drop into
`vllm/model_executor/layers/fused_moe/configs/`:
- `E=256,N=128,device_name=Tesla_V100-SXM2-32GB.json` (Qwen3.6-35B-A3B, TP4)
- `E=128,N=176,device_name=Tesla_V100-SXM2-32GB.json` (gemma-4-26B-A4B, TP4)

Produced by `tools/moe_volta_tune.py` (this repo) — a feasibility-pruned, shell-ordered tuner:
- **A-priori SMEM filter** `num_stages*(BM*BK+BK*BN)*2 ≤ 96 KB` drops infeasible big tiles
  *without compiling them* (288→204 configs; the stock `benchmark_moe.py` compiles them then
  catches `OutOfResources`, costing 100–240 s each — the reason a full stock tune is multi-day
  on Volta).
- Shell-walk from the low corner with a monotone early-stop (lossless on all 18 jobs tuned).

The JSON lookup (`get_moe_configs`) runs *before* `get_default_config`, so the two parts layer
automatically: tuned shapes use the exact JSON; everything else falls to the heuristic.

## Evidence (e2e, real models, this repo's `tools/moe_stages_ab_vllm021.sh`)

Decode tok/s, TP4 fp16 cudagraph FULL_DECODE_ONLY, outputs bit-identical to stock (pure speed):

| | stock | heuristic/ladder | tuned JSON |
|---|---|---|---|
| q35b single-stream | 15.56 | 65.91 (4.2×) | 65.85 |
| q35b 8 concurrent (agg) | 24.9 | 163.7 | **180.8** |
| g26b single-stream | 10.91 | 43.66 (4.0×) | 43.71 |
| g26b 8 concurrent (per-user) | 3.56 | 19.21 | **20.27** |

Single-stream: heuristic == tuned (M≤4 tiles identical). Concurrency: the tuned per-M JSON is
~5–10% faster (finer granularity across the fluctuating batch). Both restore sparse>dense.

## Scope / caveats

- fp16/bf16 unquantized MoE only. Quantized paths (fp8/int4) are untouched.
- The shipped JSONs cover the decode range M=1–64; for a complete upstream file, extend the tune
  to the full M ladder (1–4096) — prefill M>64 already gets BLOCK_K=64 from the stock default, so
  the decode range is where the fix matters.
- Structural ceiling remains: even tuned, Triton's sm_70 MoE GEMM is ~40× off the memory-bandwidth
  floor (no tensor-core `tl.dot` path on Volta). This config fix removes the *pathology*, not the
  architectural gap — a custom GEMV kernel is needed to close the rest.
- Filenames are TP-specific (N is the per-rank shard): re-tune per (model, TP).

## Reproduce

```bash
# tune (single shape, single GPU, ~15 min):
docker run --rm --gpus '"device=0"' -v $PWD:/work -w /work -e PYTHONPATH=/work/src \
  -e TUNE_E=256 -e TUNE_TOPK=8 -e TUNE_K=2048 -e TUNE_NSHARD=128 \
  vllm-v100:vllm021-cu126 python3 tools/moe_volta_tune.py /work/results/moe_volta_tune_q35b
# or the 8-GPU fleet (both shapes, ~50 min):  ./tools/moe_volta_tune_fleet.sh
# e2e verify:  MODEL_KEY=q35b ARMS='base auto' NUSERS=8 ./tools/moe_stages_ab_vllm021.sh
```
