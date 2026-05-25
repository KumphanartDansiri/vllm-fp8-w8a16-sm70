# V100 FP8 W8A16 — Session Log

Cold-start summary for picking up in a new session. Read top to bottom.

---

## Session 4 handoff (2026-05-24) — installable package + memory migration

### Project scope

Make DeepSeek-style block-FP8 W8A16 quantized models run on NVIDIA Tesla V100
(`sm_70`) under vLLM, which upstream rejects for FP8
(`Fp8Config.get_min_capability = 89`). The immediate target is Qwen3-Next
hybrid self-attention + linear-attention models served on a 4x V100 box.

Why not GPTQ:
- FP8 gives roughly 3x larger KV-cache budget than FP16 on this hardware.
- The real workload hits prefill `M >= 16` and batched serving.

Project root:
- `/home/kumphanartd/vllm-fp8-w8a16-sm70`
- Standalone pip-installable package: `fp8-w8a16-sm70`
- GitLab remote: `tl-group/opensource/vllm-fp8-w8a16-sm70`
- Branch: `main`

Cold-start ritual: read this file first. It is the source of truth.

### Current package layout

```text
src/fp8_w8a16_sm70/
├── fp8_dequant.cu     # all kernels: Phase 1,2,4,5,A.1,A.2,A.3,A.4 WMMA
├── vllm_serve.py      # monkey-patch wrapper
├── module.py          # FP8W8A16Linear nn.Module
└── ext_loader.py      # centralized JIT compile
```

Entry point:

```bash
python -m fp8_w8a16_sm70.vllm_serve
```

### Related folders and external projects

| Path | Role |
|---|---|
| `/home/kumphanartd/vllm` | Read-only reference for vLLM internals. Old `customize-v100` branch is historical; do not treat it as canonical. |
| `/home/you/workspace/flash-attention-v100` | Third-party `sm_70` FlashAttention-2 fork. Wants cu129. Not currently wired into serve path; current deployment uses `TRITON_ATTN`. |
| `/home/you/workspace/1catai-vllm` | Separate workspace; not detailed in memory. |
| `/home/aiagent/vllm-env` | Production baseline stack mirrored in `docs/AIAGENT_ENV.md`. |
| `/mnt/models/Qwen3.5-4B-FP8` | Small dense Qwen3-Next, TP=1 verified. |
| `/mnt/models/Qwen3.6-27B-FP8` | 27B hybrid: 17 self-attn + 47 linear-attn, TP=4 verified. |
| `/mnt/models/Qwen3.6-27B-GPTQ-Int4` | aiagent production baseline: same architecture, INT4. |
| `/mnt/models/Qwen3.5-122B-A10B-FP8` | 122B-A10B MoE FP8, TP=8 verified on 8x V100. |
| `/home/kumphanartd/vllm/docs/v100reference` | NVIDIA Volta whitepaper / tuning guide PDFs, not in this repo. |

### Dependency envelope

See `REQUIREMENTS.md`.

Hard anchor:
- vLLM 0.18.x, the last version known here to tolerate `sm_70`.
- vLLM 0.20 dropped `sm_70`.
- torch 2.10.0.
- flashinfer 0.6.6.
- Python 3.10-3.13.
- CUDA 12.x only. CUDA 13 drops `sm_70`.

Required serve flags:

```bash
--enforce-eager \
--attention-backend TRITON_ATTN \
--no-enable-chunked-prefill \
--disable-custom-all-reduce \
--quantization fp8
```

Deployment choice:
- `+cu128` wheels in a `vllm-v100-dev:cu128` docker image.

### Chronology

Session 1: get it working at all.
- Wrote the monkey-patch wrapper to replace `Fp8LinearMethod` with
  `FP8W8A16Linear`.
- Built `sm_70` FP8 dequant + GEMM kernels: Phase 1-5 and A.1-A.3.
- Verified coherent end-to-end generation at TP=1 on the 4B model.
- Wrote diagnostic toolkit: offline attention-shape test, byte-hash dump,
  microbenchmarks.

Session 2: fix TP>1 garbage tokens and characterize performance.
- Symptom: TP=4 inference produced repeated `"!!!!!"`.
- Root cause: every custom kernel launch used default stream 0 and raced with
  NCCL on non-default streams, consuming half-written output and cascading to
  NaNs/token 0.
- Fix: all launches pass `at::cuda::getCurrentCUDAStream()` via the
  `V100_FP8_STREAM` macro.
- Steady-state performance reached GPTQ-Int4 parity: decode 6.025 vs 6.114
  tok/s, prefill roughly +15%, wall roughly +1.5%.
- Found `/tmp/torchinductor_root` was ephemeral; persistent mount reduced
  cold restart from roughly 14 minutes to roughly 40 seconds after cache warmup.

Session 3: WMMA / Tensor Cores for prefill.
- Extended microbenchmarks with an M sweep (`BENCH_SWEEP=1`).
- Found A.2 CUDA-core kernel was 10-15x behind cuBLAS at M=64, and 30-50x
  behind at M=512+.
- End-to-end parity was partly because the GPTQ baseline on `sm_70` also falls
  back to CUDA cores; Marlin needs `sm_75+`.
- Wrote `fp8_w8a16_gemm_wmma_poc`: 64x64 tile, 4 warps, double-buffered A/B,
  per-warp 16x16 FP32 staging.
- Result: 9-13x over A.2 at M=64-4096, with 22-25 TFLOP/s peak.
- Dispatch threshold: `FP8_WMMA_MIN_M=64`, with A.2 tail fallback.

WMMA end-to-end A/B:

| M | WMMA off | WMMA on | Speedup |
|---|---:|---:|---:|
| 128 | 747 ms | 283 ms | 2.64x |
| 512 | 2768 ms | 479 ms | 5.77x |
| 1024 | 5472 ms | 869 ms | 6.30x |

Other WMMA findings:
- Init engine improved from 61 s to 40 s.
- 95% of Linear calls hit WMMA during serving.
- Shared-memory padding did not help; bank conflicts were not the bottleneck.
- Dropping `C_smem` helped by 12-16%, occupancy 25% -> 37%.
- Double-buffering A/B helped by only about 1%; `stall_wait` dominated.

Session 4:
- Landed WMMA path: commit `9c773d1`.
- Restructured to `src/fp8_w8a16_sm70` package layout: commit `695cc90`.
- Updated imports and docker paths: commit `17790ef`.
- Project directory moved from `/home/kumphanartd/vllm/experiments/v100_fp8_test`
  to `/home/kumphanartd/vllm-fp8-w8a16-sm70`.

### Open and deferred

- Further WMMA optimization is deferred. Closing the remaining 3-4x gap to
  cuBLAS likely needs larger tiles or breaking the `c_frag` dependency chain.
  Current speedup is already transformative and prefill is no longer the
  bottleneck at typical M.
- FP8 MoE is now verified. `/mnt/models/Qwen3.5-122B-A10B-FP8` loads and
  serves coherently on 8x V100 with the Volta block-FP8 MoE fallback plus the
  active-expert-list optimization. Steady 32k-context agent decode is roughly
  2.5-2.7 tok/s. Stage 2B is the next work item: optimize MoE, which accounts
  for ~71-74% of decode time.
- A.3 `atomicAdd` is not bit-deterministic across runs. This is acceptable for
  serving; document only.
- `--enforce-eager` cannot be dropped because vLLM compiled paths assume
  `sm_80+`.
- `flash-attention-v100` integration is not done and would require a cu129
  toolchain decision.

### Working-style notes

- Relay GPT/Claude peer review carefully. Validate claims and do not defer or
  argue without evidence.
- Measure before recommending optimization; architectural intuition is not
  sufficient evidence.
- Custom CUDA kernels in PyTorch extensions must use
  `at::cuda::getCurrentCUDAStream()`.

---

## V100 FP8 W8A16 — Session Wrap-up (2026-05-23, session 2)

Historical session-2 summary preserved below.

> **History note:** session 1 (earlier on 2026-05-23) brought 4B+TP=1 working and
> initially loaded 27B+TP=4 but inference produced `"!!!!!"`. Session 2 (this one)
> found and fixed the root cause and characterized performance fully. See bottom
> of this file for the historical session-1 record (kept for context).

---

## Goal achieved (session 2 final state)

vLLM 0.18.0 serves DeepSeek-style block-FP8 W8A16 models on Tesla V100 (sm_70)
**with coherent inference output AND production-acceptable performance.**

Verified working configurations on this box (8× V100-SXM2-32GB, driver 535.288.01):

| Model | TP | Result | Init engine (warm) | Decode tok/s | Notes |
|---|---|---|---|---|---|
| Qwen3.5-4B-FP8 | 1 | ✅ Coherent text | (not re-measured) | ~12 | working since session 1 |
| Qwen3.6-27B-FP8 | 4 | ✅ Coherent reasoning text | ~1 min | **6.0 tok/s, parity with GPTQ-Int4** | bug fixed + cache mount working |

Same hardware aiagent uses for production GPTQ-Int4 serving (`/home/aiagent/vllm-env/`).

---

## THE BUG AND THE FIX (session 2's main result)

**Bug:** custom CUDA kernels in `fp8_dequant.cu` launched on stream 0 (`kernel<<<grid, block>>>(...)`),
not on PyTorch's current CUDA stream. At TP>1 vLLM/NCCL runs on non-default streams.
Without explicit stream synchronization, downstream torch reads of our kernel output were
not stream-ordered with the kernel writes → consumed half-written buffers → cascading
garbage → logits collapse → greedy argmax over NaN row returns token id 0 = `"!"`.

Why it eluded testing for so long:
- Offline kernel tests (test_attention_shapes.py) passed 100/100 — single-stream, no race
- TP=1 inference worked — no NCCL, no extra streams
- Hash verification of loaded weights passed — uses standard torch ops, not our kernel
- Only at TP>1 with NCCL coordinating did the race manifest

**Fix:** every kernel launch in `fp8_dequant.cu` now passes `at::cuda::getCurrentCUDAStream()`
as the 4th `<<<...>>>` argument via the `V100_FP8_STREAM` macro. One-line per launch site, 7 sites total.

Diagnostic crystal: my `_maybe_log_apply_stats` read the same Python tensor object twice
in immediate succession and saw inconsistent values (pre/post `out` showed different
`nan` count and `abs_max`). That can only happen if the buffer's contents are still being
written between reads — signature of a stream race, distinguishable from a kernel correctness bug.

---

## Steady-state performance (session 2 measured)

Same prompt, same TP=4, same hardware:

| | GPTQ-Int4 27B | Our FP8 27B (post-fix) |
|---|---|---|
| Decode tok/s | 6.114 | **6.025** (−1.5%) |
| Prefill (16-tok prompt) | 0.16 s | **0.17-0.20 s** (+15%) |
| Wall 300-tok generation | 49.22 s | 49.96 s (+1.5%) |

**Steady-state inference is essentially at parity.** No Plan A (WMMA) needed for typical
interactive use. Hand-written naive CUDA-core kernels are fast enough at the M values
real workloads hit. (See `bench_profile_scale.py` for the microbenchmark that grounded this.)

---

## Cold-start (Triton autotune / FLA Mamba kernel compilation)

The 14-min init engine on first deploy is **dominated by Triton autotune**, NOT our slow GEMM kernel.

Microbenchmark evidence (`bench_profile_scale.py`):
- Our FP8 kernel: 24 s total per profile_run forward at M=4096
- cuBLAS FP16 reference at same shapes: 0.6 s total
- So GEMM is 40× slower than cuBLAS, BUT init engine is 840 s → **GEMM is only ~3% of the cost**
- The other ~97% (~816 s) is Triton autotune + FLA Mamba compile

Scaling experiment proved autotune dominance:
- max_model_len=1024 init engine = 46 s (M^2.10 scaling would predict ~52s, matches)
- max_model_len=4096 init engine cold = 840 s
- max_model_len=8192 init engine warm = 93 s (cache hit dominant)

---

## Cache mount discovery and operational recipe

**The Triton kernel cache lives at `/tmp/torchinductor_root/` inside the docker container** — ephemeral by default,
wiped on container exit. We added a persistent mount in `run_docker.sh`:

```bash
-v "$HOME/.cache/vllm-torchinductor:/tmp/torchinductor_root"
```

After one cold deploy (~14 min), the host directory accumulates ~2600+ compiled cubin files (~1.6 GB).
Subsequent restarts hit cache for these kernels → init engine drops to ~1 min.

Critical observation (session 2 end): **the cache state ACCUMULATES across all max_model_len configurations.**
Each run at any config (1024, 2048, 3072, 4096, 8192) compiled additional kernels that benefit all future runs at any config.
After several mixed-config runs today, M=4096 first-curl response time dropped from 8 min (warm restart, partial cache) to 30 sec (warm restart, fully populated cache).

### Operational recipe for production

1. **One-time cold deploy:** pay ~14-22 min for cold init engine + first-curl autotune; cache writes to host
2. **Optional: run `prewarm.py`** with expected prompt-length buckets to populate cache for typical shapes
3. **Open to traffic** — steady-state inference is GPTQ-class
4. **Subsequent restarts:** ~1-2 min init engine, ~30 sec first prompt at any cached shape
5. **max_model_len is essentially free** after cache is warm — pick whatever fits your real workload

---

## Code state at end of session 2

Files modified (this directory):
- **`fp8_dequant.cu`** — `V100_FP8_STREAM` macro + 7 kernel-launch fixes (the bug fix)
- **`serve_fp8_v100.py`** — monkey-patch wrapper; added instrumentation infrastructure (toggleable via env vars)
- **`run_docker.sh`** — added `GPUS=`, `PORT=` env vars; added `/tmp/torchinductor_root` mount; added `attn-test`, `dump-hashes` subcommands
- Memory at `/home/kumphanartd/.claude/projects/-home-kumphanartd-vllm/memory/` — updated project state, added lesson about PyTorch C++ extension streams

Files added this session:
- **`test_attention_shapes.py`** — offline kernel correctness test against torch reference; the validation that proved kernel was fine in isolation (100/100 pass)
- **`dump_offline_hashes.py`** — offline emulator that generates per-rank shard hashes; used for the 20/20 hash diff that ruled out loader bugs
- **`bench_profile_scale.py`** — microbenchmark that proved slow-GEMM hypothesis wrong (24s of 840s)
- **`prewarm.py`** — drives endpoint with prompt-length buckets to populate per-shape Triton autotune
- **`perf_harness.py`** — drives endpoint at varying max_tokens with prefill+decode linear fit
- **`AIAGENT_ENV.md`** — frozen reference of production stack

Env vars supported by `serve_fp8_v100.py` (all default-off for production):
- `VLLM_V100_FP8_DEBUG_SHAPES` (off|mismatch|full) — log per-Linear weight/scale shapes at PWAL
- `VLLM_V100_FP8_DEBUG_APPLY` (off|on) — per-call x/out finite-stats; triggers on FIRST/DECODE/WARN/BAD/CAST
- `VLLM_V100_FP8_APPLY_WARN` (default 1000) — warn threshold for output abs_max
- `VLLM_V100_FP8_APPLY_MAG` (default 10000) — bad threshold for output abs_max
- `VLLM_V100_FP8_HASH_LAYERS` (off|on) — hash dump per (layer, rank) for selected layers

`run_docker.sh` env vars:
- `GPUS=4,5,6,7` (default `all`) — docker GPU isolation
- `PORT=8001` (default 8000) — host port for serve

---

## Open items / deferred

1. **WMMA / Phase A.4** — Properly scoped now: only relevant for long-context prefill (M >> 100) or batched serving (max_num_seqs > 1). Not needed for typical interactive workload. Deferred indefinitely.

2. **The "M=4096 warm-restart first-curl was slow" anomaly** — between two M=4096 warm-cache runs, the second was fast and the first was slow. Almost certainly cache-state-progression (additional intermediate kernels compiled by M=3072/8192 runs in between). Not worth investigating further unless re-emerges.

3. **`Fp8MoEMethod` not patched** — if we ever serve an FP8 MoE model with FP8 expert weights, we'd need to extend the patch similarly. Not in scope for tested models.

4. **A.3 atomicAdd is not bit-deterministic** — across runs FP atomicAdd reorders. Single-run reproducibility is ULP-level FP16 noise; cross-run can differ by a few ULPs. Fine for serving, not bit-stable.

5. **`--enforce-eager` required** — torch.compile / CUDAGraph paths in vllm assume sm_80+ optimizations. Can't drop on V100.

---

## Lessons saved to memory (for future projects)

- **PyTorch C++ extension CUDA streams**: Custom CUDA kernels MUST launch on `at::cuda::getCurrentCUDAStream()`. Default stream 0 is a silent race bomb at TP>1 with NCCL. `C10_CUDA_KERNEL_LAUNCH_CHECK()` only checks launch errors, does NOT synchronize.
- **GPT peer-review pattern**: this user relays my analyses to GPT for peer review; engage carefully and validate claims with code/data. Multiple cross-checks today caught at least 2 of my overclaims.
- **Measure before optimizing**: the WMMA / Plan B perf work nearly happened because we believed slow-GEMM dominated init engine. Microbenchmark + scaling experiment proved that wrong before any code was written. Saved ~1-2 weeks of misdirected work.

---
## Session 1 (earlier 2026-05-23) — historical record below
---


---

## Files in this directory

### Kernel (CUDA)
- **`fp8_dequant.cu`** — all CUDA kernels in one file:
  - Phase 1: `fp8_e4m3_to_fp16` (proper sign/exp/mantissa conversion)
  - Phase 2: `fp8_e4m3_to_fp16_scaled` (per-group 1D scale)
  - Phase 4: `fp8_e4m3_to_fp16_block_scaled` (2D 128×128 block scale, what real models use)
  - Phase 5: `fp8_w8a16_gemm` (naive fused dequant+matmul, CUDA cores)
  - Phase A.1: `fp8_w8a16_gemm_a1` (A.1 = vectorized uint4 W loads)
  - Phase A.2: `fp8_w8a16_gemm_a2` (A.1 + M-tiling BLOCK_M=8)
  - Phase A.3: `fp8_w8a16_gemm_a3` (A.1 + K-axis CTA splitting via FP32 atomicAdd)
  - All launches have `C10_CUDA_KERNEL_LAUNCH_CHECK()` after them.

### Python module
- **`fp8_w8a16_module.py`** — `FP8W8A16Linear(nn.Module)`:
  - Drop-in replacement for `nn.Linear`
  - Auto-dispatches kernel variant by M (decode → A.3 K=8, prefill → A.2, etc.)
  - Handles 3D `[batch, seq, dim]` input, optional bias

### vLLM integration
- **`serve_fp8_v100.py`** — runtime monkey-patch wrapper:
  - Patches `Fp8Config.get_min_capability` → 70
  - Patches `Fp8LinearMethod.__init__` → force `use_marlin=False`, `use_deep_gemm=False`
  - Patches `Fp8LinearMethod.process_weights_after_loading` → skip Marlin layout, keep [N,K] FP8 + [Nb,Kb] FP16
  - Patches `Fp8LinearMethod.apply` → dispatch to our kernel; **fail-closed** for non-block FP8 on V100
  - Logs `[serve_fp8_v100 pid=X]` banner to prove patches ran in every TP worker

### Test scripts
- `test_fp8.py` — Phase 1 + 2 dequant bit-exactness
- `test_phase5_matmul.py` — naive GEMM correctness vs reference
- `test_phase_a1.py` / `test_phase_a2.py` / `test_phase_a3.py` — A/B/C optimization benchmarks
- `test_phase_b_module.py` — module integration (3 checks)
- `test_phase_c_mlp.py` — 3-GEMM SwiGLU MLP block validation
- `inspect_fp8_model.py` — model file format inspector
- `test_wrapper_imports.py` — verify monkey-patches land on real vllm classes
- `dev_sanity.py` — dev image health check

### Infrastructure
- **`Dockerfile`** — `vllm-v100-fp8-test:cu124` (kernel correctness work)
- **`Dockerfile.dev`** — `vllm-v100-dev:cu128` (vLLM 0.18 + torch 2.10 + cu128)
- **`run_docker.sh`** — wrapper script with these subcommands:
  - `build`, `build-dev` — build images
  - `test`, `inspect`, `matmul`, `phaseb`, `a1`, `a2`, `a3`, `mlp` — kernel/correctness tests
  - `dev-shell`, `dev-test` — interactive / one-shot in cu128 image
  - `serve <script.py> [args]` — run vLLM with port 8000 forwarded
  - Passes `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` and `--shm-size=8g` for TP>1 stability
  - Mounts persistent caches: `~/.cache/vllm-triton`, `~/.cache/vllm-torch`, `~/.cache/vllm-extensions`

### Documentation
- **`AIAGENT_ENV.md`** — frozen reference of aiagent's verified-working production stack (versions, flags, why each is set)
- **`README.md`** — Phase 1 hello-world readme

---

## How to reproduce

### One-time setup
```bash
cd /home/kumphanartd/vllm/experiments/v100_fp8_test

# Build the dev image (~10 min, downloads ~6 GB of wheels)
./run_docker.sh build-dev

# Verify the image works
./run_docker.sh dev-test dev_sanity.py
```

### Serve a model (always inside tmux to survive SSH drops)
```bash
tmux new -s vllm

# 4B at TP=1 — fastest path to a working serve
./run_docker.sh serve serve_fp8_v100.py \
    --model /mnt/models/Qwen3.5-4B-FP8 \
    --quantization fp8 --dtype float16 --enforce-eager \
    --attention-backend TRITON_ATTN --max-num-seqs 1 \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.50 \
    --max-model-len 8192 --no-enable-chunked-prefill \
    --disable-custom-all-reduce --host 0.0.0.0 --port 8000

# 27B at TP=4 — larger model, requires 4 GPUs for memory
./run_docker.sh serve serve_fp8_v100.py \
    --model /mnt/models/Qwen3.6-27B-FP8 \
    --quantization fp8 --dtype float16 --enforce-eager \
    --attention-backend TRITON_ATTN --max-num-seqs 1 \
    --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 4096 --no-enable-chunked-prefill \
    --disable-custom-all-reduce --host 0.0.0.0 --port 8000
```

### Test from another terminal
```bash
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "/mnt/models/Qwen3.5-4B-FP8",
         "prompt": "Hello", "max_tokens": 30, "temperature": 0.0}' \
    | python3 -m json.tool
```

### Cold-start timings (V100, vllm 0.18, --enforce-eager)
- 4B + TP=1: ~14 min warmup, ~12 tok/s decode
- 27B + TP=4: ~14 min warmup (TP parallelizes autotune), TP=2 hits KV cache OOM, TP=1 doesn't fit
- Cache mounts persist Triton+kernel binaries but **most of the 14 min is forward-pass compute, not Triton compile** — can't be eliminated without WMMA (faster compute) or skipping warmup

---

## Performance (CUDA-core kernel, no tensor cores yet)

| Workload | Our kernel | cuBLAS FP16 ref (same hardware) |
|---|---|---|
| Decode M=1 | ~12 tok/s (we win — cuBLAS has launch overhead) | ~10 tok/s |
| Prefill M=128 | ~3 tok/s | ~10 tok/s (cuBLAS wins ~3× via tensor cores) |

So we are competitive at decode, behind at prefill. Closing the prefill gap = WMMA (Phase A.4).

---

## Known issues / open items

0. **🔴 27B + TP=4 numerical bug (TOP PRIORITY for next session).**

   ### Symptom
   - 4B + TP=1: coherent text ✅
   - 27B + TP=4 + FP8 (our patches): server returns 200 OK, but text is
     `"!!!!!!!!!!!!!!!!"` × max_tokens (logits dominated by one token, or NaN/Inf)
   - 5 tok/s steady-state throughput → kernel is *running*, just wrong numbers

   ### Critical comparison data (2026-05-23 session)
   - **27B + TP=4 + FP16 (stock vllm, no patches)**: ✅ generates correct
     Rayleigh-scattering paragraph at 6.5-6.7 tok/s decode
   - Same hardware, same TP=4, same Triton attention + Mamba kernels
   - **This eliminates**: TP sharding in general, Triton attention,
     Mamba/FLA kernels, NCCL all-reduce, Qwen3-Next architecture support
   - **Confirms**: the bug is specifically in **how our FP8 path handles
     per-shard weight + scale relationships at TP > 1**

   ### Timing comparison (27B + TP=4)
   ```
                       FP16 (stock)   FP8 (our patches)
   init engine took    505.70 s        849.61 s    (1.7× slower)
   loading weights      23.67 s          6.28 s    (cached on 2nd run)
   weights per GPU      13.01 GiB        7.26 GiB  (FP8 ~half)
   KV cache budget      78K tokens       221K tokens
   ```
   - The 1.7× warmup slowdown is consistent with FP8 path using CUDA cores
     vs cuBLAS tensor cores for the matmul during profile forwards
   - Will improve with WMMA (Phase A.4)

   ### Strong suspect — scale sharding for row-parallel layers
   - **Column-parallel** layer (e.g. `gate_proj`, `up_proj`, fused QKV):
     N is split → weight `[N/TP, K]`, scale `[N/(TP·128), K/128]`
   - **Row-parallel** layer (e.g. `down_proj`, `out_proj`):
     K is split → weight `[N, K/TP]`, scale `[N/128, K/(TP·128)]`
   - Our `_our_apply` reads `layer.weight_block_size` (= [128, 128] from config)
     and assumes both dims of scale are 128-block-aligned
   - **If vLLM doesn't shard `weight_scale_inv` consistently with `weight`**,
     or sets `weight_block_size` to the original config value while the
     actual scale tensor was sharded differently, our kernel reads wrong
     scale values → silent corruption

   ### Diagnostic plan for next session
   1. **First, isolate TP vs 27B**: run 4B + TP=2 with our patches.
      - If 4B+TP=2 generates correct text → bug is 27B-specific (architecture-level)
      - If 4B+TP=2 produces "!" too → bug is purely the TP path

   2. **Inspect per-shard layer state in our patched `process_weights_after_loading`**:
      add print statements showing:
      ```
      layer.weight.shape, layer.weight_scale_inv.shape,
      layer.input_size_per_partition, layer.output_size_per_partition,
      layer.weight_block_size, computed block_h/block_w from scale shape
      ```
      Run with `--tensor-parallel-size 4` and confirm: for EVERY layer the
      derived `block_h = N / scales.shape[0]` and `block_w = K / scales.shape[1]`
      both equal 128. If any layer comes out with block_w != 128 (likely
      `down_proj` at TP=4 with K=17408/4=4352 vs scale.shape[1]=136 → 4352/136=32),
      that's the bug.

   3. **Compare layer-by-layer FP8 output vs FP16 reference**:
      hook each Linear's forward to dump output stats (mean, std, abs_max, NaN count)
      while running the same prompt through both 27B+TP=4+FP16 and 27B+TP=4+FP8.
      Find the first layer where outputs diverge → that's where the bug lives.

   ### Honest performance expectation after fix
   Even with bug fixed, our FP8 at TP=4 likely runs at **~3-5 tok/s decode** vs
   FP16's ~6.7 tok/s (no tensor cores in our kernel = 30-50% slower). The win
   is **2× weight memory savings** and 3× larger KV cache budget, not speed.
   Phase A.4 (WMMA) closes that speed gap.

1. **`Fp8MoEMethod` not patched.** If we ever serve an FP8 MoE model with FP8 expert weights (not the case for Qwen3.5-4B or Qwen3.6-27B which are dense MoE-gate-bf16), we'd need to extend the patch to `Fp8MoEMethod` analogously.

2. **A.3 atomicAdd is not bit-deterministic** across runs (FP atomicAdd reorders). Single-run reproducibility is ULP-level FP16 noise; cross-run can differ by a few ULPs. Fine for serving, not bit-stable.

3. **BF16→FP16 scale cast** was 0-loss for the Qwen3 models tested (`max precision loss = 0.000000e+00`). NOT guaranteed lossless in general — different scale magnitudes could lose mantissa precision. Validation step in `inspect_fp8_model.py` reports actual error.

4. **Patches propagate via vLLM 0.18 worker re-exec.** Verified working but not robust to future vLLM changes (Ray executor, forkserver, different launch styles). If it breaks, write a vLLM plugin or `sitecustomize.py` instead.

5. **`--enforce-eager` required.** torch.compile / CUDAGraph paths in vllm assume sm_80+ for various optimizations. We can't drop this flag on V100.

6. **First-inference Triton compile is per-shape.** Any new prompt shape vllm hasn't seen triggers up to 5-15 min of recompilation. Server-side this is bounded by `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` (set by `run_docker.sh`). Client-side curl needs `--max-time 1200+`.

7. **Historical note, superseded 2026-05-24:** this section originally said
   MoE FP8 models were untested. That is no longer true:
   Qwen3.5-122B-A10B-FP8 now loads and serves on 8x V100. See the Stage 2A
   record at the end of this log.

---

## Next session: FP16 baseline comparison (optional, do before or alongside WMMA)

To give the V100 FP8 numbers context, run the same model architecture as
plain FP16/BF16 (no `--quantization fp8`, no monkey-patches) and compare:

1. `init engine took XXX seconds` — expected similar (~14 min)
2. Steady-state decode tok/s — expected FP16 is 3-5× faster (cuBLAS tensor cores)
3. KV cache room — FP16 weights take 2× memory, so KV shrinks accordingly

Requires a 27B BF16/FP16 version of Qwen3-Next on disk (not currently present
under `/mnt/models/`). To find or download:

```bash
ls /mnt/models/ | grep -i qwen | grep -v -i -E 'fp8|gptq|int4'
# If absent, fetch via aiagent's hf downloader. ~54 GB, ~30 min on this network.
```

Then serve with stock vllm (no wrapper):

```bash
docker run --rm -it --gpus all \
    -v /mnt/models:/mnt/models:ro \
    -v "$HOME/.cache/vllm-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-torch:/root/.cache/torch" \
    -p 8000:8000 --shm-size=8g \
    vllm-v100-dev:cu128 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model /mnt/models/Qwen3.6-27B \
        --dtype float16 --enforce-eager \
        --attention-backend TRITON_ATTN --max-num-seqs 1 \
        --tensor-parallel-size 4 --gpu-memory-utilization 0.85 \
        --max-model-len 4096 --no-enable-chunked-prefill \
        --disable-custom-all-reduce --host 0.0.0.0 --port 8000
```

The decode tok/s difference is the headline number for "what did our V100 FP8
work buy us?" — typically you trade ~3-5× speed (tensor cores) for ~2× memory
footprint savings (FP8 storage).

## Next session: Phase A.4 — WMMA tensor cores

### Why
At prefill (M≥32), our CUDA-core kernel runs at ~3 tok/s vs cuBLAS FP16 at ~10 tok/s. The gap closes by ~3× if we use V100 tensor cores (HMMA.884) instead of CUDA-core FMA.

### Design sketch (already discussed in this session)
- Use `nvcuda::wmma` C++ API (Volta exposes HMMA.884 via 16×16×16 fragments)
- Inner loop:
  1. Load FP8 weight bytes from HBM into registers (vectorized uint4 — keep from A.1)
  2. Dequant FP8 → FP16 in registers
  3. **Write FP16 to shared memory in WMMA-compatible layout** ← the hard part
  4. `wmma::load_matrix_sync(a_frag, ..., shared_layout)`
  5. `wmma::mma_sync(c_frag, a_frag, b_frag, c_frag)`
  6. Repeat for next K-chunk
- Shared memory tile layout needs swizzling/padding to avoid bank conflicts
- Tile shape: probably 32×32×16 or 64×64×16 per CTA
- Double-buffering manually (no `cp.async` on Volta)
- Consider switching to Marlin-style byte-perm dequant + scale-bias-fusion if dequant becomes the bottleneck feeding tensor cores

### Effort estimate
1-2 weeks focused work. Mostly debugging fragment layout (opaque from C++) and bank conflicts (invisible without nsight-compute). Final validation against existing test infrastructure (Phase 1-5, A/B/C, MLP).

### Pre-work for next session
- Read `csrc/quantization/marlin/marlin_template.h` for tile/pipeline patterns to inform but not copy
- Read `csrc/quantization/gptq/q_gemm.cu` — its CUDA-core structure is the closest reference for V100
- Confirm V100 WMMA quirks via NVIDIA's CUDA C++ Programming Guide section on Volta WMMA

---

## Quick links

- This repo: `/home/kumphanartd/vllm` (branch: customize-v100)
- Experiments dir: `/home/kumphanartd/vllm/experiments/v100_fp8_test/`
- Verified env: see `AIAGENT_ENV.md`
- Models: `/mnt/models/Qwen3.5-4B-FP8`, `/mnt/models/Qwen3.6-27B-FP8`
- aiagent's prod baseline: `/home/aiagent/vllm-env/bin/python -m vllm.entrypoints.openai.api_server ...` (see AIAGENT_ENV.md for full command)

## 2026-05-24: Qwen3.5-122B-A10B-FP8 on 8x V100, Stage 2A complete

`/mnt/models/Qwen3.5-122B-A10B-FP8` now loads and serves coherently on 8x V100
with vLLM 0.18.0 through `src/fp8_w8a16_sm70/vllm_serve.py`.

Implemented MoE support:
- Bypass stock `Fp8MoEMethod` backend selection on Volta. Stock vLLM raises
  `No FP8 MoE backend supports deployment configuration` on `sm_70`.
- Keep block-FP8 MoE weights in model-loaded layouts and cast scales to FP16.
- Route MoE expert GEMMs through existing V100 FP8 W8A16 dense kernels.
- Preserve legacy path with `VLLM_V100_FP8_MOE_ACTIVE_LIST=0`.
- Default-on Stage 2A optimization:
  `VLLM_V100_FP8_MOE_ACTIVE_LIST=1`.

Stage 2A optimization details:
- Before active-list, the MoE fallback scanned all 256 local experts per layer
  call with `mask.any().item()`.
- Profile showed `mask_sync` around 12-18 ms per MoE call, roughly 45% of MoE
  layer wall time.
- Active-list fix uses:

```python
expert_iter = torch.unique(local_topk[local_topk >= 0]).tolist()
for local_expert in expert_iter:
    ...
```

- This intentionally does one CPU/GPU sync per MoE call, not one `.item()` per
  active expert.
- Profile after fix: `empty_iters=0`, `skipped_experts≈245`, MoE wall dropped
  from roughly 28-34 ms/layer-call to roughly 9-12 ms/layer-call in diagnostic
  mode.

Validation:
- Coherent deterministic completions:
  - `"The capital of France is Paris..."`
  - V100 paragraph prompt with `temperature=0`.
- Stage 2A warmed profile-off 32-token prompt:
  - `real 0m12.87s` for 32 output tokens, roughly 2.5 tok/s.
- Legacy scan with `VLLM_V100_FP8_MOE_ACTIVE_LIST=0` returns to roughly
  0.7-0.9 tok/s.
- 32k context server starts and serves the local AI agent:
  - `--max-model-len 32768`
  - `--max-num-batched-tokens 32768`
  - `--max-num-seqs 1`
  - KV cache memory around 4.66 GiB per GPU.
  - GPU KV cache size around 101,904 tokens.
  - Maximum concurrency for 32,768 tokens/request reported as 11.70x.
  - Prompt throughput around 750-836 tok/s on the agent request.
  - Decode throughput around 2.5-2.7 tok/s.

Useful serve command for real agent use:

```bash
cd ~/vllm-fp8-w8a16-sm70
GPUS=all PORT=8001 \
./docker/run_docker.sh serve \
    --model /mnt/models/Qwen3.5-122B-A10B-FP8 \
    --served-model-name qwen-v100 \
    --quantization fp8 --dtype float16 --enforce-eager \
    --attention-backend TRITON_ATTN --tensor-parallel-size 8 \
    --max-num-seqs 1 --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.80 --max-model-len 32768 \
    --no-enable-chunked-prefill --disable-custom-all-reduce \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

Use `--served-model-name qwen-v100` or configure the client to send the exact
model path. A local client request with `model=gemini-2.5-flash` correctly
returned 404 because that served model name does not exist.

Diagnostic env vars:
- `VLLM_V100_FP8_MOE_PROFILE=1`
- `VLLM_V100_FP8_MOE_PROFILE_EVERY=64`
- `VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1` (Stage 2B experimental)
- `VLLM_V100_FP8_DECODE_BREAKDOWN=1`
- `VLLM_V100_FP8_DECODE_BREAKDOWN_EVERY=32`

Do not leave decode breakdown enabled for production; it records hundreds of
CUDA events per token and spams logs. It is for measurement only.

Full decode breakdown instrumentation:
- Hooked only coarse Qwen3-Next/Qwen3.5 modules to avoid double-counting:
  - `Qwen3NextSparseMoeBlock`
  - `Qwen3NextGatedDeltaNet`
  - `Qwen3_5GatedDeltaNet` mapped as GDN
  - `Qwen3NextAttention`
  - `LogitsProcessor`
- Explicitly does not hook decoder-layer parents.
- Uses `register_forward_pre_hook(..., with_kwargs=True)` and
  `register_forward_hook(..., with_kwargs=True)` because GDN/attention are
  called with keyword args (`hidden_states=...`).
- Counts actual decode tokens at `LogitsProcessor`.

Steady 32k-context breakdown from the local AI-agent run:

| Section | ms/token | Share |
|---|---:|---:|
| Qwen3NextSparseMoeBlock | ~268-278 | ~71-74% |
| Qwen3NextGatedDeltaNet | ~59-66 | ~16-17% |
| Qwen3NextAttention | ~27-30 | ~7-8% |
| LogitsProcessor | ~0.6-0.9 | ~0.2% |
| Other residual | ~5-22 | ~1-6% |
| Total | ~377-382 | ~100% |

Interpretation:
- MoE is the dominant decode cost by a wide margin.
- FlashAttention-V100 is not the next large lever; full attention is only
  around 7-8% of decode.
- GDN/linear-attention is second but still much smaller than MoE.
- Stage 2B MoE optimization is technically justified.

Expected Stage 2B upside math:
- Current: ~378 ms/token, roughly 2.6 tok/s.
- If MoE drops from ~270 ms/token to ~160 ms/token:
  total becomes ~268 ms/token, roughly 3.7 tok/s.
- If MoE drops to ~100 ms/token:
  total becomes ~208 ms/token, roughly 4.8 tok/s.
- Realistic target range for a good Stage 2B: 4-5 tok/s.

Stage 2B starting point:
- Optimize `_our_moe_apply` / expert dispatch, not attention.
- Current path is expert-major Python looping:
  `index_select -> w13 GEMM -> activation -> w2 GEMM -> index_add_`.
- Active-list removed empty expert scans, but remaining cost is per-active-
  expert work, tiny GEMM launch overhead, `nonzero`, `index_select`, and
  `index_add_`.
- Reasonable Stage 2B options:
  1. Sort/permutation grouping by expert id, then contiguous slices. This can
     reduce `nonzero`/indexing overhead and make launch patterns cleaner.
  2. Batched/grouped MoE CUDA kernel for all active experts. Best upside, more
     work.
  3. Keep current FP8 dense GEMM kernels as inner kernels initially; only fuse
     routing/scatter after measurement proves it is worth it.

Stage 2B first implementation attempt:
- Added an experimental grouped-routed A.3 CUDA kernel:
  `fp8_w8a16_grouped_routed_gemm_a3`.
- Purpose: reduce MoE decode GEMM launch fanout from two launches per active
  expert to one grouped `w13` launch plus one grouped `w2` launch per MoE layer
  call.
- Python path is guarded by:
  `VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1`.
- Default remains off until V100 validation. The Stage 2A active-list path is
  still the production default.
- Added bench controls for the first measurement session:
  - `VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=32` keeps grouped GEMM scoped
    to decode/small-batch routes and avoids long-prefill WMMA regressions.
  - `VLLM_V100_FP8_MOE_GROUPED_K_SPLIT=auto|1|2|4|8` supports hard-pinned
    maximum `k_split` sweeps instead of guessing whether the natural dispatcher
    chose 8 or 4. The per-GEMM selector falls back to a lower valid split when
    a small K cannot satisfy the requested split; on Qwen3.5-122B-A10B-FP8 the
    first grouped log showed `hidden=3072`, `intermediate=128`, so natural
    dispatch is `w13_k_split=8`, `w2_k_split=1`.
  - `VLLM_V100_FP8_MOE_GROUPED_LOG_ONCE=1` prints the first grouped dispatch:
    hidden size, intermediate size, route slots/count, selected `k_split`s,
    and whether `view(torch.uint8).contiguous()` was zero-copy.
- Local static checks passed (`py_compile`, `git diff --check`). JIT compile
  could not be run in the current shell because `torch` is not installed there;
  validate inside the Docker/vLLM environment before enabling in production.

Stage 2B first V100 measurement (2026-05-24):
- Environment: 8x V100, `/mnt/models/Qwen3.5-122B-A10B-FP8`, TP=8,
  `max_model_len=32768`, `max_num_seqs=1`, profile off unless noted.
- Coherence gate passed with `VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1`:
  prompt `"The capital of France is"` produced coherent greedy Paris text.
- JIT compile passed inside Docker/vLLM.
- First grouped dispatch log for Qwen3.5-122B-A10B-FP8:
  `M=1`, `route_slots=8`, `route_count=8`, `hidden=3072`,
  `intermediate=128`, `block=(128,128)`, uint8 views were zero-copy.
- Important sweep detail: `intermediate=128` means `w2` can only use
  `k_split=1`. The env sweep pins a maximum split per GEMM, so rows below are
  `w13_k_split=N`, `w2_k_split=1`.

Profile-off short sweep, same prompt, one warmup, then 32/64 output tokens:

| Variant | w13 split | w2 split | 32 tok wall | 64 tok wall | Fit decode |
|---|---:|---:|---:|---:|---:|
| Stage 2A active-list baseline | per expert | per expert | 13.30 s | 25.40 s | 2.646 tok/s |
| Grouped routed | 8 | 1 | 7.91 s | 14.03 s | 5.225 tok/s |
| Grouped routed | 4 | 1 | 7.93 s | 14.12 s | 5.169 tok/s |
| Grouped routed | 2 | 1 | 7.84 s | 14.00 s | 5.197 tok/s |
| Grouped routed | 1 | 1 | 7.93 s | 14.06 s | 5.219 tok/s |

Interpretation:
- Grouped routed GEMM is a real Stage 2B win: roughly 1.98x decode throughput
  versus Stage 2A on the short sweep.
- `k_split` is not very sensitive for this shape; 8 was nominally fastest, but
  1/2/4 are within measurement noise. Natural `auto` chooses 8 for `w13` and 1
  for `w2`, which is a reasonable default.
- The "atomics are waste" hypothesis did not show a clear win; `k_split=1`
  was near-tied, not clearly better.

Profile-on grouped `w13=8,w2=1` notes:
- Profile-on wall is much slower and should not be used for throughput
  decisions (`32` tokens took 24.06 s, 1.33 tok/s).
- Cumulative profile logs are polluted by model warmup/prefill at the beginning;
  read later lines where `active_hist` is dominated by `8:*` decode calls.
- Late cumulative grouped MoE profile around `calls=1600`:
  `avg_wall=5.647ms` per MoE call, `routing=0.036ms`, `mask_sync=0.469ms`,
  `index_select=0.317ms`, `w13_gemm=0.981ms`, `activation=0.269ms`,
  `w2_gemm=0.577ms`, `scatter=0.904ms`.
- Remaining grouped-MoE hotspots are now scatter and `w13` GEMM, not empty
  expert scans.
- Profile output currently reports nonsensical cumulative `routed_items` during
  warmup because prefill/warmup calls enter the same aggregate; fix/segment the
  profile accounting before using those counters as facts.

Long profile-off confirmation:
- Same prompt, one warmup, `max_tokens=200`; the model stopped naturally before
  200 tokens in both cases, so compare actual completion-token throughput.
- Stage 2A active-list baseline: `151` completion tokens in `57.66 s`,
  `2.619 tok/s`.
- Grouped routed `auto` (`w13_k_split=8`, `w2_k_split=1`): `148` completion
  tokens in `30.41 s`, `4.866 tok/s`.
- Long-run result confirms the short-sweep conclusion: grouped routed decode is
  about `1.86x` faster than Stage 2A on this prompt and server envelope.

Stage 2C plan (after v0.2.0):
1. Default-on grouped routed GEMM with
   `VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=32` safety guard.
   **DONE in v0.2.0.**
2. Fix grouped profile accounting. Cumulative `routed_items` and per-section
   totals currently mix warmup/prefill/decode; segment by phase or shape
   before trusting further optimization signals.
3. Optimize scatter/indexing. Profile shows `index_add_` (~0.9ms) and
   `nonzero`/`index_select` materialization are the remaining grouped-MoE
   hotspots. Candidates: fuse scatter into the `w2` epilogue (write directly
   to `out[token_idx]` via atomicAdd), FP32 accumulator on `out`, fold
   `route_w` into the `w2` epilogue. Not CUDA graphs and not grouped-WMMA
   yet -- both are larger and farther from the measured signal.
4. Cache small per-layer constants. `expert_map.to(device=...)` is cheap per
   call but accumulates over 48 MoE calls x every decode token; cache once at
   PWAL. Keep the zero-copy uint8 view assumption documented.
5. Harden grouped CUDA kernel. The partial-CTA `__syncthreads()` pattern is
   inherited from A.3 and safe for current Qwen shapes (`N` is
   `BLOCK_N_A3`-aligned), but should be guarded before broadening model
   support.

Target: push grouped long-run decode from ~4.9 tok/s toward 5.5+ tok/s
without touching attention.

DeepSeek V4 note:
- DeepSeek-V4-Flash is not a small extension of this work. Official Flash uses
  FP4 expert weights plus FP8 other weights and likely vLLM 0.20+ DeepSeek-V4
  code. Current repo supports block-FP8 W8A16, not FP4/W4A16 experts.
- Treat DeepSeek V4 as a separate project, not Stage 2B.

## Stage 2C (2026-05-25): profile-mode hygiene + layer caches

Result: +5-7% production decode on Qwen3.5-122B-A10B-FP8 TP=8 vs v0.2.0
(4.866 -> 5.093 tok/s warmed-mean of 4 curls, profile-off, default config).
Attribution: layer caches, not the route-prep fast path.

Done over two Claude/GPT review rounds:

Step A -- profile segmentation. Stats keyed by
`(phase, M, route_slots, grouped, fast)`. Phase tag derived as
`decode if M <= MOE_DECODE_M_MAX and route_slots <= MOE_GROUPED_MAX_ROUTE_SLOTS
else prefill`. Per-rank warmup-skip (200 calls default) before recording.
Reports distinguish `cuda_event_sections_ms` from `wall_sections_ms` and
compute `unattributed = call_wall - both`. Top-6 buckets and top-3 layers
per cycle.

Step B -- profile-mode hygiene. `active_experts_stat` (torch.unique on
local_expert_ids) gated behind `VLLM_V100_FP8_MOE_PROFILE_ACTIVE_STAT=1`,
default off (was ~6% of the MoE call when measured). Finer wall timers
`py_inner_loop`, `py_dispatch_w13`, `py_dispatch_w2` added inside the
grouped path so the unattributed bucket can be further split.

Step C -- route-prep fast path. When `expert_map is None` (TP-replicated
experts, no EP filtering -- the Qwen3.5-122B-A10B-FP8 case), replace
`nonzero(local_topk >= 0)` + `local_topk[token_idx, route_idx].to(int64)`
+ `topk_weights[token_idx, route_idx].to(float16)` with
`local_topk.reshape(-1).to(int64).contiguous()` +
`topk_weights.reshape(-1).to(float16)` + a process-cached `token_idx`
keyed by `(M, K, device)`. Removed ~0.286 ms/call of GPU-stream work.

Step D -- free caches. `_get_layer_uint8_weights(layer)` caches
`layer.w13_weight.view(torch.uint8).contiguous()` and same for w2 as
`layer._v100_w13_u8` / `layer._v100_w2_u8`. `_get_layer_expert_map_dev`
caches `expert_map.to(device=...)` (dormant for this model where
`expert_map is None`). These caches run regardless of fast-path setting.

Decode-only profile bucket (`phase=decode, M=1, route_slots=8, grouped=1`)
on Qwen3.5-122B-A10B-FP8, `VLLM_V100_FP8_MOE_PROFILE=1`, steady state:

| Config | avg_wall | cuda_sections | wall_sections | unattributed |
|---|---:|---:|---:|---:|
| Step A (fast=0) | 1.802 ms | 0.810 | 0.207 | 0.785 |
| Step C+D (fast=1) | 1.740 ms | 0.210 | 1.381 | 0.149 |

Step C+D cuda-event delta -0.6 ms/call is real but the `wall_sections`
growth is mostly accounting (`py_inner_loop` double-counts the inner wall
timers it wraps). True `avg_wall` saving: 0.062 ms/call x 48 layers = 3
ms/token under profile-on.

Profile-off warmed A/B (4 curls per leg, `max_tokens=200`, mean):

| Config | warmed tok/s |
|---|---:|
| v0.2.0 baseline | 4.866 |
| Step C+D, fast_route_prep=1 | 5.027 |
| Step C+D, fast_route_prep=0 | 5.093 |

fast_route_prep=1 is ~1.3% slower than fast=0 in production despite the
CUDA-stream savings. Hypothesis: the cached `token_idx` tensor has a
long-lived GPU address that competes with other tensors' L2 footprint, or
the reshape + identity-cast + contiguous chain has slightly more Python
overhead than the gather it replaces at this decode shape, or Triton
autotune state diverged. Either way the data is clear.

Decision: ship default `fast_route_prep=0`. Keep Step C code behind env
flag (`VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1` opts in) for future models
where the pipeline may not absorb the saved GPU work. Keep all Step D
caches on (they deliver the actual +5-7% lift).

Decode-breakdown cross-check (profile-on, 32-token window):
- `Qwen3NextSparseMoeBlock` 145.8 ms/token (54.5%, calls=1440 -> 45/tok;
  later window 48/tok = 1 per layer)
- `Qwen3NextGatedDeltaNet` 69.2 ms/token (25.9%)
- `Qwen3NextAttention` 43.9 ms/token (16.4%)
- `LogitsProcessor` 17.2 ms/token (6.4%)
- Total: 267.5 ms/token = 3.74 tok/s under profile-on (consistent with
  per-call sync overhead).

The 145.8 ms `SparseMoeBlock` includes ~84 ms of `_our_moe_apply`
(48 x 1.74) plus ~62 ms of router/topk/combine/shared-expert wrapping
**outside** our profile coverage. That gap is now the largest single
target for Stage 2D.

Lessons:

- CUDA-event savings do not imply wall savings under per-call-sync profile
  mode. The CPU/GPU pipeline already overlaps Python launch with prior GPU
  work; removing kernels from the stream can leave wall unchanged if the
  sync still has to wait for downstream queued work. Profile-off A/B is
  the only trustworthy throughput proxy.
- Prior "5.65 ms/MoE call, scatter dominates" reading was warmup/prefill
  contamination. Segmented profile gave 1.74 ms/MoE call decode-only with
  scatter at 0.054 ms (3% of call). The plan was rewritten mid-Stage 2C
  on the corrected data; the original "fused scatter" lever was
  deprioritized.
- Two-round Claude/GPT peer review (in [[feedback-gpt-review-pattern]])
  caught a label-swap in GPT's first A/B analysis and a stream-overlap
  framing error in Claude's. Both were corrected before code landed.

Stage 2D candidates (deferred, not opened yet):

1. Instrument `Qwen3NextSparseMoeBlock` directly to break out router/topk
   vs `_our_moe_apply` vs combine/shared-experts. Target the ~60 ms/token
   wrapper gap.
2. GDN/FLA Mamba kernel work (60+ ms/token, 17-25% of decode).
3. Fused MoE kernel (subsumes w13 + activation + w2 + scatter into one
   launch). Only if (1) and (2) are exhausted.

Not Stage 2D:
- FlashAttention-V100 (self-attn 16% in profile-on, ~7-8% in profile-off;
  cu128 -> cu129 toolchain bump still not justified).
- CUDA graphs (`--enforce-eager` requirement).
- Deferred-sync profile mode (not needed once `py_inner_loop` separates
  wall from sync wait).

End of log.
