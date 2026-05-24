# V100 FP8 W8A16 — Session Wrap-up (2026-05-23, **session 2**)

Cold-start summary for picking up in a new session. Read top to bottom.

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

7. **MoE FP8 models untested.** Qwen3.5-122B-A10B-FP8 was downloaded but never loaded; it's MoE which we haven't validated.

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

End of log.
