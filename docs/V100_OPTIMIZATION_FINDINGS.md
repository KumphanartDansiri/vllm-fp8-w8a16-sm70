# V100 (sm_70) optimization findings — consolidated reference

Source-of-truth for the 2026-06-12/13 optimization arc (Opus 4.8 → Fable → Opus 4.8, with
Codex peer-review). Internal/technical; the publication version is derived from this. Every
claim here is measured and committed — commit hashes and `results/` paths are cited inline.

---

## TL;DR

Three optimization fronts examined on V100 + vLLM 0.21 (image `vllm-v100:vllm021-cu126`). One
real shippable win, two honest "already optimal / not worth it" closures:

| front | verdict | headline |
|---|---|---|
| **FP16 MoE decode** | **FIXED + shipped** | stock `BLOCK_K=128` decode default cratered Triton on sm_70; `BLOCK_K=64` → **4–9× e2e** (35B 15.6→65.9 single, 3.2→22.4/user @8) |
| **FP8 MoE decode** | already near-optimal, **no work** | GEMV-dominated; the w2 "anomaly" is an M=1-only under-occupancy artifact that self-heals at load |
| **Vision encoder (ViT)** | **baseline is the ceiling** | V100 SDPA already auto-selects the EFFICIENT (CUTLASS mem-efficient) kernel; padded FA-V100 is 2–2.7× slower |

The unifying principle (see §4) explains all three: V100 has fast tensor-core compute but none of
the async memory-feed hardware (`cp.async`/TMA) that sm_80→sm_100 added — so it wins in the
memory-bound/resident regime (decode GEMV) and is at-ceiling in the compute/tile-staging regime
(prefill, ViT). Our wins were all *moving work into V100's favorable regime*; the dead ends were
all attempts to win the regime later silicon was built to fix.

---

## 1. FP16 MoE decode — the real win (SHIPPED)

**Symptom:** stock vLLM FP16 MoE ran *slower than a same-class dense model* on V100 — backwards for
a sparse MoE. Qwen3.6-35B-A3B = 15.6 tok/s vs dense 27B = 37–41; gemma-4-26B-A4B = 10.9 vs dense
31B = 17.6 (TP4, cudagraph, ns=8).

**Diagnosis (3 steps, measure-driven):**
1. **`num_stages` hypothesis REFUTED.** A prior session traced the config-miss → `get_default_config`
   Ampere default `num_stages=4` and theorized the cp.async-less V100 hated deep staging. Measured:
   e2e sweep `num_stages {4,3,2}` was **flat** (15.57 every arm, bit-identical output).
   `results/moe_stages_ab_q35b_20260612_132040`, `..._g26b_20260612_140610`.
2. **Where the time goes** (`tools/moe_decode_microbench.py`): `fused_moe_kernel` = 98.9% of
   `fused_experts`, **90× off the memory floor** at M=1 (Triton sm_70 has no tensor-core `tl.dot`
   → FMA fallback + 16×-padded M tiles).
3. **The real lever = `BLOCK_SIZE_K`** (`tools/moe_decode_tile_sweep.py`): kernel time scales with
   BLOCK_K (64→632µs, 128→1450, 256→2300; N/warps/stages ~irrelevant) — register-spill on Volta.
   **Stock `get_default_config` picks `BLOCK_K=128` exactly and only in the decode branch (M≤64)**;
   prefill already gets 64. So only decode is hit. (fused_moe.py:1274 @0.21.)

**Fix + validation:**
- Single-stream e2e: **35B 15.57→65.87 (4.23×)**, gemma **10.90→43.62 (4.00×)**, output bit-identical.
  `results/moe_stages_ab_*_151642`.
- M=1..16 sweep: stock degrades ~linearly with M (spill traffic) → win **grows** with batch, up to
  ~9× at M=16. Per-M optima: M≤4 = `16/32/64 w4 s2`, M≥8 = `16/128/64 w8 s2`.
  `results/moe_decode_msweep_*`.
- 8-user e2e: 35B per-user **3.16→22.41 (7.1×)**, agg 24.9→163.8; gemma per-user 3.6→19.2.
  `results/moe_stages_ab_*_160027`.
- Autotuned (8-GPU feasibility-pruned shell-walk, `tools/moe_volta_tune_fleet.sh`): per-M JSONs that
  **beat the hand heuristic ~5–10% at concurrency** (e2e adjudicated: `auto` arm > `kbest` arm).
  `results/moe_stages_ab_*_05*` (post-reboot rerun).

**What ships:** see §5. Commits: `bcd1d7f` (root cause) → `f586150` (M-sweep) → `9757a8d` (plugin) →
`6f9282a`/`3706082` (tuner+fleet) → `e6bc100` (e2e adjudication) → `bd749d1` (upstream doc).

---

## 2. FP8 MoE decode — already near-optimal (NO WORK)

The FP16 fix does **not** touch FP8: our FP8 MoE uses custom CUDA kernels (`fp8_dequant.cu`
coalesced/grouped GEMV), never Triton `fused_moe`. That's *why* FP8 was already fast (70 tok/s).

Profiled the FP8 decode path (`tools/moe_fp8_profile_decode.sh`, eager — cudagraph capture conflicts
with the profiler's CUDA-event syncs):
- **M=1:** GEMV/compute = 73% of GPU time; route/scatter glue = 27%. `w2_gemm` = 44% of MoE GPU and
  2.3× slower than `w13` despite half the weight bytes → looked like *the* target.
- **M=8 (8-user) gate:** the w2 "anomaly" **vanishes** — w13≈w2, glue shrinks to 15%. Per-slot w2 is
  5.2× more efficient at M=8 than M=1. So w2's M=1 cost was **fixed under-occupancy overhead**
  (wide-N=2048/short-K=128 starved at 8 slots), not a kernel flaw; it amortizes away at load.

**Verdict:** no kernel work justified. w2 tuning would help only 1–2 user latency (~+11% on an
already-comfortable 70 tok/s); at concurrency the GEMVs are balanced and aggregate is 164–180 tok/s.
Codex's "attack route/scatter" hypothesis measured unsupported (glue is a minority and shrinking).
The *only* scenario to revisit: single-user latency on 122B-A10B TP8 (~45 tok/s) — re-profile M=1
there first. Commits `ce80954`, `8f563c5`. Artifact `results/moe_fp8_profile_20260613_072405`.

---

## 3. Vision encoder (ViT attention) — baseline is the ceiling (NO-GO)

All fleet models share one SigLIP vision tower: hidden=1152, 16 heads, **head_dim=72**, 27 layers.
V100 ViT attention defaults to **Torch SDPA** (cuda.py `get_supported_vit_attn_backends`: sm70 order
short-circuits to TORCH_SDPA before TRITON; FLASH is sm80-gated). Not a missing-JSON cliff.

- **ai-bond FA has the right API** (varlen `(T,H,D)`+cu_seqlens matches ViT exactly) but the kernel
  compiles only `D ∈ {16,32,64,128,256}`; **D=72 hard-errors** (`TORCH_CHECK`). Not graceful fallback.
- **Cheap pad-72→128 bridge** (`tools/vit_fa_v100_d72_microbench.py`): numerically correct (cos=1.0)
  but **0.37–0.47× SDPA** at every ViT shape (S=256–2048, single + batched) — 2–2.7× slower. Flat
  ratio, no crossover. `results/vit_fa_v100_d72_2026061308*`.
- **Deeper custom kernel also dead** (extrapolation): strip the 1.78× pad → native D=72 ~0.71×,
  D=80 ~0.64× — still < SDPA.
- **SDPA is genuinely on its best internal path** (probed, forcing each backend): V100 auto-selects
  **EFFICIENT_ATTENTION** (CUTLASS mem-efficient, non-materializing, tensor-core), **5–7× faster than
  MATH**, handles D=72 and additive packed-image masks without dropping to MATH. The FA microbench's
  SDPA baseline (0.954ms@S2048) == the EFFICIENT path → FA lost to V100's *best* attention, fairly.

**Verdict:** V100 ViT stays on SDPA. Don't build the FA bridge or a custom head-dim kernel — at ViT's
short sequences SDPA's CUTLASS mem-efficient kernel (a hand-tuned Volta software pipeline) is the
ceiling. Real vision levers remain `--skip-mm-profiling` (startup, ~45×) + `--limit-mm-per-prompt`,
and only matter if images are actually served (fleet workload is text). Commits `ea0765c`, `76263fc`.

---

## 4. The unifying principle — V100's memory-feed gap

`cp.async` (sm_80 `LDGSTS`) and its successors (Hopper TMA, Blackwell TMEM) are a four-generation
hardware campaign against one bottleneck: **feeding fast tensor cores from memory.** V100 (the first
tensor-core arch) has the compute but a global→shared copy must go `LDG → register → STS`, consuming
the very registers/occupancy V100 uses to hide latency. So:

- **Software pipelining can't fully imitate cp.async** — deeper Volta double-buffering spends
  registers→occupancy, the resource doing the latency hiding. Our `num_stages` sweep is direct
  evidence: more stages did nothing on V100 (plateau at double-buffer).
- **The baselines already software-pipeline** (CUTLASS mem-efficient SDPA, Triton `num_stages`). So
  "imitate cp.async to win" isn't an untapped lever — it's table stakes the baseline has. Beating it
  means out-engineering CUTLASS's Volta pipeline, not adding a missing optimization.
- **Consequence — V100 is competitive by regime:** memory-bandwidth-bound / resident-weight work
  (decode GEMV) is fine (HBM2 ~816 GB/s measured, little staging); compute-bound tile-staging
  (prefill, big GEMM, ViT) is where the missing async-feed bites.

This is why every win was *moving work into the favorable regime* (FP8-resident decode; the MoE fix is
"stop over-staging shared memory on a chip that can't hide it") and every dead end fought the regime
later silicon was built for (FA-for-ViT, deeper pipelining).

**Strategy for the sm_70 lifeline:** play to resident/bandwidth-bound decode; don't fight the
compute/staging regime. See [[feedback_engine_selection_arch_cadence]] in memory.

---

## 5. What ships (implementation) → see also tools/TOOLS.md and docs/VOLTA_MOE_UPSTREAM.md

- **Plugin patch** `_patch_volta_moe_default_config` in `src/fp8_w8a16_sm70/vllm_serve.py`
  (env `VLLM_V100_MOE_FP16_TUNED`, **default ON**): sm<80 + fp16/bf16 fused-MoE small-M →
  `BLOCK_K=64` heuristic. Universal fallback for any model/TP. `=0` restores stock (for A/B).
- **Bundled tuned configs** `src/fp8_w8a16_sm70/moe_configs/` (auto-loaded via
  `VLLM_TUNED_CONFIG_FOLDER` if unset): exact per-M JSONs for q35b-TP4 / g26b-TP4. JSON lookup runs
  before the heuristic, so matching shapes get the exact tune, others get the heuristic. TP-specific.
- **Layering:** `get_moe_configs` (bundled JSON) → `get_default_config` (heuristic) → stock. Clean.
- FP8 path and our coalesced/grouped GEMV kernels are unchanged (they bypass Triton fused_moe).
- **Upstream artifact:** `docs/VOLTA_MOE_UPSTREAM.md` packages the root cause + heuristic + JSONs +
  e2e evidence for an aphrodite PR (vLLM dropped sm_70 by policy → vLLM PR DOA).
