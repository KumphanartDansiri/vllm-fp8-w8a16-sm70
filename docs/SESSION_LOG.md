# V100 FP8 W8A16 — Session Log

Cold-start summary for picking up in a new session. Read top to bottom.

> **NEXT SESSION → Qwen3.5-122B-A10B-FP8: apply the coalesced kernels (the flagship
> can't-fit-FP16 case — the real benefit of 8×V100-32GB).** Branch `coalesced-fp8-gemv`,
> HEAD `83a6e96`. GLM-4.5-Air is DONE: decode **30.7→45.4 (coalesced attn)→56.6
> (+coalesced MoE w13)** = 1.84×, cudagraph; envelope in
> `docs/GLM45_AIR_V100_CONFIG.md`. The 122B is the next + final large-MoE target;
> together GLM-Air + 122B are the headline of the upcoming big version update.
>
> **KEY DIFFERENCE — the 122B is Qwen BLOCK-FP8 (`quant_method=fp8`), NOT
> compressed-tensors.** So it routes through `Fp8LinearMethod` + `_our_moe_apply_grouped`
> (in `vllm_serve.py`), NOT the CT path where Stage G1 is wired
> (`_v100_ct_mixed_moe_routed` in `compressed_tensors_v100.py`). Two pieces:
> 1. **Attention Linears** — already go through `_v100_fp8_gemm`, where the dense
>    coalesced GEMV is wired (gate: M==1, block_h∈{1,128}, K%128==0). Qwen block-FP8
>    is block_h=128 → should benefit from `VLLM_V100_FP8_COALESCED_GEMV=1` ALREADY.
>    **First step: verify** (A/B the 122B with the coalesced flag on/off).
> 2. **MoE w13** — `_our_moe_apply_grouped` calls `fp8_w8a16_grouped_routed_gemm_a3`
>    directly; it does NOT yet use the new `fp8_w8a16_grouped_gemv_coalesced`.
>    **Wire it** (mirror the Stage G1 decode-w13 dispatch: small-R decode →
>    grouped coalesced GEMV; gate e.g. reuse `VLLM_V100_CT_MOE_W13_COALESCED` or a
>    Qwen-specific one). The kernel handles block_h=128 (numtest-covered), so it
>    transfers directly.
>
> Baseline: 122B-A10B-FP8 TP=8 cudagraph = **34.6 tok/s** (45-47 with MTP). Expect a
> coalesced lift (attn + MoE w13), TP=8-all-reduce-tempered like GLM-Air. Tools:
> `coalesced_gemv_e2e_ab_vllm021.sh` (MODEL=122B TP=8) + the grouped numtest.
> AFTER both large MoE models land → the BIG VERSION UPDATE: README rewrite
> (breakthrough reframes "dense loses"→"FP8-resident ≈FP16 for can't-fit-FP16
> Volta"; fleet table Gemma-31B 4.3× / Qwen-27B 3.2× / Gemma-26B-MoE 1.9× /
> GLM-Air 1.84× / 122B) + version bump (0.5.0 taken → 0.6.0) + push/publish
> (resumes [[project_github_publish_paused]], needs ssh passphrase). See the
> [[project_gemma4_fp8_resident]] RESUME header for the full coalesced playbook.

---

## Session note (2026-06-19) — Dense FP8 dequant breakthrough on V100

Supersedes the earlier "dense FP8 dequant terminus" conclusion. The NCU diagnosis
was right that dense large-K `coal_m` was dequant-bound, but the cost was not
intrinsic enough to stop: the slow branchy e4m3 converter was the bottleneck.

Validated fast path: `fp8_e4m3_to_fp16_bits_fast` uses the branchless
shift-and-rebias trick (`mag << 7`, multiply by `2^8`) plus NaN fixup. It is
value-identical to the old converter in numtests across attn/gdn/mlp/channel and
block-FP8 shapes, M=1/2/8: cos(ref)=1.00000000, cos(old)=1.00000000, max_abs only
fp16-LSB scale.

NCU on the real `fp8_w8a16_gemv_coalesced_m_kernel<4,8>` path (5120x5120, M=8):

| metric | old slow dequant | branchless dequant | delta |
|---|---:|---:|---:|
| time | 0.260 ms | 0.194 ms | -25% |
| total instructions | 70.6M | 53.2M | -25% |
| ALU | 28.6M | 23.1M | -19% |
| FMA | 18.5M | 16.1M | -13% |
| LSU | 9.0M | 4.1M | -55% |
| DRAM peak | 11% | 15% | rising |

Dense FP8 `coal_m` now beats FP16 cuBLAS at realistic low decode M:
M=1 `0.063 vs 0.078 ms`, M=2 `0.076 vs 0.098 ms`, ties M=4
`0.107 vs 0.109 ms`, and improves M=8 from `2.29x` to `1.64x` behind.

Interpretation: the useful conclusion is no longer "dense FP8 is memory-only on
V100." It is: **slow software e4m3 decode was the limiter; branchless dequant makes
dense FP8 a speed path for M<=4 while preserving the residency win.** Remaining
large-M work is now narrower: promote the fast converter across all FP8 kernels,
then test fast-dequant + half2/vectorized dequant for M=8.

E2E confirmation on Qwen3.6-27B, TP4, same-model FP8-fast vs FP16 serving
(`results/q27b_fp8fast_e2e`, `results/q27b_fp16_e2e`, 512 generated tokens):

| concurrency | FP8-fast | FP16 | verdict |
|---|---:|---:|---|
| C1 per-user | 52.5 tok/s | 39.1 tok/s | FP8 1.34x faster |
| C2 per-user | ~37 tok/s | 31.1 tok/s | FP8 ~1.2x faster |
| C4 per-user | 29.9 tok/s | 30.2 tok/s | parity |
| C8 per-user | 19.7 tok/s | 29.3 tok/s | FP16 faster |
| C8 aggregate | 158 tok/s | 234 tok/s | FP16 faster |

This flips the dense 27B result from the old slow-FP8 triad (`36.4` vs `39.1`
tok/s at C1) to FP8-fast beating FP16 (`52.5` vs `39.1`). The microbench-to-e2e
translation held: FP8 wins C1/C2, ties C4, and still loses C8 because the M=8
kernel remains behind FP16 cuBLAS. The dense-FP8 headline is now: **FP8-resident is
faster and lighter than FP16 for low-user dense decode on V100; FP16 remains the
high-concurrency C8 choice where it fits.**

Promotion status: the canonical `fp8_e4m3_to_fp16_bits()` body now uses the
branchless converter, so all old call sites inherit the fast path; the explicit
`fp8_e4m3_to_fp16_bits_fast()` remains as a compatibility alias. Validation after
the global promotion:
- `ct_fp8_resident_numtest_vllm021.sh`: PASS, covering A.1/A.2/A.3 and WMMA
  resident Linear paths, including partial-N.
- `qwen27b_fp8_gemm_microbench.sh` smoke: PASS numerically, coalesced/A-path errors
  only fp16-LSB scale; fast dense decode still beats cuBLAS at M=1/2 and ties M=4.
- `ct_fp8_grouped_moe_numtest_vllm021.sh`: PASS, grouped channel MoE including
  partial-N.
- `ct_fp8_grouped_coalesced_numtest_vllm021.sh`: PASS, grouped coalesced routed
  GEMV vs A.3/FP32, channel and block scales.
- `ct_fp8_fused_tiled_numtest_vllm021.sh`: PASS, fused grouped-tiled path.
- `ct_fp8_tiled_prefill_numtest_vllm021.sh`: PASS, tiled prefill vs grouped/ref.
- `ct_fp8_channel_wmma_numtest_vllm021.sh`: PASS, channel WMMA plus block_h=128
  no-regression.
- `ct_fp8_mixed_moe_numtest_vllm021.sh`: PASS, composed route/w13/act/w2/scatter.

Standalone reconstruct microbench remains useful context. Full-matrix FP8 dequant
vs GPTQ-int4 dequant, repeated 1000 iterations on V100, showed GPTQ only about
10-18% faster for reconstruct (`GPTQ/FP8 ~= 0.82-0.89`) because both formats write
the same FP16 output matrix. GPTQ's serving speed, when correct, is therefore not
explained by standalone dequant alone; its useful lesson is fused unpack/dequant
near the dot product.

### Dead ends (2026-06-20) — M=8 dense-decode probes that do NOT help

After the converter promotion the only open dense-decode gap is M=8 (C8): the
scalar `coal_m` GEMV scales ~linearly in M while cuBLAS (tensor cores) is nearly
M-flat, so FP8 loses C8. We built and microbenched candidate M=8 levers. Result
(5120x5120 attn, UNROLL=4, ms/call; `qwen27b_fp8_gemm_microbench.py`):

| M | cuBLAS FP16 | coal_m (prod) | coal_h2 (half2) | h2_err | sk8 (split-K) |
|---|---:|---:|---:|---:|---:|
| 1 | 0.078 | 0.057 | 0.055 | 0.001 | 0.077 |
| 2 | 0.098 | 0.072 | 0.071 | 0.002 | 0.097 |
| 4 | 0.109 | 0.101 | 0.099 | 0.002 | 0.137 |
| 8 | 0.095 | 0.165 | 0.162 | 0.002 | 0.220 |

- **half2 (`fp8_w8a16_gemv_coalesced_m_half2_kernel`): NO-GO.** ~2% at M=8 and a
  real precision regression (`h2_err` 0.001-0.002): it does the multiply in
  half2 (`__hmul2`) but then converts products back to FP32 to accumulate, and
  the per-byte dequant stays scalar. It attacks the FMA pipe; the NCU bottleneck
  is ALU/dequant (~40%). Wrong pipe, small downside.
- **split-K (`fp8_w8a16_gemv_coalesced_m_splitk_kernel`): NO-GO** for these
  shapes. Slower at every M (M=8: 0.220 vs 0.165) — N=5120 already exposes enough
  blocks, so the extra FP32 reduction pass is pure cost. (May still help tiny-N.)
- **grouped-mtile / wmma-m16 / gptq dequant-tax kernels:** investigation probes,
  microbench-only, left OFF.

Conclusion: **dense C8 needs dequant *instruction* reduction (vectorized 4xFP8
decode via `prmt`/byte-perm), not multiply packing or split-K.** That remains a
HYPOTHESIS to NCU-arbitrate (Volta byte-perm can be eaten by register pressure /
packing / half-conversion throughput), not an assumed win. The microbench->e2e
map held exactly: coal_m wins C1/C2, ties C4, loses C8, matching the e2e triad.

All of the above probe kernels are committed microbench-only (pybind entry points
exercised by `qwen27b_fp8_gemm_microbench.py` / `wmma_m16_shape_microbench.py` /
`dequant_tax_microbench.py`); none are wired into the serving dispatch.

---

## Discussion note (2026-06-09) — FlashAttention V100 integration shape

This is an architecture preference/handoff note, not a validated implementation.
The useful mental model is three layers:

1. Python/API layer: the caller-facing contract, like a public `.h` file
   (`flash_attn_func`, `flash_attn_varlen_func`, `flash_attn_with_kvcache`,
   argument order, return shape).
2. C++/pybind extension layer: exported module/symbol names, e.g.
   `flash_attn_2_cuda` / `flash_attn_v100_cuda` with `fwd`, `bwd`,
   `varlen_fwd`, `fwd_kvcache`, etc.
3. CUDA implementation layer: the actual Volta-specific launchers/kernels. This
   may differ completely from upstream as long as it honors the contract.

For a "lower the sm70 gate and work" experience, layers 1 and 2 must look like
standard FlashAttention/vLLM FlashAttention. Layer 3 can be custom V100 code.
If layer 1 or 2 diverges, the vLLM caller must be patched.

Local reference read:
- `/home/kumphanartd/flash-attention-v100` (ai-bond) keeps a more natural
  FlashAttention-like outer shape: standard-ish Python functions and
  `flash_attn_2_cuda` compatibility/symlink behavior.
- `/home/kumphanartd/1catai-vllm/flash-attention-v100` exposes more
  vLLM-serving-specific paths such as `flash_attn_decode_paged` and
  `flash_attn_prefill_paged`, plus paged KV/decode workspace/partition helpers.
  This may reflect intentional deterministic caller dispatch, an incremental
  development checkpoint style, or both.
- vLLM 0.21 stock `FLASH_ATTN` is sm80+ (`supports_compute_capability >= 8.0`).
  It already uses a serving-shaped FlashAttention path, especially
  `flash_attn_varlen_func(...)` with `block_table`, `seq_lens`, `cu_seqlens_q`,
  max sequence lengths, scheduler metadata, and paged KV cache tensors.

Preferred direction if we pursue FlashAttention V100: keep the public wrapper as
standard as possible, then hide V100-specific dispatch under that wrapper. Use
ai-bond as the API/package-compatibility reference and 1catai as the reference
for serving cases (paged prefill, single-token decode, long-context partitioning,
FP8 KV utilities). With Claude+Codex assisting, hidden dispatch is acceptable if
it is observable.

Debug/observability split:
- Development-only inspection: tensor dumps, step traces, reference comparisons,
  L2/cos checks, detailed intermediate timings. Useful while proving correctness;
  remove or leave behind heavy debug flags once settled.
- Runtime decision controls: path summaries, force/disable knobs, partition-size
  selection, fallback gates, coarse counters/profiling. Keep these longer because
  they affect routing, A/B testing, and emergency rollback.

Suggested shape:
- `FLASH_V100_DEBUG=0`: quiet
- `=1`: path decisions (`paged_decode`, `paged_prefill`, fallback)
- `=2`: shapes/layouts/dtypes
- `=3`: kernel timings/counters
- `=4`: correctness self-checks/reference comparisons
- `=5`: tensor dumps/deep trace

Keep behavior knobs separate from debug level, e.g. `FLASH_V100_FORCE_PATH`,
`FLASH_V100_DISABLE_PAGED`, `FLASH_V100_DECODE_PARTITION_SIZE`.

---

## Session handoff (2026-06-09) — Phase 3: CUDAGRAPH WORKS, GLM-Air decode 30.7 tok/s

The big unlock, immediately after Phase 2d. GLM-4.5-Air-FP8 mixed (w13 FP8-resident
+ freed, grouped w2) now decodes at **~31 tok/s single-stream with cudagraph**
(eager was 2.68; the old w2 loop was 0.37). Coherent, KV/concurrency win intact.

**Run it:** `MODE=cudagraph ./tools/glm45_air_fp8_load_vllm021.sh` (with the usual
`VLLM_V100_CT_FP8_RESIDENT=1 VLLM_V100_CT_MOE_W13_RESIDENT=1
VLLM_V100_CT_MOE_W13_FREE_FP16=1 VLLM_V100_CT_MOE_W2_GROUPED=1`). The `MODE`
knob sets `--compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}` +
pins `VLLM_ATTENTION_BACKEND=TRITON_ATTN`.

**Why mode=0 is mandatory (the whole story):** vLLM 0.21's default `-O3
VLLM_COMPILE + FULL_AND_PIECEWISE` runs a TorchDynamo **fullgraph** trace
(`vllm/compilation/wrapper.py:150 fullgraph=True`) BEFORE cudagraph. Our pybind
kernels are wrapped in `torch._dynamo.disable` (vllm_serve.py:64-75 — needed
because Dynamo can't trace pybind C funcs). Under fullgraph, a graph break is a
FATAL error: `torch._dynamo.exc.Unsupported: Skip calling
torch.compiler.disable()'d function`. `mode=0` (CompilationMode.NONE) skips Dynamo
entirely, so the disable'd pybind ops just run eager INSIDE the captured cudagraph.
This is the same `mode=0 + FULL_DECODE_ONLY` envelope the Qwen 0.18 path used.

**Why FULL_DECODE_ONLY survives mode=0:** `resolve_cudagraph_mode_and_sizes`
checks the attention backend's `AttentionCGSupport`. **TRITON_ATTN = ALWAYS (=3)**
(`vllm/v1/attention/backends/triton_attn.py:126`), so it is NOT downgraded. vLLM
auto-selects TRITON_ATTN on V100 (FA2 needs cc>=8) but the script PINS it for
determinism.

**Static audit (Claude+Codex) predicted capturability and was right:**
- `V100_FP8_STREAM = at::cuda::getCurrentCUDAStream()` → kernels launch on the
  capture stream (correct).
- No host syncs on the grouped decode path. The OLD w2 per-expert loop's
  `unique().tolist()` + data-dependent launch count was a HARD cudagraph blocker —
  so **Phase 2d (grouped w2) was a PREREQUISITE for Phase 3**, not just a perf fix.
- Static shapes per captured size; wrapper `torch::zeros`/`.to(fp16)` allocate but
  are capturable via the graph mempool.
- Caveat: our grouped kernels use `atomicAdd` (split-K) → non-deterministic at
  temp=0 EAGER-vs-eager too, so token-exactness is unattainable regardless of
  cudagraph. Bar = late divergence + coherent (cudagraph-vs-eager: char 298,
  single-word paraphrase — PASS).

**Measured (file-verified, MODE=cudagraph, warmed, MAXTOK=256, both temp0):**

| MAXLEN | decode tok/s | KV cache | concurrency |
|---|---:|---:|---:|
| 2048 | 30.69 | 563k | 275× |
| 8192 | 31.14 | 482k | 58.9× |
| 32768 | 31.07 | 159k | 4.85× |

All capture cleanly (FULL decode, 0.07 GiB pool, sizes [1,2,4,8,16] cover NS=8),
0 errors/OOM. NOTE: that table is SHORT-CONTEXT decode (256-tok gen from a short
prompt) in servers CONFIGURED for 2k/8k/32k — flat ~31 because the decoded sequence
is shallow regardless of MAXLEN.

**DECODE-AT-DEPTH (the real long-context number; `DEPTH=` knob prefills a long
prompt; file-verified `/tmp/v100_cg_depth/`, labeled by actual prompt_tokens):**

| actual KV depth | decode tok/s | prefill TTFT |
|---|---:|---:|
| shallow (~250) | ~31 | 0.3s |
| 6,070 | 27.05 | 32s |
| 26,198 | 18.50 | 169s |

The decode falloff (31→27→18.5) is the ATTENTION cost over a deeper KV (TRITON_ATTN
decode kernel scans more KV blocks/token) — inherent to ANY model at depth, NOT our
FP8 kernels (O(1)/token); FP16-fused would fall off identically. Usable throughout
(≤8k = comfortable ~27; ~26k = 18.5, heavy but works). Anchor 30.7 vs the FP16-fused
EAGER baseline (4.65); the fair single-stream comparator (FP16-fused cudagraph) is a
separate run not yet done.

**SEPARATE FINDING — prefill TTFT is heavy: 169s for a 26k-token prompt.** That's
our EAGER CUDA-core w13/w2 grouped GEMMs at large M (prefill is NOT cudagraphed
under FULL_DECODE_ONLY, and our kernels aren't tensor-core). So **WMMA-w13 is NOT
moot — it's the PREFILL lever** (cudagraph already solved decode). Revisit WMMA for
long-prompt TTFT, gated on a microbench.

**FP16-FUSED-CUDAGRAPH COMPARATOR DONE — mixed-FP8 DOMINATES (file-verified
`/tmp/v100_fp16fused_cg/`).** Same stack (FP8 CT Linears + cudagraph mode=0), only
MoE differs, 2k short-context:

| MoE config (both cudagraph) | decode tok/s | concurrency |
|---|---:|---:|
| mixed-FP8 (W13 resident + grouped w2) | 30.7 | 275× |
| FP16-fused (W13_RESIDENT=0) | 4.81 | 78× |

**Mixed-FP8 wins BOTH axes: 6.4× faster AND 3.5× more concurrent** — the OPPOSITE of
eager (where FP16-fused 4.65 > mixed 2.68). Why: at decode M=1 our mixed path was
LAUNCH-bound (many small pybind/grouped kernels) → cudagraph removed launch overhead
→ 2.68→30.7 (11.4×); the FP16-fused MoE is KERNEL-bound (vLLM Triton fused-MoE on
sm_70 reads full-FP16 expert weights, poorly optimized for Volta) → cudagraph barely
helps (4.65→4.81). So the "memory-not-speed" framing is RETIRED: **mixed-FP8 +
cudagraph is simply the best GLM-Air config on V100, faster AND higher-concurrency
than FP16-fused.** Caveat: FP16-fused here runs mode=0 (forced by our pybind CT
Linears, no inductor); a pure-FP16 mode=3 path wasn't measured but abandons the FP8
memory win and barely fits FP16 — not the operative use case.

**Script hardening landed (Codex review):** `glm45_air_fp8_load_vllm021.sh` now
uses array-based `EXEC_OPTS=(...)` (was a fragile JSON-no-spaces string), always
pins `VLLM_ATTENTION_BACKEND`, banner shows `mode=$MODE`, and the streaming measure
records the gold-standard API `completion_tokens` (`stream_options.include_usage`)
alongside the SSE chunk count, plus a warmup request to dodge Triton-JIT spikes.

**NEXT:** ~~decode-at-depth~~ DONE. ~~FP16-fused-cudagraph comparator~~ DONE.
~~promote MODE=cudagraph default~~ DONE (bare invocation → mode=cudagraph → 30.75
tok/s; MODE=eager escape hatch kept for profiling). Remaining: earn headline
numbers; PREFILL kernel work (below); then Gemma-4 FP8.

**PREFILL SCOPING DONE 2026-06-09 (microbench `tools/prefill_wmma_microbench_vllm021.sh`,
file-verified) — the 169s TTFT@26k is ~94% the grouped MoE kernels, and the root
cause is bigger than "no WMMA":**
- (A) Dense WMMA(tensor-core) vs A.2(CUDA-core) on N=K=4096: WMMA = **8–9.8×**
  faster (17 vs 1.75 TFLOP/s at M=2048). Tensor cores are a big prefill win.
- (B) Grouped MoE kernels (the prefill path) run at **0.24 (w13) / 0.29 (w2)
  TFLOP/s** — ~7× slower than even dense A.2, ~70× slower than WMMA. At R=32768:
  w13 389ms + w2 164ms = 553ms/layer.
- Extrapolation: GLM 26k prefill R=209,584 rows/layer ×45 layers ×553ms×(R/32768)
  = **~159s ≈ the measured 169s TTFT. So grouped MoE = ~94% of prefill.**
- **ROOT CAUSE:** the grouped kernel launches ONE CTA PER ROUTED ROW
  (`gridDim.y=R`, M=1/CTA) = a per-row GEMV with ZERO cross-row weight reuse
  (re-reads each expert weight per row → bandwidth-bound → 0.24 TFLOP/s). Fine for
  DECODE (R=topk tiny, cudagraphed — why decode is great), terrible for PREFILL
  (R=200k).
- **PLAN (the prefill project):** a TILED grouped GEMM for the prefill path —
  sort/group routed rows by expert, then a real per-expert [M_e,K]×[N,K] GEMM that
  REUSES the weight across rows. Tiled CUDA-core (A.2-class) alone → ~7× → 169s→~24s;
  + WMMA (dequant FP8→FP16 in regs, HMMA.884, the existing PoC's approach but for
  CHANNEL scale block_h=1, which the PoC doesn't yet support) → ~10–15s. Dispatch:
  large R (prefill) → tiled kernel; small R (decode) → keep the per-row kernel
  (cudagraphed). This is its own kernel project; decode is unaffected.

**STAGE 1 DONE + VALIDATED 2026-06-09 — and it CORRECTS the "94% MoE" claim.**
Implemented `_v100_ct_tiled_prefill_moe` (group rows by expert → per-expert a2(w13
FP8) + cuBLAS(w2 FP16)), dispatched at R≥256 (`VLLM_V100_CT_MOE_PREFILL_TILED=1`,
`_MIN_R=256`); decode (R<256, cudagraph) untouched. Numtest
`ct_fp8_tiled_prefill_numtest_vllm021.sh`: tiled vs per-row grouped **cos=1.00000**,
both match FP32 ref. E2E (MODE=cudagraph): **26k TTFT 169s→74s (2.3×)**, 6k 32s→14s,
decode UNCHANGED (30.86/27.03/18.62), coherent, ENGAGED.
- Overhead microbench `tiled_prefill_overhead_microbench_vllm021.sh` (R=209584):
  tiled MoE = **9.7× on the kernels** (364 vs 3538 ms/layer → 16.4s vs 159s/45L),
  grouping+scatter only **3%** (Codex's caution — clean, not the bottleneck). The
  per-expert a2(w13) loop is 341ms (94% of tiled MoE); w2 cuBLAS 11ms.
- **KEY CORRECTION:** MoE was NOT 94% of *in-model* prefill — that extrapolation
  measured only the MoE kernel in isolation. After tiling MoE to ~16s, e2e is still
  74s, so the remaining **~58s is the DENSE CT Linears (qkv/o_proj)** which ALSO run
  A.2 CUDA-core at prefill: they're CHANNEL scale (block_h=1), and `wmma_layer_ok`
  needs block_h=128, so they fall to A.2 (1.75 TFLOP/s) — plus attention prefill at
  26k. (The 46 *fallback* Linears use dequant-FP16+cuBLAS = fast; the 138 *resident*
  channel Linears are the slow ones — a prefill-vs-memory tradeoff the WMMA kernel
  resolves.)
- **STAGE 2 = a CHANNEL-SCALE WMMA kernel** (extend the PoC from block_h=128 to
  block_h=1). It speeds BOTH the dense Linears AND the tiled-MoE-w13 loop (both A.2
  today). Bench: WMMA 9.8× over A.2 → projected e2e **74s → ~20s**. Decode
  (FP8-resident + cudagraph, 30.7) is unaffected.

**STAGE 2 DONE + the projection was WRONG (honest result) 2026-06-09.** Extended the
WMMA kernel to block_h=1 (per-output-row scale; `sr=(block_h==1)?n_start+row:scale_row`
at the 2 dequant sites + binding relax). Standalone numtest
`ct_fp8_channel_wmma_numtest_vllm021.sh`: channel-WMMA cos=1.00000 L2rel=3e-4,
block_h=128 no-regression. Wired into dense `_v100_fp8_gemm` (channel N%64==0 →
WMMA at prefill M; `VLLM_V100_CT_CHANNEL_WMMA=1` kill switch; decode M≤8→A.3
untouched). A/B 26k (both tiled-prefill on): WMMA off=74.1s, on=**63.9s** — only
**−10s (1.16×)**, decode unchanged (18.6), coherent. **LESSON: the "dense Linears
≈40s" estimate was WRONG — they were ~11s.** So the isolated microbenches (tiled
MoE=16s, dense=11s) do NOT explain the in-model 64s. **PHASE 4 TOTAL: 169s→64s
(2.6×).** NEXT: **STOP ESTIMATING — instrument the real prefill** (per-section CUDA
timers in the forward). Prime suspect = the per-expert PYTHON LOOP launch overhead
in the live engine (128 experts × 45 layers ≈ 5760 sequential tiny a2/cuBLAS
launches — a tight microbench loop hides it, the live engine pays it) + attention
at 26k. If loop-bound, the fix is **Stage 1.5 = ONE fused grouped-tiled CUDA kernel**
(on-device routing, one launch/layer), NOT more WMMA. MoE-w13 partial-N WMMA
(N=352, Stage 2b) deferred.

**STAGE 1.5a DONE 2026-06-09 — fused w13 works; it REVEALED the prefill floor is
ATTENTION, not our code.** Built `fp8_w8a16_grouped_tiled_gemm` (ONE launch/layer,
per-tile expert via binary search over GPU-side offsets; sync-free route-prep =
argsort+scatter_add+cumsum, NO `.tolist()`). Numtest
`ct_fp8_fused_tiled_numtest_vllm021.sh` bit-identical to per-expert a2 (cos=1.0;
partial-N=352, 0-tile experts, BM-tails). **w2 A/B (Codex's catch): per-row grouped
kernel is a DECODE kernel — 43s GPU at prefill R=209k (no cross-row reuse). Option A
(fused w13 + grouped w2) = 94s REGRESSED; option B (fused w13 + per-expert cuBLAS
w2) = 60.2s SHIPPED.** `VLLM_V100_CT_MOE_PREFILL_FUSED=1`. PROFILE(B): fused_prep
sync-free 111ms (17.5s grouping stall GONE); our-code wall/GPU = 1.09 = **GPU-bound
(host-loop overhead RETIRED)**, but our code is only ~17s of the 60s TTFT → **~42s
(70%) is OUTSIDE our kernels = self-attention@26k + TP all-reduce.** PHASE 4 TOTAL
169s→60s (2.8×), decode 30.7 untouched. **MoE/Linear prefill has hit DIMINISHING
RETURNS — floor is now attention+comm.** Contained lever left = Stage 1.5b: WMMA on
the 12s w13 GPU (channel-WMMA exists, needs partial-N N=352 to wire into the fused
kernel; ~10s → ~50s). Big ~42s residual = separate V100-Flash-attn project. C++
wrapper hardening added (route-layout numel==E, weight [E,N,K]).

**Optimization stance / backlog:** GLM-Air is now in a working, serving-viable
FP8-resident envelope, but this does **not** mean every kernel/path is fully
optimized. We made the path work, removed the dominant w2 Python-loop bottleneck,
and unlocked cudagraph. Further tuning is intentionally demand-driven: if a real
workload shows a problem, investigate that bucket and update this log with the new
evidence. Known remaining optimization categories:
- **Decode-at-depth / attention-KV behavior:** short-context decode is ~31 tok/s
  even under MAXLEN=32k, but true 8k/32k-prompt decode is still the measurement
  that decides whether attention/KV scheduling needs work.
- **Dense FP8 GEMM:** likely suboptimal. This is not urgent for the large-MoE
  target, but becomes important if the goal is dense 27B-class FP8 on one 32GB
  V100.
- **MoE w13 GEMM:** still CUDA-core grouped FP8. Cudagraph makes decode
  comfortable, so WMMA/tensor-core w13 is no longer immediate, but it remains a
  possible prefill or throughput lever.
- **Wrapper allocation / epilogue cleanup:** C++ wrappers allocate FP32 outputs
  (`torch::zeros`) and cast to FP16; capturable but not ideal. Preallocated
  buffers or fused epilogues could reduce overhead.
- **Route/scatter fusion:** `index_select`, activation, route weight, and
  `index_add_` remain separate ops. A fused MoE path could reduce launch count and
  memory traffic if profiling shows it matters.
- **Comparators / promotion:** FP16-fused cudagraph is the fair short-context
  single-stream comparator; run it before broad headline claims. Promote
  `MODE=cudagraph` for GLM-Air only after the long-prompt and comparator story is
  clear.

---

## Session handoff (2026-06-09) — Phase 2d: GLM-Air mixed-MoE w2 decode FIXED + VALIDATED

Co-developed with **Codex** (OpenAI o-series CLI peer reviewer — the user relays
analyses between Claude and Codex; this is the official subscription name, not
"GPT"). Codex implemented the kernel in the working tree in parallel; Claude
reviewed, fixed a real gap, and ran all the hardware gates.

**What shipped:** the GLM-4.5-Air-FP8 mixed-MoE decode bottleneck is gone. The old
`_v100_ct_mixed_moe_routed` ran w2 (down) as a per-expert Python loop
(`for e in torch.unique(expert_ids).tolist(): hidden[m] @ w2[e].T`) — a GPU→CPU
sync + Python loop every layer every token, pinning decode at ~0.37 tok/s. It's
replaced by a single grouped FP16 routed GEMM launch.

**Design (Claude + Codex consensus): keep w2 FP16, route through a NEW grouped
kernel** `fp16_grouped_routed_gemm` (in `fp8_dequant.cu`). Chosen OVER making w2
FP8-resident (pad K 176→256) because the FP16 kernel uses scalar `__half` loads →
handles GLM's K=I/TP=176 (not 128-aligned) natively with no padding, and adds zero
FP8 rounding to w2. The big memory win was already w13-free; w2's 176 K-tail is
small, so keeping it FP16 is a good trade. Flag `VLLM_V100_CT_MOE_W2_GROUPED=1`
(default on; `=0` kill switch → the old loop). `_K_SPLIT` tunes K-occupancy,
`_CHUNK` bounds routed rows per launch.

**Claude's fixes on top of Codex's kernel:**
- The new kernel launches `gridDim.y = R` like the w13 kernel, so long-context
  prefill (R = M·topk > 65535) would crash *"invalid configuration argument"*.
  Codex had defined `_CT_MOE_W2_CHUNK` but left it UNWIRED — Claude added the
  R-chunk loop mirroring w13.
- Forwarded the 3 W2 env vars into the container in
  `glm45_air_fp8_load_vllm021.sh` (the kill switch was uncontrollable before).
- Made the load script STREAM the generation so it reports decode tok/s (TTFT vs
  steady-state), and surface the w2 self-check tally.
- Added a load-time real-weight w2 self-check (`_ct_moe_selfcheck_w2`, mirrors
  the w13 one), gated by `VLLM_V100_CT_FP8_MOE_SELFCHECK=1`.

**Validation (all on 8×V100 TP8, files in `/tmp/v100_w2_ab/`, `/tmp/v100_gateB/`):**
- Kernel unit test `tools/ct_fp16_w2_grouped_numtest_vllm021.sh` — 10/10 PASS
  (decode/prefill R, partial-N, K=176 tail, k_split 1/2, invalid −1 routes,
  R=0), L2rel≈2e-4, cos=1.000000 vs an FP32 reference.
- Real-weight w2 self-check — **45/45 layers OK, bad=0, L2rel=0.0002** on the
  actual GLM w2 weights (w13 also 45/45, no regression).
- Decode A/B (both temp=0): **grouped 2.71 tok/s vs loop 0.37 → ~7.3×**, TTFT
  0.53s vs 2.80s. Both coherent.
- **Gate B** `tools/ct_mixed_moe_e2e_diff_vllm021.sh` (pure-FP16 baseline vs
  grouped-w2-mixed, all 45 layers) — diverges at char 127/1000, LATE + both
  outputs fully coherent (clean France-essay planning), mixed ERR=0, self-check
  BAD=0, ENGAGED + FREED. PASS per Codex's bar (no early <char-40 divergence =
  no missing-shared bug; tiny FP8-rounding numeric diff acceptable).

**THE HONEST PERF FRAMING (important — don't overclaim):** the mixed path is a
**memory-and-concurrency win, NOT a single-stream speed win.** Gate B measured the
pure-FP16 fused-MoE baseline at **4.65 tok/s** vs mixed **2.68 tok/s** — mixed is
~1.7× SLOWER per stream because our w13/w2 grouped kernels are CUDA-core while the
FP16 fused path uses tensor cores. What mixed buys: it FREES ~8.3 GB/GPU FP16 w13
→ KV cache 168k→572k tokens, concurrency ~82×→279×, and 32k context the FP16-fused
path can't fit. Grouped w2's actual win = it moved mixed from 0.37 (7× slower than
baseline → unusable) to 2.68 (1.7× slower → serving-viable) while keeping that
memory advantage. For multi-user serving the aggregate throughput likely flips in
mixed's favor (far more concurrent streams). 2.68 eager is still below the
comfortable single-stream UX band, so **headline tok/s claims are NOT yet earned.**

**NEXT (agreed order, Claude + Codex):**
1. ~~Real-weight w2 self-check~~ DONE (45/45).
2. ~~Update SESSION_LOG / README~~ (this entry; README pending).
3. **Scope FP8 cudagraph** — the big lever. Prior evidence: cudagraph gave the
   GPTQ-Int4 122B path ~10× over eager. Our FP8 grouped kernels launch on a
   custom CUDA stream (`V100_FP8_STREAM`) — the key question is whether they're
   cudagraph-capturable on Volta, or need rework. This is the path to lifting
   2.68 toward/past the FP16-fused 4.65.
4. **WMMA w13 GEMM** after cudagraph constraints are understood (tensor-core
   w13 would close the per-kernel gap; gate on a microbench).

---

## Session handoff (2026-06-08) — vLLM 0.21 stock sm_70 base validated

The vLLM 0.21.0 source-build lane is no longer hypothetical. The current
experimental image is `vllm-v100:vllm021-cu126`, built from
`/home/kumphanartd/vllm-0.21` using `docker/Dockerfile.vllm021_cu126`.

Key stack decision:
- vLLM 0.21 pins torch 2.11.0. Use the **cu126** PyTorch wheel because
  torch 2.11+cu128 no longer includes Volta in `torch.cuda.get_arch_list()`.
- CUDA 12.6 source build still keeps the `sm_70` path viable. CUDA 13 remains
  out of bounds.

Validation artifacts:
- `/tmp/v100_smoke021/SUMMARY.txt`
- `/tmp/v100_smoke021/SUMMARY_modes.txt`
- `/tmp/v100_smoke021/*_sample.txt`
- `/tmp/v100_smoke021/*_serve.log`

Stock vLLM 0.21.0, no FP8 patches, on Tesla V100:

| Model | eager tok/s | cudagraph tok/s | cudagraph + MTP |
|---|---:|---:|---|
| Qwen3.6-27B FP16 dense GDN | 6.81 | 36.28 | 42.32, exact-match/lossless, 82.6% accept |
| Qwen3.6-35B-A3B FP16 MoE | 6.61 | 15.34 | coherent but exact-diff, 86.0% accept |
| gemma-4-31B-it FP16 dense | 8.11 | 28.50 | no MTP head |
| Qwen3.5-122B-A10B GPTQ-Int4 TP8 | 5.31 | 52.01 | worker-init crash |

Interpretation:
- Gate 4 passed: stock vLLM 0.21.0 runs on V100 in eager and cudagraph across
  dense, MoE, GPTQ-Int4, and Gemma 4 workloads.
- Cudagraph on Volta is real and high leverage. The 122B-Int4 baseline reaches
  52.01 tok/s, roughly 9.8x its eager speed.
- MTP is validated on the dense 27B canary with token-for-token exact output
  vs cudagraph.
- 35B-A3B MTP exactness failure is likely benign MoE+FP16 nondeterminism: the
  output is coherent and forks only after a long identical prefix. Acceptance
  alone is not a correctness proof; keep exact/diff or sample-text checks.
- 122B-Int4 MTP crash is separate from FP8. The MTP head is recognized, but the
  Volta GPTQ path is constrained to `gptq_gemm`; Marlin is not available on
  `sm_70`.

Next work:
- Start the FP8 W8A16 port to vLLM 0.21. The source inspection found the main
  insertion points still intact: `Fp8Config.get_min_capability`,
  `Fp8LinearMethod` init/PWAL/apply, and `Fp8MoEMethod` /
  `Fp8OnlineMoEMethod` init/PWAL/apply.
- Use the existing 0.21 smoke scripts as the stock-base harness, then add an
  FP8 path once the mounted `fp8_w8a16_sm70.vllm_serve` wrapper engages under
  the 0.21 image.
- Defer the 35B-MTP divergence investigation unless it becomes relevant to an
  FP8 MTP claim.

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

## Stage 2D Step 1+2A+2B.1 (2026-05-25): measurement-only attribution of the MoE wrapper

Tagged at `v0.3.1`. Pure instrumentation; no behavior change vs `v0.3.0`. The
goal was to attribute the ~60 ms/token wrapper gap between
`Qwen3NextSparseMoeBlock` (~145 ms in profile-on) and our `_our_moe_apply`
(~84 ms), which was visible at the close of Stage 2C but not decomposed.

Three layers of sub-attribution added:

**Step 1: sub-MoE module hooks.** `_BREAKDOWN_SPARSE_MOE_CHILDREN` attaches
instance-tagged `_v100_breakdown_section` hooks to `self.gate` (`moe_router`),
`self.experts` (`moe_experts`), and `self.shared_expert`/`self.shared_experts`
(`moe_shared`) on every `Qwen3NextSparseMoeBlock`. Refactored
`_breakdown_pre_hook` / `_breakdown_post_hook` to read the instance attribute
first (class-map fallback preserves backward-compat). Gated by
`VLLM_V100_FP8_DECODE_BREAKDOWN_MOE_SUBS=1` (default on when
`VLLM_V100_FP8_DECODE_BREAKDOWN=1` is set).

**Step 2A: residual math fix + one-shot shared-expert structure dump.**
Source read of `vllm/model_executor/models/qwen3_next.py` and
`shared_fused_moe.py` revealed that `self.shared_expert` is invoked **from
inside** `SharedFusedMoE.forward()` (line 28: `shared_out = self._shared_experts(hidden_states)`).
So the `moe_shared` hook fires *inside* the `moe_experts` measurement window;
they are NOT siblings. Without the fix, the parent residual
`moe_other = parent - router - experts - shared` double-subtracted
`moe_shared`, inflating the apparent wrapper gap by exactly `moe_shared`.

Fix: `_BREAKDOWN_RESIDUAL_OF["Qwen3NextSparseMoeBlock"]` lists only the direct
siblings (`moe_router`, `moe_experts`). New `_BREAKDOWN_NESTED_OF` maps
`moe_experts -> (moe_shared,)`. Reporter renders nested subs indented as
`+-- moe_shared X.XXX ms/token (YY% of moe_experts)` plus a derived
`moe_experts (excl. nested)` row.

One-shot rank-0 runtime dump under `VLLM_V100_FP8_DEBUG_SHARED_EXPERTS=1`
prints `Qwen2MoeMLP` structure on the first MoE-block forward (post-PWAL):
type, child types, quant_method (incl. is_fp8, block_quant, block_size),
weight dtype/shape/contiguity, and whether `Fp8LinearMethod.apply` was
swapped to our patched function.

Definitive runtime evidence (rank 0 dump):

| Child | Class | Quant method | dtype | Shape | Notes |
|---|---|---|---|---:|---|
| `gate_up_proj` | `MergedColumnParallelLinear` | `Fp8LinearMethod` | `float8_e4m3fn` | (256, 3072) | `block_quant=True, block_size=[128, 128]` |
| `down_proj` | `RowParallelLinear` | `Fp8LinearMethod` | `float8_e4m3fn` | (3072, 128) | `block_quant=True, block_size=[128, 128]` |
| `act_fn` | `SiluAndMul` | n/a | - | - | - |
| `expert_gate` | `ReplicatedLinear` | `UnquantizedLinearMethod` | `float16` | (1, 3072) | conditional sigmoid weight |

`Fp8LinearMethod.apply patched_by_v100=True`. **Classification: already-FP8.**
Shared experts run through `_v100_fp8_gemm` (same path as the routed MoE
inner GEMM). No optimization wedge there; the 26 ms/token is structural M=1
memory-bound work over two block-FP8 Linears across 48 layers.

`intermediate=128` per TP=8 shard for the shared expert (= per-rank
`shared_expert_intermediate_size / TP`), matching the routed-expert
intermediate observed in the Stage 2B grouped log.

**Step 2B.1: sub-attribution of the `moe_other` residual.** Monkey-patched
`Qwen3NextSparseMoeBlock.forward` with measurement-only CUDA-event timers
bracketing the combine and all-reduce sections (semantics identical to
upstream; gated by `VLLM_V100_FP8_MOE_OTHER_PROFILE=1`):

- `moe_other_combine` = the `final_hidden_states[0] + final_hidden_states[1]`
  fp16 add (routed + shared).
- `moe_other_allreduce` = `self.experts.maybe_all_reduce_tensor_model_parallel(...)`
  or the sequence-parallel all-gather branch; same bucket name either way.
- `moe_other_residual` = derived; covers Python view/reshape, attribute
  lookups, dispatch glue.

Resulting decomposition at steady-state decode (tokens=576 sample, profile-on,
3 instrumentation layers all on):

```
Qwen3NextSparseMoeBlock          127.555 ms/token  (55.8% of decode)
  moe_router                       1.487
  moe_experts                     69.283
    moe_shared (nested)           26.163
    moe_experts (excl. nested)    43.120
  moe_other                       56.784
    moe_other_combine              3.001  ( 5.3% of moe_other)
    moe_other_allreduce           50.431  (88.8% of moe_other)
    moe_other_residual             3.352  ( 5.9% of moe_other)
GDN                               67.690  (29.6%)
Attention                         30.251  (13.2%)
LogitsProcessor                    0.955
Other (residual)                   2.169
Total                            228.620  ms/token
```

**Headline finding: `moe_other` is 89% TP all-reduce.** 48 NCCL all-reduce
calls per token on a `[1, 3072]` fp16 buffer (~6 KB) at ~1.05 ms each.
Python-wrapper overhead is only ~6 ms/token (combine + residual); the
wrapper-bypass lever from earlier Stage 2D planning is **dropped from the
active list** -- too small to justify a SparseMoeBlock.forward replacement.

**Source read (Stage 2D Step 2C.1) on the all-reduce path:**

`--disable-custom-all-reduce` is **dead weight** on this DGX-1 V100 host.
vLLM 0.18's `CustomAllreduce` auto-disables at runtime when
`current_platform.is_fully_connected(physical_device_ids)` is False, which
requires 1-hop NVLink between every TP rank pair. `nvidia-smi topo -m`
shows the typical DGX-1 hypercube -- two NVLink quadrants ({0,1,2,3},
{4,5,6,7}) with several cross-quadrant pairs at NODE (PCIe-through-host-bridge),
not NVLink. So the topology gate trips at TP=8 regardless of the flag.

Removing the flag is therefore a no-op; the AR still goes through NCCL on
the mixed-topology ring/tree algorithm. The 50 ms/token cost is small-message
NCCL latency, not bandwidth. Not addressable by re-enabling custom AR at
TP=8 on this host.

**`SymmMemCommunicator: Device capability 7.0 not supported` is a separate
warning** (sm_90+ symmetric-memory communicator), unrelated to custom AR
availability.

`tools/bench_v100.sh` was updated to drop `--disable-custom-all-reduce`
from the default serve command (replaced with an explanatory comment).
Trailing positional args still allow it to be added back per-run if a
future host needs it.

**Stage 2D Step 2C.2 (2026-05-25, GPT-fired, Claude cross-checked):** no-code A/B
confirmed empirically. Run dir
`/tmp/v100_bench/20260525_095017_stage2d_step2c2_no_disable_ar/`, built from
commit `e7c81e1` (= v0.3.1), serve command verified flag-free in
`config.txt`. Engine config logged `disable_custom_all_reduce=False` and
all 8 worker ranks emitted the explicit warning at boot:

```
WARNING [custom_all_reduce.py:154] Custom allreduce is disabled because
it's not supported on more than two PCIe-only GPUs. To silence this
warning, specify disable_custom_all_reduce=True explicitly.
```

| Metric | 2B.1 (flag ON) | 2C.2 (flag OFF) | Δ |
|---|---:|---:|---:|
| Mean tok/s (2 curls) | 4.064 | 4.116 | +1.3% |
| Total ms/token (decode) | 228.620 | 225.654 | -1.3% |
| `moe_other_allreduce` ms/token | 50.431 | 47.985 | -4.9% |
| `moe_other_allreduce` share of `moe_other` | 88.8% | 88.7% | -0.1pp |
| MoE block ms/token | 127.555 | 125.599 | -1.5% |
| GDN ms/token | 67.690 | 67.160 | -0.8% |
| Coherence (Paris loop) | pass | pass | - |

All differences within the 2-3% serve-restart noise band documented in
Stage 2C. Flag is confirmed empirically dead weight on this DGX-1 V100
TP=8 host. The `tools/bench_v100.sh` flag drop landed in v0.3.1 is
correct as-is; no further code change needed.

**Stage 2D Step 2C closed.** Next: Step 2D source-read of GDN
(`Qwen3NextGatedDeltaNet`) — 67 ms/token, 29.6% of decode, now the
largest remaining unexplained lever.

**Stage 2D Step 2D** (pending): pivot to GDN
(`Qwen3NextGatedDeltaNet`, ~68 ms/token, 29.6% of decode) which is now the
largest unexplained lever after the AR finding. Per-AR latency at TP=8 on
this topology is topology-limited; non-trivial optimization paths
(structural AR-frequency reduction, custom-AR-with-NVLink TP=4 within a
quadrant) are documented but not on the Stage 2D plan.

**Param hygiene** as a separate later pass: this stage proved
`--disable-custom-all-reduce` is sediment. Other candidates that have
accumulated by trial-and-error and not been validated:
`--no-enable-chunked-prefill`, `--attention-backend TRITON_ATTN`,
`--max-num-seqs 1`, `--gpu-memory-utilization 0.80`. Only `--enforce-eager`
is documented as load-blocking-when-removed.

New env vars added in v0.3.1 (all measurement-only, default off unless noted):

| Var | Default | Purpose |
|---|---|---|
| `VLLM_V100_FP8_DECODE_BREAKDOWN_MOE_SUBS` | 1 (when DECODE_BREAKDOWN=1) | Step 1: hook gate / experts / shared_expert sub-modules of every SparseMoeBlock |
| `VLLM_V100_FP8_DEBUG_SHARED_EXPERTS` | 0 | Step 2A: one-shot rank-0 dump of the shared expert runtime structure |
| `VLLM_V100_FP8_MOE_OTHER_PROFILE` | 0 | Step 2B.1: monkey-patch SparseMoeBlock.forward with CUDA events around combine + all-reduce |

## Stage 2D Step 2D.1/2D.2/2D.3 (2026-05-25): GDN attribution + cross-cutting TP all-reduce finding

Tagged at `v0.3.2`. Pure instrumentation; no behavior change vs v0.3.1.
Decisive finding: **TP all-reduce dominates decode at ~41.5% of profile-on
wall (≈96 NCCL calls/token × ~1.08 ms each)**, topology-limited at TP=8
on this DGX-1 V100 hypercube. Stage 2D closes as a bottleneck-map
release; the AR bottleneck is structural and out of Stage 2D scope to
attack.

**Step 2D.1: source-read `Qwen3NextGatedDeltaNet`.** Decode call graph:

```
forward(hidden_states, output):
  Part 1: in_proj_qkvz(...) [FP8, our patch]
          in_proj_ba(...)   [block-FP8 alignment mismatch → FP16 fallback]
          Python rearranges + cat
  Part 2: torch.ops.vllm.gdn_attention_core(...) → self._forward_core(...):
          self.conv1d.weight.view(...)   [weight read, conv1d.forward NEVER called]
          causal_conv1d_update(...)      [FLA Triton, decode variant]
          rearrange_mixed_qkv(...)
          fused_sigmoid_gating_delta_rule_update(...)  [FLA Triton, decode variant]
  Part 3: norm(...)         [RMSNormGated]
          rearranges
          out_proj(...)     [RowParallelLinear, FP8 + hidden all-reduce]
```

Linears in GDN: `in_proj_qkvz` (FP8, our patch, heavy), `out_proj`
(RowParallel FP8 + AR, heavy), `in_proj_ba` (FP16 fallback, small),
`conv1d` (FP16, never called via forward at decode). The two FLA Triton
kernels (`causal_conv1d_update`, `fused_sigmoid_gating_delta_rule_update`)
are third-party.

**Step 2D.2: read-only sub-attribution of GDN.** Instance-tagged forward
hooks on `in_proj_qkvz`, `in_proj_ba`, `norm`, `out_proj` (no `conv1d`
hook — its forward is never invoked at decode/prefill) plus a
class-method monkey-patch of `Qwen3NextGatedDeltaNet._forward_core` tagged
`gdn_core`. Gated by `VLLM_V100_FP8_DECODE_BREAKDOWN_GDN_SUBS=1` (default
on when DECODE_BREAKDOWN is set). Also added a per-section residual
label map so `Qwen3NextGatedDeltaNet`'s residual prints as
`gdn_other (rearrange/cat/residual)` rather than the MoE label.

First V100 measurement (run dir
`/tmp/v100_bench/20260525_103059_stage2d_step2d2_gdn_subs/`) overturned
the "FLA dominates" hypothesis from Step 2D.1:

```
Qwen3NextGatedDeltaNet           83.886 ms/token (34.3% of decode)
  + gdn_in_qkvz                   3.826  ( 4.6% of GDN)
  + gdn_in_ba                     1.657  ( 2.0%)
  + gdn_core                     17.577  (21.0%)
  + gdn_out_proj                 40.178  (47.9%)   <-- biggest, unexpected
  + gdn_other                    20.648  (24.6%)
```

`gdn_out_proj` at 47.9% of GDN, `gdn_core` (the FLA Triton kernels we
feared) at only 21%. Arithmetic from `gdn_in_qkvz` (ColumnParallel, no
AR) = 3.826/36 = ~0.106 ms per FP8 GEMM at M=1, but `gdn_out_proj`
(RowParallel) = 40.178/36 = ~1.116 ms per call. Δ ≈ 1.01 ms — matching
the ~1.05 ms MoE-block all-reduce latency we already measured.

Source-read of `vllm/model_executor/layers/linear.py:1498-1525` confirmed:

```python
def forward(self, input_):
    ...
    output_parallel = self.quant_method.apply(self, input_parallel, bias_)
    if self.reduce_results and self.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output_parallel)
    ...
```

**Every `RowParallelLinear` at TP>1 does an internal NCCL all-reduce
after the GEMM.** `reduce_results` defaults to True. Our forward hook on
`gdn_out_proj` captures both as a single bucket. The Qwen3-Next attention
block also uses `RowParallelLinear` for its `out_proj`, so its bucket
similarly conflates GEMM + AR.

**Step 2D.3: cross-cutting `row_parallel_ar` instrumentation.** To verify
the AR-dominance hypothesis without per-Linear attribution overhead,
monkey-patched `RowParallelLinear.forward` with a verbatim mirror of
upstream (same input split, same GEMM, same bias) that brackets only the
`tensor_model_parallel_all_reduce(output_parallel)` call with CUDA
events. Single aggregated bucket `row_parallel_ar` across all
RowParallelLinear instances. Gated by
`VLLM_V100_FP8_ROW_PARALLEL_AR_PROFILE=1`. Rendered in the breakdown as
a **cross-cutting attribution line** below `Total` with an explicit
`[cross-cutting attribution; already counted in module buckets above]`
annotation, NOT summed into the per-token total. `tools/bench_v100.sh`
extract grep extended to capture `row_parallel_ar` + the cross-cutting
annotation + Stage 2D Step banners.

V100 measurement (run dir
`/tmp/v100_bench/20260525_105438_stage2d_step2d3_row_ar/`):

```
Total                           250.555 ms/token  (100.0%)
  Qwen3NextSparseMoeBlock         129.054  (51.5%)
    + moe_router                    1.590
    + moe_experts                  70.020
        +-- moe_shared             26.254
        +-- moe_experts excl       43.766
    + moe_other                    57.443
        +-- moe_other_combine       2.968
        +-- moe_other_allreduce    51.182  (89.1% of moe_other)
        +-- moe_other_residual      3.293
  Qwen3NextGatedDeltaNet           87.020  (34.7%)
    + gdn_in_qkvz                   3.630
    + gdn_in_ba                     1.131
    + gdn_core                     15.853
    + gdn_out_proj                 46.770  (53.7% of GDN; mostly AR)
    + gdn_other                    19.636
  Qwen3NextAttention               30.849  (12.3%)
  LogitsProcessor                   0.948
  Other (residual)                  2.686
  [cross-cutting attribution; already counted in module buckets above]
  row_parallel_ar                  52.703  (21.0% of total) calls=1536
```

Calls/token from `calls=1536` over the 32-token window:
`row_parallel_ar` = 48/token (36 GDN out_proj + 12 attention out_proj),
`moe_other_allreduce` = 48/token (one per MoE block).

**Total visible TP all-reduce:**

```
moe_other_allreduce  51.182 ms/token (48 calls/tok, ~1.066 ms each)
row_parallel_ar      52.703 ms/token (48 calls/tok, ~1.099 ms each)
─────────────────────────────────────────────────────────────────
total                103.885 ms/token = 41.5% of profile-on decode
                     96 NCCL all-reduces/token at ~1.08 ms each on 6 KB
```

**Stage 2D bottleneck map (final, profile-on, 250 ms/token total):**

| Bucket | ms/token | % of decode | Optimization domain |
|---|---:|---:|---|
| **TP all-reduce (combined)** | **103.9** | **41.5%** | **structural, topology-limited at TP=8** |
|   moe_other_allreduce | 51.2 | 20.4% | explicit `maybe_all_reduce_tensor_model_parallel` |
|   row_parallel_ar | 52.7 | 21.0% | hidden inside `RowParallelLinear.forward` |
| moe_experts routed-only | 43.8 | 17.5% | our FP8 path, near floor at M=1 |
| Attention (incl. ~12 ms AR) | 30.8 | 12.3% | Triton attention; AR portion already in row_parallel_ar |
| moe_shared (FP8 MLP) | 26.3 | 10.5% | structural memory-bound on our path |
| gdn_other (rearrange/cat/norm/python) | 19.6 | 7.8% | misc; small targets |
| gdn_core (FLA Triton conv1d + recurrent) | 15.9 | 6.3% | third-party Triton |
| GDN out_proj GEMM (≈ 46.8 - AR portion) | ~7 | ~2.8% | our FP8 path |
| moe_other_combine + residual | 6.3 | 2.5% | python wrap |
| moe_router + Logits + Other + small | ~8 | ~3.2% | free |

Every other named bucket is ≤17.5%. The MoE inner GEMM (routed) and
shared experts are already on our optimized FP8 path. Attention and GDN
core kernels are third-party Triton. Wrappers and Python are tiny.

**Future levers (documented, deferred):**

1. **TP=4 single-quadrant test.** GPUs 0-3 (or 4-7) form a fully NVLink-
   connected subset on this DGX-1 hypercube. At TP=4, vLLM's
   `CustomAllreduce` would auto-enable (`is_fully_connected` returns
   True), bypassing NCCL latency on small messages. Halves the per-rank
   model shard, doubling per-rank memory pressure on already-tight
   32 GB V100s — Qwen3.5-122B-A10B-FP8 weights ~30 GB/rank at TP=4,
   leaving <2 GB for KV cache and runtime. Feasibility: TBD; separate
   memory-bound experiment with reduced `max_model_len`.
2. **Custom V100 small-message AR.** A bespoke CUDA IPC + warp-level
   collective tailored to 6 KB fp16 messages on partial-NVLink topology.
   Engineering project; would need to upstream into vLLM's communicator
   abstraction.
3. **Reducing AR frequency.** Batch ARs across adjacent transformer
   blocks. Each TP-sharded operator's correctness depends on its own
   AR; batching is invasive in vLLM's per-layer dispatch model.
4. **GDN/attention Triton kernel work.** Smaller pool combined
   (~22% of decode if you sum `gdn_core` + attention compute), and
   third-party.

New env vars added in v0.3.2 (all measurement-only):

| Var | Default | Purpose |
|---|---|---|
| `VLLM_V100_FP8_DECODE_BREAKDOWN_GDN_SUBS` | 1 (when DECODE_BREAKDOWN=1) | Step 2D.2: hook in_proj_qkvz / in_proj_ba / norm / out_proj on every Qwen3NextGatedDeltaNet + monkey-patch `_forward_core` for `gdn_core` bucket |
| `VLLM_V100_FP8_ROW_PARALLEL_AR_PROFILE` | 0 | Step 2D.3: monkey-patch `RowParallelLinear.forward` to time only the AR call; rendered as cross-cutting bucket below the breakdown |

**Stage 2D is closed.** The bottleneck map is fully evidenced. The
remaining decode wall is dominated by a topology-limited communication
cost that is not addressable without structural changes to either the TP
configuration (memory feasibility) or vLLM's per-block AR dispatch
(invasive). Stage 2E candidates (TP=4 memory feasibility, param hygiene
pass) are separate stages.

---

## Stage 3 (v0.4.0, 2026-05-26): Production breakthrough — Qwen3.5-122B-A10B-FP8 TP=8 at 34.76 tok/s

### Headline

`Qwen3.5-122B-A10B-FP8` on 8× V100 TP=8 sustained **34.76 tok/s** decode, σ ≈ 10 ms across 4 curls. **6.83× over the v0.3.2 cu128+eager baseline of 5.09 tok/s** that Stage 2D closed against. Comfortably above the 20 tok/s shippable floor; in the user's "30+ comfortable" band.

`Qwen3.6-35B-A3B-FP8` (same arch family, TP=4) sustained **52.87 tok/s**, **8.05× over its own eager-mode baseline of 6.57 tok/s** measured this stage.

Quality validated on both — full 200-token Rayleigh-scattering paragraph completions, capability-grade output, no garbage / corruption.

### What changed: a new image

`docker/Dockerfile.vllm018_py312` builds `vllm-v100-py312-test:cu128`. Held constant from the legacy cu128 image: vLLM 0.18.0, torch 2.10.0+cu128, transformers 4.57.6, our `fp8_w8a16_sm70` monkey-patches. One variable bumped: **Python 3.10 → 3.12** (forced ubuntu 22.04 → 24.04 base because deadsnakes was too brittle for the cudagraph path verification).

Launcher: `docker/run_docker_vllm018_py312.sh` with a new `serve-fp8` mode that mirrors `run_docker.sh:serve` — mounts repo at `/work`, exports all `VLLM_V100_FP8_*` env vars, invokes `python3 -m fp8_w8a16_sm70.vllm_serve`. The non-FP8 `serve` mode (plain `vllm serve`) is retained for cross-stack baselining.

### Three findings that compounded into the 6.83×

**1. Python ≤3.10 was the cudagraph blocker, not vLLM 0.18 or torch 2.10.**

Source comment at `vllm/compilation/compiler_interface.py:377` admits the `standalone_compile.FakeTensorMode` AttributeError is "Python ≤3.10 specific" — the `patch.object()` string resolver returns the wrapper function instead of the module on Py3.10. Stage 2D had us forcing `--enforce-eager` to dodge the error; we assumed it was a vLLM internals bug. It wasn't. Holding vLLM 0.18 + torch 2.10 constant and bumping Python alone fixed it. Verified end-to-end on Qwen2.5-7B-Instruct first (45.64 tok/s on stock vLLM 0.18 + py3.12 + cudagraph; statistically identical to 1catai-vllm's 45.32 measured separately).

**2. `VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1` inverts under cudagraph.**

Stage 2C A/B testing on cu128+eager had us default the env var to 0 because fast=1 was 1.3% slower. Under cudagraph capture, the trade-off **inverts completely**. The slow path uses `torch.nonzero(valid_mask, as_tuple=True)` (line 1436, data-dependent output shape) and `int(token_idx.numel())` (line 1437, GPU→CPU sync via `__int__`). Both are fatal under stream capture — emit `cudaErrorStreamCaptureUnsupported`. The fast path, built in Stage 2C (`_get_token_idx_cached`, `route_count = M*topk`, cached `arange`), is static-shape end-to-end. **The Stage 2C-era code path that we shelved as marginally slower for eager is the production code path for cudagraph.** Big lesson: never delete code paths whose value depends on a runtime mode you haven't enabled yet.

**3. `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'` is required for monkey-patches + cudagraph.**

vLLM's default `VLLM_COMPILE` mode (3) requires `fullgraph=True` — no graph breaks anywhere in the model forward. That forbids `torch._dynamo.disable` around our pybind11 extension entry points (`_ext.fp8_w8a16_gemm_a1/a2/a3/wmma_poc/grouped_routed_gemm_a3`). The fix path that actually works: skip dynamo entirely (mode=0), keep cudagraph capture on the decode hot path (FULL_DECODE_ONLY). Loses torch.compile fusion (small loss on MoE; would be a large loss on dense models) but unblocks the FP8 path. The proper long-term cleanup is registering the 5 extension entry points as `torch.library.custom_op` with fake/meta impls, which lets us run mode=3 + FULL_AND_PIECEWISE later. Deferred — current numbers already cleared the production floor.

### Configuration knobs that emerged

| Env var | v0.4.0 cudagraph default | Why differs from cu128 eager |
|---|---|---|
| `VLLM_V100_FP8_MOE_FAST_ROUTE_PREP` | **1** (was 0) | Slow path's `torch.nonzero` / `int(numel())` syncs are forbidden under capture; fast path's static-shape route prep is the only safe one. |
| `VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM` | 1 (unchanged) | Grouped path is cudagraph-clean; fallback `_our_moe_apply` is not (`.tolist()` at line 1904, `.item()` at 1921/1925). |
| `VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS` | **128** (was 32) | With `max-num-seqs=8` and top-k=8, route_slots can hit 64 (or 128 if capture set includes M=16). Default 32 caused fallback into the cudagraph-unsafe path. |

Plus the vLLM-side `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'` flag (above).

### Cross-stack methodology / dead-end branches

This stage explored two parallel hypotheses before landing on stock vLLM 0.18 + py3.12 as the winner:

**1catai-vllm v1.0.0 fork** — Built `docker/Dockerfile.1catai` + `run_docker_1catai.sh` + `tools/bench_1catai.sh`. Their stack ships FA2-v100 attention, native SM70 FP8 MoE method (TurboMind s884), MTP speculative decoding. Empirical findings:

- Loads cleanly on dense models (Qwen2.5-7B, Qwen3.6-27B Dense).
- **`Qwen3_5MoeForConditionalGeneration` arch fails to load** — their loader expects `language_model.` prefix on weight keys; on-disk checkpoint has flat keys. Stock vLLM 0.18 doesn't have this bug.
- **FP8 capability gate at `fp8.py:157`** blocks all FP8 models — same as stock vLLM but 1catai's `Fp8SM70MoEMethod` was never reachable through it.
- **FA2-v100 is 2.65× SLOWER than TRITON_ATTN on hybrid attn+GDN models** (Qwen3.6-27B Dense: 14.97 vs 39.60). This is the "GDN dichotomy" — FA2-v100 helps pure-dense-attention models (e.g. Qwen3-30B-A3B-2507 at 12.63 tok/s, beating our 9.67 on the same model), but hurts hybrid models.

For our Qwen3.5/3.6-family production target (with GDN), 1catai is not the right base. Its remaining unique value is MTP speculative decoding — deferred for evaluation as an orthogonal 2-3× optimization on top of v0.4.0.

**Refactor `_our_moe_apply_grouped` for cudagraph safety** — Briefly considered when the grouped path hit `cudaErrorStreamCaptureUnsupported`. GPT (peer-review) correctly identified `torch.nonzero` at line 1436 as the actual culprit, not the grouped GEMM kernel itself. After source inspection we found that **a pre-existing fast path (Stage 2C, `_MOE_FAST_ROUTE_PREP=1`) was already static-shape and capture-safe** — no refactor needed, just an env var flip. This was the cheapest win of the whole stage; would have been a ~2-4 hour refactor if we'd gone straight to writing new code.

### Cross-model results table

| Model | Stack | TP | tok/s | Shippable (≥20)? |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct (toy) | py3.12 + cudagraph | 1 | 45.64 | ✅ (not deployment-grade capability) |
| Qwen3.6-27B Dense FP16 | py3.12 + cudagraph (FULL_AND_PIECEWISE) | 4 | 39.60 | ✅ |
| Qwen3.6-27B-FP8 | py3.12 + cudagraph (FULL_DECODE_ONLY) + monkey-patches | 4 | 11.36 | ❌ (dense FP8 dequant cost ≈ FP8 BW saving) |
| Qwen3.6-35B-A3B FP16 | py3.12 + cudagraph (FULL_AND_PIECEWISE) | 4 | 15.76 | ❌ |
| **Qwen3.6-35B-A3B-FP8** | **py3.12 + cudagraph (FULL_DECODE_ONLY) + monkey-patches** | **4** | **52.87** | **✅** |
| **Qwen3.5-122B-A10B-FP8 (prod target)** | **py3.12 + cudagraph (FULL_DECODE_ONLY) + monkey-patches** | **8** | **34.76** | **✅✅✅** |
| Qwen3-30B-A3B-Instruct-2507 (no-GDN MoE) | py3.12 + cudagraph | 4 | 9.67 | ❌ (1catai+FA2 wins on no-GDN, gives 12.63) |

### What's still on the table for Stage 3.5+ (deferred)

- **`torch.library.custom_op` proper registration** for the 5 `_ext.*` entry points (with fake/meta impls). Unlocks `mode=3 + FULL_AND_PIECEWISE` and brings torch.compile fusion back. Estimated +20-40% on top of 34.76 tok/s.
- **Stage 2D bottleneck profiling on v0.4.0** — re-run the AR-percentage measurement; on py3.12+cudagraph, AR fraction should be much higher relative to compute (since compute compressed 6.83×). If AR is now >60% of decode, then **MTP via 1catai becomes the natural next investment** (speculative decode amortizes AR cost across multiple tokens).
- **V100 MoE config autotune** — vLLM ships no `E=128/N=192/Tesla_V100-SXM2-32GB.json`; default config is Ampere/Hopper-tuned. Should be +20-40% on the MoE path.
- **Determinism investigation** — observed minor newline-pattern variance between curls 1-2 (\n\n) vs 3-4 (\n) at temp=0 on the 122B. Likely a captured-graph dispatch quirk between batch-size slots; not affecting semantic quality. Worth investigating before any A/B that depends on bit-level reproducibility.
- **1catai MTP probe** — if Stage 3.5 needs another 2-3×, evaluate 1catai with MTP after patching their `language_model.` loader. Standalone investment, ~1-2 hours of source work + wheel rebuild.

### Files shipped in v0.4.0

| File | Purpose |
|---|---|
| `docker/Dockerfile.vllm018_py312` | py3.12 + cu128 image (new) |
| `docker/run_docker_vllm018_py312.sh` | Launcher with `build`/`shell`/`serve`/`serve-fp8`/`py` modes (new) |
| `docker/Dockerfile.1catai` | 1Cat-vLLM v1.0.0 wheel image (new, used for cross-stack baselining only) |
| `docker/run_docker_1catai.sh` | 1catai launcher (new) |
| `tools/bench_1catai.sh` | 1catai bench helper paralleling `bench_v100.sh` (new) |
| `docker/run_docker.sh` | Port-mapping fix `${PORT}:8000` → `${PORT}:${PORT}` (modified) |
| `tools/bench_v100.sh` | `grep ... \|\| true` on empty env scan; `ENFORCE_EAGER` env var (modified) |
| `src/fp8_w8a16_sm70/vllm_serve.py` | `torch._dynamo.disable` block for `_ext.*` (modified; harmless on eager, may be removed once custom_op refactor lands) |
| `pyproject.toml` | 0.3.2 → 0.4.0 (modified) |

---

## Stage 3.1 (2026-05-26, post-v0.4.0): Dense FP8 diagnosis — three-way 27B comparison and deployment rule

After Stage 3 production breakthrough landed, ran a controlled three-way comparison on Qwen3.6-27B (dense + GDN hybrid, TP=4) to understand why dense FP8 was the *only* low number on the v0.4.0 stack:

| 27B variant (TP=4) | Stack | tok/s | bytes/weight | vs FP16 |
|---|---|---|---|---|
| **FP16** (no quant) | py3.12 + cudagraph FULL_AND_PIECEWISE | 39.60 | 2.0 | 1.00× baseline |
| **GPTQ-Int4** (stock vLLM, exllama path) | py3.12 + cudagraph FULL_DECODE_ONLY | **47.61** | 0.5 | **1.20× faster** |
| **our FP8** (block-FP8 W8A16) | py3.12 + cudagraph FULL_DECODE_ONLY + monkey-patches | 11.44 | 1.0 | **0.29× — 3.46× SLOWER** |

Bandwidth math predicts Int4 ~2× faster than FP8 on memory-bound decode. We observe Int4 4.16× faster — the extra ~2× is our path's overhead. **Same FP8 path wins on MoE+GDN by huge margins (35B-A3B-FP8 52.87, 122B-A10B-FP8 34.76) because only ~3B active params/token are touched, so per-GEMM dispatch and dequant amortize over far fewer calls.**

GPT independent review confirmed the diagnosis and proposed a concrete three-step microbench (saved in `docs/STAGE_3.1_NEXT_STEPS.md`):
1. Direct `_ext.fp8_w8a16_gemm_*` timing vs cuBLAS FP16 on dense 27B shapes.
2. Python wrapper passthrough probe to isolate `_v100_fp8_gemm` dispatch cost.
3. Three-branch attribution: kernel-itself slow → not worth fixing; wrapper slow → custom_op refactor unlocks fix; both fine → fusion/graph behavior is the cost.

### Deployment rule (settled by data, endorsed by both Claude and GPT)

| Workload | Recommended stack | tok/s |
|---|---|---|
| 122B / 35B-A3B / any MoE+GDN | **our FP8** OR **GPTQ-Int4** if available | 34.76 / 52.87 (our FP8); 63.62 (Int4) |
| Dense+GDN (27B-class) | **GPTQ-Int4** or **FP16**, NOT our FP8 | 47.61 / 39.60 |
| Small dense | FP16 or Int4, NOT our FP8 | — |

### Failed 27B-FP8 profile run — also informative

Attempted to enable Stage 2D breakdown profiling under cudagraph for diagnosis. Crashed with `cudaErrorInvalidValue` at first decode. Root cause: `_breakdown_post_hook` (vllm_serve.py:492) and `_breakdown_flush` (vllm_serve.py:502) call `torch.cuda.Event(enable_timing=True).elapsed_time()`, which triggers a host-side read of GPU state mid-stream-capture → forbidden. Stage 2D hooks are eager-only by construction. Cleanup queued: add `torch.cuda.is_current_stream_capturing()` guard to make them safe under both modes.

### Upgrade ladder for further perf (post-v0.4.0, in priority order)

Per GPT review, treat each as a controlled-variable change with the same three-test smoke matrix at each rung (122B-FP8 ≥34 / 35B-A3B-FP8 ≥52 / 27B sanity ≥39):

1. **Driver bump** — current likely 535. Candidates R570/R575/R580. Avoid CUDA-13-implying combos (drops sm_70). Safest; expected modest gain via NCCL/CUDA-graph runtime improvements.
2. **CUDA runtime image** — cu129 if wheels available. Do NOT try CUDA 13.
3. **torch upgrade** — only after rung 1+2 stable. Pre-check: `torch.cuda.get_arch_list()` must include sm_70.
4. **vLLM upgrade** — highest risk; internal API changes likely break our monkey-patches. Treat as branch, not default.

Full plan in `docs/STAGE_3.1_NEXT_STEPS.md`.

### Files added/modified in Stage 3.1

| File | Purpose |
|---|---|
| `docs/STAGE_3.1_NEXT_STEPS.md` | Next-session action plan: dense-FP8 microbench protocol + controlled upgrade ladder + open Stage 3.5+ items (new) |
| `docs/SESSION_LOG.md` | This Stage 3.1 closeout section (modified) |

End of log.

## 2026-05-26 Dense FP8 diagnostic handoff

Reference question file: `/tmp/v100_bench/GPT_question_dense_fp8_diagnostic_20260526.md`.

Current production rule is now evidence-backed:

| Workload | Recommended path |
|---|---|
| Qwen3.5/Qwen3.6 MoE+GDN FP8 | `fp8_w8a16_sm70` on py3.12 vLLM 0.18, `mode=0`, `FULL_DECODE_ONLY`, grouped MoE, fast route prep |
| Dense/GDN 27B-class | GPTQ-Int4 if available; otherwise FP16 |
| Small dense | FP16 or Int4; not the custom FP8 path |

Dense FP8 remains the open diagnostic:

| Qwen3.6-27B variant | TP | tok/s |
|---|---:|---:|
| FP16 stock | 4 | 39.60 |
| GPTQ-Int4 stock / exllama | 4 | 47.61 |
| Custom block-FP8 W8A16 | 4 | 11.44 |

This means the custom dense FP8 path is not merely losing the expected FP8-vs-Int4 bandwidth ratio; it is losing an extra roughly 2x beyond bandwidth expectation. The issue appears specific to our dense FP8 implementation, not a structural V100 limit, because stock Int4 is net-positive on the same model, stack, and hardware.

Most plausible causes to test, in priority order:

1. Python/per-call wrapper overhead in `_v100_fp8_gemm`.
2. Kernel-level dequant+GEMM cost in `fp8_dequant.cu`.
3. Lost compiler/fusion benefits from `mode=0` versus full vLLM compile.
4. Captured-graph node replay overhead for many small custom extension calls.

Cheapest next diagnostic should be a microbenchmark rather than a model refactor. Preferred sequence:

1. Isolated kernel loop: call `_ext.fp8_w8a16_gemm_a1/a2/a3/wmma_poc` directly on representative dense/GDN shapes and compare against FP16 `torch.matmul`/cuBLAS on the same shapes. This separates CUDA kernel cost from vLLM/Python/model overhead.
2. Python wrapper overhead probe: time `_v100_fp8_gemm` versus direct `_ext.*` calls in a tight loop with already-allocated inputs. This quantifies pure Python dispatch/variant-selection cost.
3. Optional no-op monkey-patch: replace `_v100_fp8_gemm` with a shape-correct precomputed output for one local experiment only. This bounds all wrapper+kernel overhead inside the full model, but is less clean because correctness is intentionally broken.

Do not spend engineering time on dense FP8 fixes before those measurements. The likely deployment answer is still "use Int4 or FP16 for dense V100 models; reserve custom FP8 for MoE+GDN where grouped MoE+cudagraph wins decisively."

## 2026-05-26 Final session wrap-up

Both peer reviews converged on the same deployment rule and next-session plan.

### Deployment rule

Do **not** use the custom block-FP8 W8A16 path for dense+GDN models on V100 unless a later microbench reveals a cheap implementation bug. Current evidence says dense FP8 is dominated by our dequant/custom-kernel path and is much slower than both FP16 and stock GPTQ-Int4.

Use:

| Workload | Preferred serving path |
|---|---|
| Qwen3.5/Qwen3.6 MoE+GDN FP8 | custom `fp8_w8a16_sm70` on py3.12 vLLM 0.18, `mode=0`, `FULL_DECODE_ONLY`, grouped MoE, fast route prep |
| Dense+GDN 27B-class | GPTQ-Int4 if available; otherwise FP16 |
| Small dense | FP16 or Int4 |

### Dense FP8 diagnostic protocol

If we revisit dense FP8, start with microbenchmarks, not refactors:

1. Direct `_ext.fp8_w8a16_gemm_*` timing against FP16 `torch.matmul` on representative Qwen3.6-27B shapes.
2. `_v100_fp8_gemm` wrapper timing against direct `_ext.*` calls to isolate Python dispatch and variant-selection overhead.
3. Optional one-off no-op/full-model probe to bound total wrapper+kernel cost inside vLLM.

Interpretation:

- Direct `_ext.*` slow: kernel/dequant implementation is the bottleneck.
- Direct `_ext.*` fast but wrapper slow: Python dispatch/variant selection is the bottleneck.
- Both fast in isolation but model slow: graph replay/fusion/model-level call pattern is the bottleneck.

### Baseline shift

The performance baseline is now:

- Python 3.12
- vLLM 0.18.0
- torch 2.10.0+cu128
- CUDA 12.x runtime
- `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`
- `VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1`
- `VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=128`
- `VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1`

Legacy Python 3.10 / `--enforce-eager` remains a correctness fallback, not the performance baseline.

### Upgrade ladder for next session

Change one layer at a time:

1. Driver upgrade first.
2. CUDA runtime / torch wheel second.
3. Torch version next, only if `torch.cuda.get_arch_list()` still includes `sm_70`.
4. vLLM version last; this is highest risk because monkey-patch APIs may move.

At every rung, rerun the same smoke/perf matrix:

| Test | Purpose |
|---|---|
| Qwen3.5-122B-A10B-FP8 TP=8 | production target; must stay near or above 34.76 tok/s |
| Qwen3.6-35B-A3B-FP8 TP=4 | smaller MoE+GDN canary; must stay near or above 52.87 tok/s |
| Qwen3.6-27B FP16 / GPTQ-Int4 TP=4 | dense sanity check |

Avoid CUDA 13 for V100/sm_70. Treat any torch/vLLM upgrade as a branch until the matrix proves it preserves the production target.

### Baseline-prep continuation

Repo-facing defaults now point at the py3.12 v0.4.0 baseline:

- `README.md` rewritten from the original FP8 hello-world note to the current serve/deployment baseline.
- `REQUIREMENTS.md` now lists `FULL_DECODE_ONLY` cudagraph as the performance path and `--enforce-eager` as the legacy fallback.
- `docker/run_docker_vllm018_py312.sh` now defaults to `VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=128` and `VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1`.
- `docker/Dockerfile.vllm018_py312` comments now describe the image as the baseline, not only an exploratory test image.

## Stage 3.6 prep (2026-05-26 evening): cu129 sibling image, controlled A/B, driver hygiene

This section documents the work that preceded the R580 acceptance summary
below: the cu129 image build, the controlled A/B that decomposed an
apparent +37% gain into a max-num-seqs config effect, the driver-hygiene
lesson from the R535 host, and the NCCL-bump experiment's outcome.

### cu129 sibling image

Built `vllm-v100-py312-test:cu129` alongside the cu128 baseline (neither
replaces the other):

- Base: `nvidia/cuda:12.9.2-devel-ubuntu24.04`
- Toolkit: CUDA 12.9.2 (cuBLAS 12.9.1.4, cuDNN 9.10.2.21)
- torch: 2.10.0+cu129 (transitively pulls `nvidia-nccl-cu12==2.27.5`, same
  version as the cu128 wheel)
- vLLM: 0.18.0 unchanged
- Adds `cuda-compat-12-9` for NVIDIA CUDA Forward Compatibility (shipped
  libcuda 575.57.08 — the R575 UMD shim that let cu129 toolkit run against
  the older R535 KMD before the bare-metal upgrade).

Build-time validation gotcha: `torch.cuda.get_arch_list()` returns `[]`
during `docker build` (no `--gpus` flag, no NVIDIA Container Toolkit UMD
injection). The `sm_70` check moved to post-build runtime in
`tools/build_cu129.sh` where `--gpus all` is available. Documented for
future image work.

### Forward-compat probe on R535 (Scenario A confirmed)

`cuda-compat-12-9` shim functional on R535 at TP=4 and TP=8. All 8 V100s
visible, `matmul on cuda:0 = 1.0`, NCCL P2P/AR across the partial-NVLink
hypercube worked correctly under forward-compat at TP=8 (the highest-risk
hypothesis going in). Conclusion: cu129 toolkit usable on R535 without
the bare-metal driver upgrade — the upgrade became a strategic move, not
a perf-blocker.

### cu129/R535 smoke matrix (7 paths)

All deltas vs documented cu128 v0.4.0 envelope baselines (which were
measured at `--max-num-seqs=1`):

| Model | Path | cu128 ref | cu129/R535 | Apparent delta |
|---|---|---:|---:|---:|
| 27B FP16 | stock, m=3 FA+P | 39.60 | 42.04 | +6.2% |
| 27B GPTQ-Int4 | stock, m=3 FD | 47.61 | 65.35 | +37.3% |
| 35B-A3B FP16 | stock, m=3 FA+P | 15.76 | 15.88 | +0.8% |
| 27B-FP8 (dense) | patched, m=0 FD | 11.44 | 11.82 | +3.3% |
| 35B-A3B-FP8 | patched, m=0 FD | 52.87 | 53.12 | +0.5% |
| 122B-A10B-FP8 TP=8 | patched, m=0 FD | 34.76 | 34.78 | +0.1% |
| 122B-A10B-Int4 TP=8 | stock, m=3 FD | 63.62 | 64.88 | +2.0% |

**Caveat: cu129 tests used `--max-num-seqs=8`; cu128 references used `=1`.**
The deltas above are config-confounded. The 27B-Int4 outlier (+37%)
triggered the controlled A/B below.

### Controlled A/B — the decisive decomposition

Same cu128 baseline image, same args, only `--max-num-seqs` varied:

```
cu128 max-num-seqs=1: 47.61 tok/s  (documented v0.4.0 envelope)
cu128 max-num-seqs=8: 65.40 tok/s  (this A/B)
cu129 max-num-seqs=8: 65.35 tok/s  (smoke matrix)
```

Output SHA `1460dc04ae` token-for-token identical between cu128 max-ns=8
and cu129 max-ns=8. **The entire apparent +37% was the `max-num-seqs`
change. cu129 contribution: +0.1% noise.** This generalizes to every
smoke-matrix row — after decomposition, cu129 is perf-neutral within
±0.5% across all 7 paths. cu129 is a support-envelope upgrade, not a
performance upgrade.

### Why max-num-seqs=8 helps single-stream perf

At runtime batch=1, vLLM still captures cudagraphs for batch sizes
{1, 2, 4, 8} during warmup. Triton autotune sees more shapes during
capture and picks better kernel configs. The batch=1 captured graph uses
those better kernels at runtime even though only batch=1 is dispatched.

Effect is path-specific:

- **Big** on dense Int4 (exllama sm_70 GEMM had headroom for kernel-config
  tuning): +37% on 27B-Int4
- **Small** on dense FP16 (already near memory-BW limit): +6% on 27B-FP16
- **Tiny** on MoE+GDN paths (compute dominated by routed expert kernels,
  not the captured-batch Linears): +0.1% to +2% on 122B/35B variants

The full sweep on 27B-Int4 R580 (`{1, 8, 16}`):

| max-num-seqs | tok/s | Δ vs ns=1 | Δ vs ns=8 |
|---:|---:|---:|---:|
| 1 | 47.61 | — | -27.2% |
| **8** | **65.40** | **+37.4%** | **0% (peak)** |
| 16 | 64.76 | +36.0% | -0.97% |

Returns diminish past 8. Plausible causes for the small regression at 16:
captured-graph set bloat (more memory, slightly higher dispatch lookup
cost), KV-cache competition under the `gpu-memory-utilization=0.85`
budget, or autotune picking a less batch-1-specific config given a wider
shape set.

### Driver hygiene lesson — mixed-source R535 install

The pre-upgrade R535 host driver had **mixed install sources**: Ubuntu apt
packages and an older NVIDIA `.run` installer running side-by-side.
Discovered during R580 upgrade prep. Likely explains several
pre-upgrade glitches we couldn't pin down:

- NCCL-bump (`nvidia-nccl-cu12==2.30.4` via pip override on cu128) crashed
  on multiple serve startups despite ctypes-verified runtime override
- Intermittent serve startup behavior observed by GPT during initial
  FP16/Int4 probes
- Slight non-determinism patterns under cudagraph capture

Single-source R580 install from NVIDIA's repo is the new operational
standard.

**Operational note for future hosts:** when troubleshooting unexplained
CUDA-level glitches, check `dpkg -l | grep nvidia` against
`/var/log/nvidia-installer.log` for source consistency. Mixed-source can
silently leave libcuda paths inconsistent across the host even though
`nvidia-smi` reports a single version.

### NCCL bump experiment status (built, verified, parked)

Built `vllm-v100-py312-nccl-test:cu128` as a sibling that pip-overrides
`nvidia-nccl-cu12==2.30.4` (from torch wheel's transitive 2.27.5).
Runtime-verified two ways:

1. `ctypes.CDLL(libnccl.so.2).ncclGetVersion()` returns `23004` → 2.30.4
2. `NCCL_DEBUG=VERSION` banner from a torchrun 2-rank `init_process_group`
   + `all_reduce`: `NCCL version 2.30.4+cuda12.9`

Override genuinely loads at runtime. But multiple production serve
startups on cu128 crashed — possibly mixed-source R535 driver artifacts
(now resolved), possibly real ABI gap vs torch 2.10.0+cu128's
compile-time 2.27.5. Deprioritized in favor of cu129+R580, which is the
NVIDIA-blessed envelope and didn't need the NCCL bump to pass acceptance.

Image retained on disk for future quick A/B if a workload reveals NCCL
headroom on the cu129+R580 base.

**Methodology lesson worth keeping:**
`torch.cuda.nccl.version()` returns torch's **compile-time** NCCL constant
(baked into the wheel), not the runtime-loaded library version. For an
NCCL override, only `ctypes.CDLL(libnccl.so).ncclGetVersion()` or the
`NCCL_DEBUG=VERSION` banner on first `ncclCommInitRank` gives runtime
ground truth.

### Files in this stage

| File | Purpose |
|---|---|
| `docker/Dockerfile.vllm018_py312_cu129` | cu129 sibling image |
| `tools/build_cu129.sh` | Build + post-build forward-compat probe with `--gpus all` |
| `docker/Dockerfile.vllm018_py312_nccl` | NCCL-bump experiment sibling on cu128 (parked) |
| `tools/build_nccl_bump.sh` | NCCL-bump build + ctypes ncclGetVersion validation |
| `tools/nccl_probe.py` | torchrun-driven 2-rank NCCL init probe — prints `NCCL_DEBUG=VERSION` banner via a real `init_process_group` + all_reduce |
| `docker/run_docker_vllm018_py312.sh` | `IMAGE` env override (1-line patch) so the same launcher can drive cu128 or cu129 images |

### Operational state heading into R580 acceptance

| Layer | Value |
|---|---|
| Host driver | R535.288.01 (mixed-source, slated for replacement) |
| Container image | `vllm-v100-py312-test:cu129` |
| Container CUDA toolkit | 12.9.2 |
| Container torch | 2.10.0+cu129 |
| Container NCCL | 2.27.5 (torch wheel transitive) |
| Forward-compat shim | active (`cuda-compat-12-9`, libcuda 575.57.08) |
| Throughput default | `--max-num-seqs 8` |
| Streaming default | `--max-num-seqs 1` (latency + FP8-path determinism) |

---

## 2026-05-26 R580 acceptance and FlashAttention integration notes

### R580 driver acceptance

Final acceptance matrix across the important serving paths:

| Row | Path | Reference | R580 | Delta | Verdict | Determinism |
|---|---|---:|---:|---:|---|---|
| 1 | 122B-A10B-Int4 TP=8 | 64.88 | 64.50 | -0.58% | PASS | 3-of-5 |
| 2 | 122B-A10B-FP8 TP=8 production | 34.78 | 34.77 | -0.02% | PASS | 3-of-5 |
| 3 | 35B-A3B-FP8 TP=4 | 53.12 | 52.82 | -0.56% | PASS | 5-of-5 |
| 4 | 27B-GPTQ-Int4 TP=4 | 65.35 | 65.42 | +0.11% | PASS | 1-of-5 perfect |
| 5 | 27B FP16 TP=4 | 42.04 | 42.16 | +0.29% | PASS | 1-of-5 perfect |
| bonus | 35B-A3B FP16 TP=4 | 15.88 | 15.62 | -1.62% | borderline | 1-of-5 perfect |
| bonus | 27B-FP8 TP=4 dense | 11.82 | 11.86 | +0.33% | PASS | 2-of-5 |

Summary statistic across the five deployment-relevant rows: mean delta
`-0.15%`. This is noise-floor. **R580 holds production.**

The remaining methodology probe, `27B-Int4 max-num-seqs=16`, showed slightly
lower throughput than the current envelope. Conclusion:

- Keep `max-num-seqs=8` as the operational default.
- `max-num-seqs=16` is valid as an experiment, but under the current load
  pattern and config it shows diminishing or negative return.
- No R580 acceptance caveat is needed for batching defaults.

### FlashAttention-V100 mental model

Current project understanding:

- FP8 integration is mostly a linear-path interception problem. It is like
  finding the relevant resistors and replacing them with parts of the correct
  value/package. The board stays mostly the same.
- FlashAttention integration is an attention-backend / serving-runtime contract
  problem. It is more like replacing an IC or MCU on a board: even if the new
  chip exposes all the legs, the pinout, timing, voltage/protocol assumptions,
  and surrounding components must match.

Surrounding "peripherals" that may need adaptation:

- attention metadata builder
- KV cache allocator/layout
- paged block tables and slot mapping
- prefill/decode scheduler split
- prefix-cache logic
- chunked-prefill behavior
- CUDA graph capture path
- workspace allocation and lifetime
- backend selection/fallback logic
- model-specific attention wrappers
- KV dtype and scale handling, including FP8 KV
- tensor-parallel assumptions

Therefore "no error" is weak evidence for FlashAttention correctness. The
right integration loop is still incremental, but it must manufacture
correctness checks:

1. Make the backend importable and selectable.
2. Route one simple path to FA-V100 and assert that the FA path actually ran.
3. Compare tensors/logits against `TRITON_ATTN` on tiny controlled cases.
4. Expand shape coverage: dense contiguous, varlen/prefill, paged decode,
   paged prefill, prefix-cache cases, FP8 KV, CUDA graphs.
5. Only after correctness, run real model throughput A/B.

Practical conclusion: keep FlashAttention-V100 as a separate backend project,
not as the next patch inside the FP8 package. For the current production
MoE+GDN targets, `TRITON_ATTN` remains the serving default; previous cross-stack
data showed FA2-v100 can help pure dense-attention models but can hurt hybrid
GDN models badly.

### Dense 27B-FP8 diagnosis and custom-op implication

Follow-up diagnosis on Qwen3.6-27B dense+GDN clarified the slow FP8 result.

Eager Stage 2D breakdown with dense-MLP hooks showed both FP8 and FP16 are
slow under `--enforce-eager` and dominated by row-parallel all-reduce:

| Path | Mode | Wall result | Notes |
|---|---|---:|---|
| 27B-FP8 TP=4 | eager + breakdown | `200 tok / 53.5s` = `3.74 tok/s` | Dense MLP ~106 ms/token, GDN ~94 ms/token, row_parallel_ar ~150 ms/token. |
| 27B FP16 TP=4 | eager + breakdown | `200 tok / 60.3s` = `3.32 tok/s` | Same broad shape; row_parallel_ar often dominates. |
| 27B FP16 TP=4 | `mode=0`, `FULL_DECODE_ONLY` | `200 tok / 4.874s` = `41.0 tok/s` | Clean run without breakdown envs; CUDA graph capture completed and serving was stable. |

This proves that `mode=0` is not itself the bottleneck for FP16. CUDA graph
replay supplies nearly all of the FP16 production speedup; Inductor/FULL_AND_PIECEWISE
adds at most a small increment on this path.

The dense-FP8 gap remains:

- 27B FP16 `mode=0 + FULL_DECODE_ONLY`: about `41 tok/s`.
- 27B-FP8 `mode=0 + FULL_DECODE_ONLY`: about `11-12 tok/s`.

Interpretation:

- The old "dense FP8 is slow because eager Python dispatch dominates" theory is
  not supported; eager FP8 is not worse than eager FP16.
- The remaining FP8 production gap is likely a mixture of:
  - real dequant+GEMM kernel cost,
  - pybind11 extension ops acting as opaque graph nodes,
  - lost fusion / scheduling / communication overlap around row-parallel AR.
- The current data does **not** prove custom-op registration would close the
  whole 3.6x gap. It does make custom-op registration the next credible lever
  if we want to chase graph-level optimization.

Deployment rule is unchanged:

- Dense+GDN 27B-class models should use GPTQ-Int4 or FP16, not this FP8 path.
- Do not invest in dense 27B-FP8 as a product target while GPTQ-Int4 and FP16
  already win.

Strategic implication for the real MoE+GDN targets:

- 122B-A10B-FP8 and 35B-A3B-FP8 already run well under `mode=0 + FULL_DECODE_ONLY`.
- If `torch.library.custom_op` registration with fake/meta impls lets the FP8
  extension participate in `mode=3 + FULL_AND_PIECEWISE`, the MoE targets may
  gain substantially.
- Treat this as a Stage 4 candidate and measure first on 122B-A10B-FP8, not on
  dense 27B-FP8.

Profiler cleanup note:

- CUDA-event breakdown hooks are eager-only by construction.
- Running `VLLM_V100_FP8_DECODE_BREAKDOWN=1` under cudagraph crashed at
  `torch.cuda.Event.elapsed_time()` during graph replay. The local tree now has
  a fail-closed guard so profiling disables itself instead of killing the
  engine, but useful attribution still requires `--enforce-eager`.


## Stage 4 — MTP speculative decoding (v0.4.1 opt-in, 2026-05-28)

### Investigation arc

Started from a different question: should we patch and pivot to the 1catai-vllm fork to gain Multi-Token Prediction (MTP) speculative decoding on Qwen3.5/3.6 MoE+GDN serving?

Three discoveries during the session collapsed that question entirely:

1. **MTP is upstream in vLLM 0.18.0.** The installed wheel includes
   `vllm/model_executor/models/qwen3_5_mtp.py` defining `Qwen3_5MultiTokenPredictor`, registers
   `Qwen3_5MTP` and `Qwen3_5MoeMTP` in [`registry.py:573-574`](../../../vllm/vllm/model_executor/models/registry.py), and
   `config/speculative.py:311` auto-rewrites `qwen3_5_moe` model_type to
   `qwen3_5_mtp`. No 1catai-specific code is required.

2. **Production checkpoints ship MTP weights baked in.** Both
   `/mnt/models/Qwen3.5-122B-A10B-FP8` and `/mnt/models/Qwen3.6-35B-A3B-FP8` have
   `text_config.mtp_num_hidden_layers=1` and ~1,560 `mtp.*` weight keys in their
   safetensors indices, FP8-quantized, MoE-structured matching the main body.

3. **The MTP head uses the same `QwenNextMixtureOfExperts` class as the main
   body.** Our existing `Fp8MoEMethod` monkey-patches intercept it transparently;
   no patch extension is needed.

Result: enabling MTP is a one-flag change to the launcher
(`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`), nothing
else.

### Initial throughput measurement (steady-state decode bench)

| Model | TP | v0.4.0 baseline | v0.4.0 + MTP | Speedup |
|---|---:|---:|---:|---:|
| Qwen3.6-27B-Dense FP16 | 4 | 39.60 | 43.18 | 1.09× |
| Qwen3.6-35B-A3B-FP8 | 4 | 52.87 | 65.14 | 1.23× |
| **Qwen3.5-122B-A10B-FP8** | **8** | **34.76** | **47.32** | **1.36×** |

Cross-hardware anchor: the 122B + MTP number is within ~80% of sustained
2×A100 performance on the same model. **These are the citation-grade
performance numbers**; they come from the dedicated steady-state bench, not
the exactness harness (see methodology note below).

### 1catai chapter closed (with prejudice)

Side investigation during the day, independently of MTP discovery:

- **FA2-V100 on hybrid attn+GDN**: 1catai-vllm v1.0.0 + FA2-V100 + cudagraph on
  Qwen3.6-27B Dense FP16 (hybrid GDN-dense) runs at 35.01 tok/s; our v0.4.0
  + TRITON_ATTN + cudagraph + MTP runs at 43.18 tok/s on the same model and
  config. FA2-V100 is a **net loss** on GDN-hybrid models. Earlier projection
  (memory: "2.65× slower on hybrid") reproduced.
- **1catai loader bug**: `Qwen3_5MoeForConditionalGeneration` in 1catai's tree
  has a `language_model.` prefix + per-expert key mapping mismatch with current
  Qwen3.5/3.6 MoE checkpoints. Dense models load; MoE models fail. Affects the
  entire production-target family. ~2-4 hours to patch.
- **1catai FP8 gate**: their `Fp8Config.get_min_capability()` is env-var-gated
  (`VLLM_SM70_FP8_DEQUANT_FALLBACK=1`, `VLLM_SM70_FP8_TURBOMIND=1`) — not hard-
  blocked as previously believed. Engineered but unreachable from default config.
- **1catai TRITON+MTP correctness regression**: when forced to TRITON_ATTN +
  MTP for an apples-to-apples comparison against our stack, 1catai produced
  `" the the the the ..."` for 200 tokens with `mean_acceptance_length=1.97`
  and `draft_accept_rate=97.5%`. **High acceptance rate alone is not evidence
  of correctness** — same diagnostic signature pattern as a verifier-alignment
  failure where the verify path is comparing the wrong positions/logits/KV
  state. Discovery itself is a methodology lesson worth keeping.

Combined: 1catai is uninteresting for our production family at every axis we
care about. **Not revisiting for production.** Their main unique value-add
(MTP) is already in stock vLLM 0.18.

### Exactness validation — methodology evolution

GPT peer-review checkpoint flagged "exactness check FIRST given today's
broken-output finding". We built an 11-prompt suite covering short factual,
code (Python + C), reasoning, repeated text, stop+EOS, ignore-EOS sustained
(500 tokens), long prose (1k token prompt), long code (1k token prompt). Each
prompt run at temperature=0 against both a baseline serve (no MTP) and an
MTP-enabled serve; token-string lists compared via Python list equality.

Mid-investigation, after the 35B-A3B FP16 test failed (5/11 identical), we
added a **baseline-vs-baseline self-test** — same serve called twice, compared
against itself. That distinguishes "MTP-introduced divergence" from "FP8 path
intrinsic nondeterminism". Without this control the FP8 result would have
been ambiguous.

Sequential testing for 122B: TP=8 fills all 8 GPUs, so baseline and MTP
serves cannot run concurrently. Extended the script with `--record-only` +
`--baseline-from-file` flags: phase 1 records baseline tokens to JSON while
the baseline serve is up; phase 2 reads from JSON and only hits the MTP serve.

### Validation matrix (final)

| Model | Baseline self | MTP vs baseline | MTP accept | Read |
|---|:---:|:---:|:---:|---|
| 27B Dense FP16 | (assumed ✓) | **11/11 ✓** | 91.9% | MTP itself is mathematically correct |
| 35B-A3B FP16 | **11/11 ✓** | 5/11 ✗ | 98.1% | MoE batch-shape FP-order makes MTP-vs-baseline non-bit-exact |
| 35B-A3B-FP8 | 8/11 | 7/11 | 96.1% | MTP adds 1 extra divergent prompt vs baseline's own noise; same-positions overlap |
| **122B-A10B-FP8 (prod target)** | **6/11** | **7/11** | **91.4%** | **MTP introduces NO additional divergence beyond baseline self-noise** |

Notable per-prompt detail on 122B: of the 4 prompts that diverge in phase 2,
two diverge at the *exact same position* as the phase 1 baseline-self-test
(short_factual_2 @ pos 26, reasoning_1 @ pos 13 — both with the same length-
mismatch pattern). MTP is exposing the same intrinsically-borderline positions
the FP8 baseline has, not new ones.

### Quality-equivalence verification (hand-inspected)

Pulled actual divergent text from result JSONs:

- 35B-A3B FP16 "capital of France": baseline says Paris is "famous for its
  vibrant culture, art, fashion..."; MTP says Paris is "famous for its fashion,
  cuisine, and art scene...". Both are factually correct, well-formed
  same-distribution prose continuations.
- 35B-A3B FP16 Fibonacci-memo: the entire Python function body is bit-identical
  for 87 tokens; divergence is purely the explanatory prose after the code
  ("It starts with a base case..." vs "The base cases are n =..."). For code
  generation, the actual code is preserved exactly.
- 35B-A3B-FP8 code_c reverse-string: after completing the requested function,
  one stack continues into a palindrome helper; the other into a vowel-reversal
  helper. Different completion, not an error.

GPT independently reviewed the raw result JSONs and concurred:
"I do not see broken-model behavior like loops, malformed text, or obvious
factual collapse." Quality-equivalence is established for the inspected
divergences.

### Performance methodology caveat (important)

The exactness harness aggregates wall time across single-sample timings
mixed with warmup-cold-cache effects. On 122B specifically, two prompts
(code_python, reasoning_1) hit cold-shape paths the script's 16-token
warmup didn't cover, dropping individual baselines to 1.83 and 0.89 tok/s.
These outliers dominate the aggregate, producing a misleading 1.02× headline.

**Use the exactness harness for output validation only; performance claims
should cite the dedicated steady-state bench numbers** above (1.36× on 122B,
1.23× on 35B-A3B-FP8, 1.09× on 27B Dense). Adding all-runs storage + a real
warmup sweep to the harness is on the v0.4.2 list.

### GPT peer-review checkpoints

Three rounds during the session, all converging:

1. **Round 1** (post 1catai brief): flagged that MTP is upstream in vLLM
   (`qwen3_next_mtp` reference in docs.vllm.ai), reorienting the investigation
   from 1catai-pivot to upstream-MTP-enablement. Decisive course correction.
2. **Round 2** (post 27B+35B exactness): recommended opt-in shipping with
   workload-shape caveats; tempered "acceptance proves correctness" framing
   ("after 1catai, acceptance alone should never be treated as proof"); listed
   six items needed before default-on (multi-sample, all-runs storage,
   production prompt mix, streaming, routing, 122B validation).
3. **Round 3** (independent read of raw result JSONs + 122B sequential
   results): confirmed methodology, confirmed quality-equivalence on hand-
   inspected divergent outputs, confirmed 122B exactness gate is closed for
   opt-in. Performance-from-exactness-harness flagged as unreliable; redirected
   to dedicated bench numbers. Softened the long-prompt warning since 122B
   did not show the regression that 35B-A3B-FP8 showed.

### Ship decision

**v0.4.1: `ENABLE_QWEN_MTP=1` opt-in, default OFF.**

What we claim:
- Adds optional Qwen3.5 MTP speculative decoding support.
- Improves decode-heavy workloads in measured tests.
- May regress long-prompt-short-output workloads (observed on 35B-A3B-FP8;
  not observed on 122B-A10B-FP8 in this probe).
- MoE/FP8 outputs are not guaranteed bit-identical to baseline.
- Observed divergences were quality-equivalent in the validation suite.

What we don't claim:
- Bit-exactness on MoE or FP8.
- Universal speedup.
- Long-prompt-short-output safety as a general guarantee.
- Default-on validation.
- Acceptance rate alone as proof of correctness.

### v0.4.2 backlog (default-on promotion gate)

Per GPT round-3 checklist:

1. Multi-sample self-tests at higher N to estimate baseline noise rate
   statistically (current data is single-sample).
2. More prompts, especially production chat/code/tool-use traffic.
3. Streaming and `/v1/chat/completions` endpoint sanity checks.
4. Per-workload routing decision: per-request `--speculative-config` (if
   vLLM supports), or operationally a two-service split (port A baseline,
   port B MTP).
5. Long-prompt-short-gen confirmation at higher N on the production target
   (single-sample 122B data was favorable but not proven).
6. Exactness script improvements:
   - Store integer token IDs (currently stores token strings only).
   - Store ALL runs when `measure-runs > 1` (currently keeps only last
     run's tokens for exactness; averages timings).
   - Record provenance metadata in output JSON (CLI args, model path,
     ports, monkey-patch env vars).
   - Fix `--baseline-from-file` warmup path: currently hits both
     endpoints during warmup, will fail when one is down for sequential
     tests.

### Files touched in v0.4.1

- `docker/run_docker_vllm018_py312.sh` — added `ENABLE_QWEN_MTP=1` env var
  support; appends `--speculative-config` to both `serve` and `serve-fp8`
  modes when set.
- `README.md` — added "Optional: MTP Speculative Decoding (v0.4.1)" section
  with workload guidance, exactness matrix, and DO/DON'T claims.
- `REQUIREMENTS.md` — added Qwen3.5 MTP as a Layer-3 deployment choice.
- `docs/SESSION_LOG.md` — this entry.

### Provenance and reproducibility

Raw exactness results (live during dev, not committed):
- `/tmp/v100_bench/exactness_27b_results.json`
- `/tmp/v100_bench/exactness_35b_a3b_fp16_results.json`
- `/tmp/v100_bench/exactness_35b_a3b_fp16_self_test.json`
- `/tmp/v100_bench/exactness_35b_a3b_fp8_results.json`
- `/tmp/v100_bench/exactness_35b_a3b_fp8_self_test.json`
- `/tmp/v100_bench/exactness_122b_baseline_phase1.json` (122B baseline self-test)
- `/tmp/v100_bench/exactness_122b_mtp_phase2.json` (122B MTP vs phase 1 baseline)

Test infrastructure (also under `/tmp`, not committed; archived to
`tools/exactness_harness/` in v0.4.2 with the script improvements above):
- `/tmp/exactness_prompts.json` — 11-prompt suite
- `/tmp/run_exactness.py` — comparison script with `--record-only` and
  `--baseline-from-file` flags added during the 122B sequential test.

---

## Session handoff (2026-06-10) — Qwen3.6-27B FP8 coalesced decode, M<=8 probe

Dense FP8 decode slowness was traced to A.3's strided per-output-row weight
loads. The separate coalesced M=1 kernel fixed the single-stream case:

| Qwen3.6-27B-FP8 TP=4 decode path | tok/s |
|---|---:|
| A.3 baseline | 11.75 |
| coalesced M=1, unroll=2 | 35.08 |
| coalesced M=1, unroll=4 | 37.24 |

M=1 is effectively proven: Qwen and Gemma both transfer, with Gemma CT-channel
landing at 28.24 tok/s (97.5% of FP16) on Claude's repeat.

### M<=8 kernel prototype

Implemented a second, gated M<=8 proof kernel rather than rewriting A.3. Shape:
one CTA per N tile, 8 warps/CTA, each warp owns one output column, reads W once
coalesced along K, stages up to 8 activation rows in shared memory, and accumulates
all M rows before writing `[M,N]`. This removes the old proof kernel's repeated
weight read per row.

Controls:
- `VLLM_V100_FP8_COALESCED_GEMV=1`
- `VLLM_V100_FP8_COALESCED_UNROLL=4` for the M=1 kernel
- `VLLM_V100_FP8_COALESCED_M_UNROLL=8` for the M<=8 kernel
- `VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8` to enable the M<=8 path

Correctness: `tools/qwen27b_fp8_coalesced_numtest.py` passes M=1,2,8 for both
block scale (`block_h=128`) and channel scale (`block_h=1`), with cos≈1.0.

Kernel microbench on Qwen attention shape (`N=K=5120`):

| path | M=2 | M=4 | M=8 |
|---|---:|---:|---:|
| old M<=8 proof kernel | n/a | n/a | ~0.602 ms |
| new tiled M<=8, unroll=2 | ~0.217 ms | ~0.220 ms | ~0.223 ms |
| new tiled M<=8, unroll=4 | ~0.218 ms | ~0.220 ms | ~0.224 ms |
| new tiled M<=8, unroll=8 | ~0.214 ms | ~0.215 ms | ~0.217 ms |

This is a real kernel-side win: M=8 is about 2.8x faster than the first proof and
about 4.6x faster than A.3 k=4 (~1.006 ms). It is still about 2.2x slower than
cuBLAS FP16 (~0.096-0.110 ms), and M-unroll barely moves the result, so the next
M<=8 kernel limit is likely accumulator/reduction/scheduling or M-specialization,
not simple HBM load coalescing.

### First concurrent e2e sniff

Two TP=4 Qwen servers were run concurrently:
- GPUs 0-3, port 8026: coalesced M=1 only (`M_MAX=1`)
- GPUs 4-7, port 8027: M<=8 enabled (`M_MAX=8`, `M_UNROLL=8`)

The M=1-only 4-concurrent client effectively stalled and fell back heavily into
A.3 variants. The M<=8 server genuinely used the new path during graph capture
and decode (example counter mix: `Coalesced GEMV-M=2816`, about 70% of calls) and
completed 4 simultaneous 256-token requests:

`m8_enabled port=8027 concurrent=4 total_tokens=1024 wall=59.313s aggregate_tok_s=17.264`

Conclusion: M<=8 is correct and the kernel-side proof is promising, but the first
multi-user e2e number is not yet a production win. It removes the pathological
M=1-only stall, yet aggregate throughput is far below the 37.24 tok/s single-stream
M=1 result. The next experiment should profile concurrent decode with breakdown
enabled (attention/GDN/AR/scheduler/cudagraph-shape effects), and also test
concurrency 2 and 4 with shorter generations to separate capture/warmup bubbles
from steady-state throughput.

Claude's Gemma-4 dense repeat changes the priority: pure dense Gemma on the same
coalesced M<=8 kernel scales correctly under concurrency, while Qwen hybrid does
not. Gemma dense FP8-resident numbers:

| config | C=1 aggregate | C=2 aggregate | C=4 aggregate | C=4 per-stream |
|---|---:|---:|---:|---:|
| M<=8 coalesced | 24.94 | 16.16* | 45.28 | 12.37 |
| M=1/A.3 fallback | 25.00 | 12.19 | 19.73 | 5.12 |

`*` C=2 was marked an outlier due to TTFT spike. The useful signal is C=4:
M<=8 coalesced beats fallback 2.3x aggregate and scales upward from single-stream.

Revised conclusion: the scary Qwen C=4 cliff is not a generic coalesced M<=8
kernel failure and not a generic cudagraph failure. It is Qwen hybrid-specific,
most likely in the GDN/MoE decode path, state update, expert dispatch, or
scheduler interaction under concurrency. M-specific GEMV specialization is now
polish, not the next highest-value step.

Recommended next Qwen branch:
1. Run Qwen hybrid concurrency probes at C=2 and C=4 with decode breakdown enabled:
   `VLLM_V100_FP8_DECODE_BREAKDOWN=1`,
   `VLLM_V100_FP8_DECODE_BREAKDOWN_GDN_SUBS=1`,
   `VLLM_V100_FP8_DECODE_BREAKDOWN_MOE_SUBS=1`,
   `VLLM_V100_FP8_MOE_OTHER_PROFILE=1`, and AR profiling if row-parallel all-reduce
   is in the path.
2. Keep `VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8` enabled so GEMV is not the known
   bottleneck while profiling the hybrid path.
3. Compare Qwen dense-ish sections against Gemma dense: if GEMV counters look good
   but per-token wall is bad, focus on GDN/MoE Python/vLLM bookkeeping and captured
   graph shape selection rather than CUDA GEMV.

Claude's Gemma-26B-A4B MoE concurrency discriminator closed the GDN-vs-MoE fork:

| Gemma-26B-A4B MoE config | C=1 aggregate | C=2 aggregate | C=4 aggregate | C=4 per-stream |
|---|---:|---:|---:|---:|
| M<=8 coalesced attention + resident MoE | 38.14 | 26.59* | 110.95 | 30.38 |
| M=1/A.3 attention fallback + resident MoE | 37.86 | 37.61 | 66.59 | 18.04 |

`*` C=2 had the same TTFT/capture bubble and should not drive the trend. The C=4
signal is decisive: MoE routing/grouped/scatter scales very well on the same
stack, while Qwen does not. GDN is now the only remaining primary suspect.

Immediate Qwen-specific hypothesis from source read:
- `Qwen3NextGatedDeltaNet` has a default-on packed recurrent decode fast path,
  gated by `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1`.
- The packed path calls `fused_recurrent_gated_delta_rule_packed_decode` after
  `causal_conv1d_update`.
- Disabling the flag routes decode through the generic
  `fused_sigmoid_gating_delta_rule_update` path.

New harness added:
- `tools/qwen_gdn_concurrency_ab_vllm021.sh`

It keeps coalesced M<=8 enabled and runs:
- `packed_cg`: packed recurrent decode ON, cudagraph
- `unpacked_cg`: packed recurrent decode OFF, cudagraph
- `packed_eager_profile`: packed ON, eager + breakdown
- `unpacked_eager_profile`: packed OFF, eager + breakdown

Readout:
- If `unpacked_cg` fixes C=4 aggregate/per-stream, the cliff is inside the packed
  recurrent GDN fast path.
- If both cudagraph configs collapse, use the eager profile rows to split
  `gdn_core`, `gdn_out_proj`/AR, and `gdn_other` under concurrency.
- If eager behaves but cudagraph collapses, investigate GDN metadata/fixed-buffer
  capture behavior in `vllm/v1/attention/backends/gdn_attn.py`.

Result: the Qwen "GDN cliff" did NOT reproduce under a streaming steady-state
harness. Packed recurrent decode is not the culprit.

Parallel CUDAGRAPH A/B, Qwen3.6-27B-FP8 TP=4, coalesced M<=8 enabled,
`MAXTOK=128`, streaming per-token measurement:

| Qwen GDN config | C=1 aggregate | C=1 per-stream | C=2 aggregate | C=2 per-stream | C=4 aggregate | C=4 per-stream |
|---|---:|---:|---:|---:|---:|---:|
| packed recurrent decode ON | 28.96 | 36.56 | 19.97 | 13.57 | 49.82 | 15.29 |
| packed recurrent decode OFF | 26.81 | 36.64 | 20.32 | 13.57 | 48.28 | 14.98 |

C=2 has the same TTFT/capture bubble pattern seen in Gemma. The C=4 signal is
the important one: aggregate scales up, and packed vs unpacked is identical
within noise.

Confirmation at the original longer decode length, `MAXTOK=256`, packed ON:

`packed_cg: C=4 aggregate=56.04 tok/s per_stream_decode=15.65 tok/s ttft=1.91s ok=4/4`

Revised conclusion: the earlier Qwen C=4 `17.26 tok/s` result was a harness /
request-shape / TTFT artifact, not a real GDN or MoE concurrency collapse. The
coalesced M<=8 path is healthy on Qwen too:
- C=1: single-stream per-stream decode remains ~36.6 tok/s.
- C=4: aggregate reaches ~50-56 tok/s.
- `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` does not materially affect decode
  throughput in this test.

Do not pursue GDN kernel surgery from the old cliff number. The remaining real
issue is ordinary sub-linear batching efficiency: C=4 aggregate is ~1.35-1.5x
single-stream rather than 4x, similar in kind to Gemma though lower. Future work
should package the dense/MoE/Qwen coalesced win and only revisit GDN if a
production traffic harness reproduces a steady-state collapse with streaming
inter-token metrics.

---

## Session note (2026-06-10) — next push benchmark plan + `--skip-mm-profiling` side note

Remember this for the next README/version/publish push: add an operational side
note for V100 vLLM users serving text from VL-capable checkpoints. For text-only
serving/benchmarks, use `--skip-mm-profiling` to skip the max-size vision-encoder
dummy profile during engine init. This is a vLLM engine option, not an FP8 kernel
optimization. In the stack matrix work it reduced cold startup from roughly
850s to roughly 19s because vision profiling dominated; the remaining FP8 LM
profile-prefill was small. Keep the caveat: this shifts multimodal encoder peak
memory responsibility to the operator and is not appropriate when validating
real image/video/audio capacity.

Benchmark default going forward:
- For text decode benchmarks of VL-capable models, default `SKIP_MM=1` /
  `--skip-mm-profiling`, and explicitly record that in result headers.
- No eager headline rows unless debugging. Publish cudagraph and cudagraph+MTP
  where applicable; tok/s must come from usable/coherent output, with exactness
  vs correctness called out for MTP.
- Compare vLLM `0.19+cu128` against `0.21+cu126`; CUDA wheel effects are
  secondary after the matrix showed cu126 ~= cu128, while 0.21 has a small
  engine-side decode delta to quantify cleanly.

Rows/models to organize into multi-aspect tables:
- Official FP16/BF16 model when it fits on the V100 box vs official FP8 model.
- For 100B+ models, find and include GPTQ-Int4 where available as the practical
  speed/quality/memory comparator.
- Model list: Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B,
  gemma-4-31B-it, gemma-4-26B-A4B-it, and GLM-4.5-Air.
- Include startup time for all FP8 models with and without vision profiling
  skipped, so users see both conservative multimodal init and text-serving init.

Existing artifact to use:
- `tools/stack_cu_matrix_ab.sh` — standardized stack/model matrix with CPU-clean
  gate, isolated cache tags, `SKIP_MM` toggle, and env-controlled reruns.

---

## 2026-06-12 (PM) — Volta FP16 MoE pathology RESOLVED: BLOCK_K, not num_stages (Fable session)

Resumed the FP16-MoE-slower-than-dense investigation from the Opus 4.8/Codex 5.5 wrap-up
(memory `project_volta_moe_fp16_patch`). Re-verified their code trace 100% (config-miss, 0/317
V100 JSONs, Ampere-blind default, Volta-aware combine contrast) — then MEASURED their bet and
refuted it, found the real lever, and validated the fix e2e on both models.

1. **num_stages e2e sweep = NULL** (`tools/moe_stages_ab_vllm021.sh`, arms base/s4/s3/s2,
   Ch1-cell-identical serve): Qwen 35B-A3B FP16 = 15.57 tok/s on ALL arms (bit-identical sha);
   gemma 26B-A4B = 10.90 flat. s4 control proved the VLLM_TUNED_CONFIG_FOLDER mechanism is
   perf-neutral. → results/moe_stages_ab_q35b_20260612_132040, ..._g26b_20260612_140610.
2. **Microbench found the sink** (`tools/moe_decode_microbench.py`, GPU4, decode shapes M=1,
   E=256, topk=8, K=2048, Nshard=128): `fused_moe_kernel` = 98.9% of fused_experts, 645 us/launch,
   **90x off memory floor** (~0.4 TFLOP/s — Triton sm_70 FMA fallback + 16x-padded M tiles).
   1.3 ms x 40 layers ≈ 56 ms/token = exactly the e2e gap vs our FP8 (70 tok/s = 14 ms).
3. **Tile sweep found the lever** (`tools/moe_decode_tile_sweep.py`): kernel time scales with
   BLOCK_SIZE_K (64→632 us, 128→1450, 256→2300) — register-spill signature of Volta codegen.
   N/warps/stages ~irrelevant. Stock default picks K=128 exactly and only in the decode branch
   (M<=64); prefill already gets K=64. gemma shapes: 4.34x kernel win.
4. **kbest e2e VALIDATED** (config JSON, small-M entries = 16/32/64 w4 s2):
   - Qwen 35B-A3B FP16: 15.57 → **65.87 tok/s = 4.23x**, output sha BIT-IDENTICAL to base.
     Beats dense 27B FP16 (37-41); 94% of our FP8-coalesced (70.06).
   - gemma 26B-A4B FP16: 10.90 → **43.62 tok/s = 4.00x**, shas identical to base arms.
     Beats dense 31B FP16 (17.6) 2.5x; BEATS our CT-FP8 gemma (~38).
   Inversion fixed on both — sparse>dense restored with ONE json file, zero source patches.

Implications: (a) honest-numbers update needed where we quote "FP8 4.5x over stock FP16 MoE"
(comparator was untuned; tuned FP16 closes to ~1.06x on 35B, surpasses CT-FP8 on gemma — FP8's
durable value = half memory + 122B-class models where FP16 can't fit); (b) clean upstream PR:
V100 config JSONs per (E,N) and/or sm<80-aware `get_default_config` (BLOCK_K=64 small-M,
mirroring moe_fused_mul_sum.py's existing Volta case). Structural ceiling stands: even tuned,
the Triton kernel is ~40x off floor on sm_70.

Tools added: moe_stages_ab_vllm021.sh (MODEL_KEY=q35b|g26b, ARMS incl. kbest),
moe_decode_microbench.py, moe_decode_tile_sweep.py (SWEEP_* env for other shapes).

---

## 2026-06-13 — Volta MoE config: own feasibility-pruned shell-walk tuner + e2e adjudication

Stock benchmark_moe.py --tune abandoned (1920-config brute force = 100-240 s/it on Volta's
SMEM-infeasible big tiles; 0/18 batches after 10h). Replaced with own tuner embodying the
user's scanning insight:
- tools/moe_volta_tune.py: a-priori SMEM prefilter stages*(BM*BK+BK*BN)*2<=96KB (closed-form,
  no compile -> 288->204 feasible, 84 monsters skipped) + shell-walk from low corner with
  monotone early-stop.
- tools/moe_volta_tune_fleet.sh: shard 18 (shape,M) jobs across 8 V100s, ~50 min, merge per-M.
- Shell early-stop LOSSLESS on all 18 jobs (12% configs @ small-M, up to 99% @ large-M).
- Merged canonical JSONs: results/moe_volta_tune_{q35b,g26b}/E=,N=,device_name=Tesla_V100.json.

E2e adjudication (base vs kbest hand-ladder vs auto fleet-JSON, both models, single + 8-user):
  q35b single:  base 15.56 | kbest 65.91 | auto 65.85   (tie)
  q35b 8-user:  base 24.9  | kbest 163.7 | auto 180.8 agg ; per-user 3.16/22.4/22.9
  g26b single:  base 10.91 | kbest 43.66 | auto 43.71   (tie)
  g26b 8-user:  base 28.0  | kbest 152.7 | auto 161.4 agg ; per-user 3.56/19.2/20.3
  -> auto wins: tie single-stream, ~5% faster at concurrency (finer per-M granularity).
     outputs bit-identical (pure speed). M=16 msweep "16/128/64" was unrepresentative.

DECISION: ship autotuned JSONs (headline shapes) + keep plugin heuristic patch as universal
fallback; they layer automatically (get_moe_configs JSON before get_default_config). Upstream
target = aphrodite (vLLM dropped sm_70 by policy). Box rebooted 05:00 (kernel 179->181)
mid-first-e2e; reran clean, nothing lost.

## 2026-06-13 (cont) — FP8 MoE decode profile settles "where's the headroom" (Codex peer-review)

Codex proposed FP8 tuning axes + hypothesis "biggest upside is dispatch + route/scatter/data-
movement, not GEMM tile." Tested with built-in per-section profiler (tools/moe_fp8_profile_decode.sh,
eager — cudagraph capture conflicts with the profiler's CUDA-event syncs, same class as k=2 MTP crash).
Decode M=1 GPU-time per MoE call (the part that survives into cudagraph production):
  GEMV/compute w13+act+w2 = 0.113ms (73%) | route/scatter glue = 0.042ms (27%) | total GPU 0.155ms.
  eager avg_wall 1.59ms; the ~1.14ms py_inner_loop+py_dispatch is cudagraph-replayed-away in prod.
VERDICT: hypothesis NOT supported at decode (GEMV-dominated; dispatch overhead = eager-only artifact).
KEY FINDING: w2_gemm 0.069ms = 44% of MoE GPU, 2.3x slower than w13 (0.030) despite HALF the weight
bytes → w2 (short-K=128/wide-N=2048) ~4.6x less efficient/byte than w13 (long-K/narrow-N). Prime
suspect = K-split atomic contention on w2's wide output (ties to the K_SPLIT non-monotone caution).
→ THE FP8 decode target is the w2 GEMV, not the route glue. Secondary: fuse route_weight_apply+scatter
into w2 epilogue ~ -14% MoE GPU. Prefill differs (glue≈GEMV + 28ms unattributed non-grouped) but less
critical. NOTE: FP8 decode is already 70 tok/s (>30 "comfortable") so w2 work is optional polish.
Commit ce80954.

## 2026-06-13 (cont) — 8-user FP8 MoE profile = the decision gate (w2 anomaly is M=1-only)

Ran the profiler at NUSERS=8 (decode bucket M=8, route_slots=64) to gate whether FP8 w2 work
is justified (Codex's missing decision point). Decode M=8 GPU/call:
  w13_gemm=0.110 w2_gemm=0.105 activation=0.014 | route/scatter glue=0.040 | total 0.269ms.
  GEMV/compute = 85%, glue = 15% (was 27% at M=1).
VERDICT: neither of Codex's branches fires. Glue did NOT grow (shrank 27%->15%); w2 does NOT
uniquely dominate (w13 ~= w2 at M=8). Per-slot: w2 0.0086(M1)->0.0016(M8) = 5.2x more efficient;
w13 2.2x. => w2's M=1 slowness is FIXED OVERHEAD (under-occupancy: wide-N=2048/short-K=128 can't
fill the GPU at 8 slots) that AMORTIZES AWAY by M=8. Not a broken kernel.
DECISION: do NOT greenlight w2 kernel work. It helps only 1-2 user case (~+11% on already-
comfortable 70 tok/s); at 8 users GEMVs are balanced+efficient, aggregate 164-180 tok/s. Epilogue
fusion dead at decode (15% & shrinking). Bank FP16 MoE as the headline; FP8 w2 = optional polish,
no trigger. ONLY revisit if single-user latency on 122B-A10B TP8 (~45 tok/s, less headroom) becomes
a target -> re-profile 122B M=1 first. Artifact: results/moe_fp8_profile_20260613_072405/.

## 2026-06-13 (cont) — Vision encoder FA bridge: measured NO-GO (Codex microbench, Claude review)
V100 ViT defaults to Torch SDPA (sm70 backend order short-circuits before TRITON). ai-bond FA varlen
API fits ViT but kernel only compiles D in {16,32,64,128,256}; vision head_dim=72 hard-errors. Cheap
bridge (pad 72->128) microbenched (tools/vit_fa_v100_d72_microbench.py): correct cos=1.0 but ~0.4x
SDPA at all ViT shapes (256-2048, single+batched) = 2-2.7x slower, flat, no crossover. Extrapolation
(strip 1.78x pad): native D=72 ~0.71x, D=80 ~0.64x — deeper kernel also loses. Root cause: FA's
O(N)-memory advantage is moot at ViT's short N (<=4k) where SDPA mem-efficient already suffices.
DECISION: V100 ViT stays on SDPA; do not build the FA-ViT bridge or a custom head-dim kernel. Real
vision levers = --skip-mm-profiling + --limit-mm-per-prompt, and only if images are served.

## 2026-06-13 (cont) — SDPA-internal-path check closes the vision thread (no free backend win)
Probed which SDPA backend V100 uses at ViT shape (D=72, fp16), forcing each (tools: /tmp ad-hoc,
torch.nn.attention.sdpa_kernel). Result: auto-selects EFFICIENT_ATTENTION (CUTLASS mem-efficient,
non-materializing, tensor-core) — NOT math. efficient vs math: S256 0.045 vs 0.328, S1024 0.297 vs
1.377, S2048 0.921 vs 4.449 ms (5-7x). D=72 supported on efficient; additive block-diagonal mask does
NOT drop it to math (efficient stays, 0.307ms). FLASH/cuDNN hard-gated sm80+. Cross-check: Codex's FA
microbench SDPA baseline (0.954ms@S2048) == efficient path -> FA lost to V100's BEST attention, not a
strawman. CONCLUSION: SDPA already extracts V100's optimal attention; no backend-steering free win.
Vision-encoder thread fully closed: V100 ViT stays on SDPA(efficient); FA/custom-kernel/cp.async-imitation
all can't beat CUTLASS's mem-efficient Volta pipeline at ViT's short seqs. Baseline is the ceiling.

## 2026-06-13 (wrap-up) — knowledge + implementation consolidation
Optimization phase closed. Deliverables:
- docs/V100_OPTIMIZATION_FINDINGS.md: consolidated source-of-truth (3 fronts + cp.async/regime principle).
- IMPLEMENTATION HARDENED: bundled autotuned JSONs -> src/fp8_w8a16_sm70/moe_configs/ (q35b-TP4, gemma-TP4);
  plugin auto-loads them via VLLM_TUNED_CONFIG_FOLDER when unset (guarded). Verified DEFAULT-ON e2e:
  serve q35b FP16 with zero extra flags -> auto-set folder + heuristic ACTIVE + JSON pickup + 66.25 tok/s
  (vs stock 15.6). Turn-key.
- tools/TOOLS.md: production vs experimental tool catalogue + results index.
- Memory consolidated (project_volta_moe_fp16_patch -> final shipped state).
Publication version derivable from V100_OPTIMIZATION_FINDINGS.md later (paused github publish / Ch1 reframe).
