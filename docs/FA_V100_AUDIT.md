# FlashAttention-V100 (ai-bond) ↔ vLLM Integration Audit

Shared function-by-function checklist for Claude + Codex + user to walk **sequentially, top-down**.
Goal: decide for each function whether dropping ai-bond into vLLM's FA path is `same` / `adapter-only` /
`kernel-divergent` / `unsupported` / and resolve every `unknown` by reading the cited evidence.

## Conventions
- **Walk order:** Section 1 → 2 → 3 → 4, defer Section 5. One row (or one small group) per discussion turn.
- **Each turn:** Claude and Codex each verify the row independently against the Evidence cell, then agree a Status.
- **Status vocab:**
  - `same` — name + params + semantics match the reference; no work.
  - `adapter-only` — matches modulo a Python-level arg reorder / passthrough; no kernel change.
  - `kernel-divergent` — semantics differ; needs CUDA-side work.
  - `gate` — not a kernel issue; a vLLM support-gate / registration that must be patched.
  - `unsupported` — absent on ai-bond side; decide if vLLM needs it.
  - `unknown — WALK` — not yet read; the point of the walk.
- **Refs:** ai-bond = `/home/kumphanartd/flash-attention-v100`; vLLM = `/home/kumphanartd/vllm-0.21` (0.19 byte-identical at the FA seam unless noted).

## Settled decisions (context for the walk)
- **vLLM 0.19 vs 0.21 FA contract = byte-identical.** Build once.
- **V1 uses ONLY `flash_attn_varlen_func`** (+ `block_table`) for BOTH prefill and decode. `with_kvcache` is never called in V1.
- **sm80 gate:** `_is_fa2_supported()` requires capability ≥ 8.0 → stock backend self-excludes on V100. Must be defeated (Section 2).
- **Block size:** ai-bond requires `page_block_size % 256 == 0`; vLLM default 16, advertises `MultipleOf(16)` (so 256 is legal). **Route A = run `--block-size 256` for the go/no-go microbench; defer Route B (teach ai-bond block_size=16).** Route A fragmentation tax (Codex math): ~6% avg / ~12% worst KV-headroom haircut at 282-seq concurrency; ~free for single-seq long prefill. Route B = real page-crossing staging (160-row tile straddles ~10 physical 16-tok pages), not a check relaxation.
  - **T7 AMENDMENT: Route A is NOT straddle-free at D=128.** `BLOCK_N_128=160` does not divide the 256 page → tile 1 (rows 160-319) crosses the boundary; the single-pointer linear load reads the wrong physical block for non-contiguous tables. Safe D: 32/64/256 (BLOCK_N divides 256). Unsafe: **128** (160), 16 (512). Real invariant = `page % BLOCK_N == 0`, not `page % 256 == 0`. Contingency: `BLOCK_N_128 160→128` (one line, forward.h). Driver auto-applies on stock-smoke failure.
- **No FP8 KV cache:** ai-bond varlen has no `q/k/v_descale` → KV must stay fp16.

## ⚠ THE FOUR FA INTEGRATION INVARIANTS (frozen 2026-06-11, Claude+Codex; each one e2e-earned)
Violating any of these reproduces a failure we hit and diagnosed on GPU. Do not relax without re-running the gates.
1. **`.so`-only exposure** — the serving container's PYTHONPATH gets `flash_attn_v100_cuda*.so` ONLY. Never ai-bond's `flash_attn`/`flash_attn_v100` python shims: vLLM's optional flash-attn probe half-succeeds on the shim (`flash_attn.ops` missing) and crashes GLM4 rotary init.
2. **dtype KV gate** — route to ai-bond only when the KV cache tensors are fp16 (`k.dtype == v.dtype == float16`). Do NOT gate on `k_descale`/`v_descale` presence: the Triton backend always passes 1.0-filled placeholder scale tensors for fp16 "auto" cache.
3. **`BLOCK_N_128 = 128`** (ai-bond `include/forward.h`, patched from 160) — with the 256-token page, BLOCK_N must divide the page or KV tiles straddle physical blocks and read foreign sequences' KV (the smoke's garbage-100.73 signature). General invariant: `page_block_size % BLOCK_N == 0`.
4. **dense-Q adapter boundary** — ai-bond's low-level `varlen_fwd` assumes dense `H_Q*D` query rows (checks only `stride(-1)==1`); vLLM passes q as a fused-QKV `.split()` view (row stride 1792≠1536 on GLM-Air/rank) → silent garbage at full speed. The adapter MUST densify (`q.contiguous()` when `q.stride(0)!=Hq*D`) and guard `out` the same way (temp + copy-back).

Gates that enforce them: `fa_v100_paged_smoke.py` (#3, contract), `fa_v100_longseq_check.py` (#3 at depth, layouts, #4 informational probe), adapter runtime gates in `fa_v100_prefill.py` (#2, #4), A/B harness staging (#1).

---

## Section 1 — Public Python API (upstream FlashAttention ↔ ai-bond)

| # | Function | vLLM Uses? | ai-bond Status | Divergence | Evidence |
|---|---|---|---|---|---|
| 1.1 | `flash_attn_func` | No (V1 uses varlen) | `same` | params identical to upstream | ai-bond `flash_attn_v100/flash_attn_interface.py:115`; verified |
| 1.2 | `flash_attn_varlen_func` | **Yes (load-bearing)** | `same` vs upstream / `adapter-only` vs vLLM-fork | upstream arg order `(…,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,…)` vs vLLM-fork order (see 2.1) | ai-bond `…interface.py:272`; verified |
| 1.3 | `flash_attn_with_kvcache` | No (unused in V1) | `same` | matches upstream; irrelevant to serving | ai-bond `…interface.py:324`; verified |
| 1.4 | `flash_attn_qkvpacked_func` | No | `unsupported` (moot) | absent in ai-bond **and** vLLM fork | grep: none in either; verified |
| 1.5 | `flash_attn_kvpacked_func` | No | `unsupported` (moot) | absent both | verified |
| 1.6 | `flash_attn_varlen_qkvpacked_func` | No | `unsupported` (moot) | absent both | verified |
| 1.7 | `flash_attn_varlen_kvpacked_func` | No | `unsupported` (moot) | absent both | verified |

**Section 1 verdict (Claude+Codex):** core 3 strong; packed helpers moot. **No open rows.**

---

## Section 2 — vLLM-facing API (the real serving contract)

| # | Symbol | Role | Status | Divergence / Action | Evidence |
|---|---|---|---|---|---|
| 2.1 | `vllm_flash_attn.flash_attn_varlen_func` | the call vLLM's backend makes | `adapter-only` | fork sig `(q,k,v,max_seqlen_q,cu_seqlens_q,max_seqlen_k,cu_seqlens_k=None,seqused_k=…,…,block_table,…,scheduler_metadata,q/k/v_descale,num_splits,fa_version,s_aux)` → reorder to ai-bond upstream; pass block_table/seqused_k through; force descale=None | `vllm_flash_attn/flash_attn_interface.py:176`; backend call site `v1/attention/backends/flash_attn.py:809`; verified |
| 2.2 | `get_scheduler_metadata` | FA3-only (sm90) | `same` (n/a on V100) | not called on Volta (FA2 path) → pass `scheduler_metadata=None` | `…flash_attn_interface.py:122,147` (`_vllm_fa3_C`); verified |
| 2.3 | `reshape_and_cache_flash` | vLLM's KV-cache WRITER | `same` (vLLM-owned) | NOT ai-bond's concern; stays a vLLM `_C` op; ai-bond only READS the cache via block_table | backend `…flash_attn.py:887`; verified |
| 2.4 | `is_fa_version_supported` / `fa_version_unsupported_reason` | support gate | `gate` | the sm80 wall lives here (`_is_fa2_supported → cap≥80`). Patch to admit sm70 + pin `fa_version`, OR register a separate sm70 backend | `vllm_flash_attn/flash_attn_interface.py:52-100`; `v1/.../fa_utils.py`; verified |

**Section 2 verdict:** only 2.1 is load-bearing kernel-adjacent; 2.4 is the gate to defeat. **No open rows.**

---

## Section 3 — Pybind / Torch-ops surface

| # | Symbol | ai-bond provides? | vLLM calls? | Status | Divergence / Action | Evidence |
|---|---|---|---|---|---|---|
| 3.1 | `fwd` / `bwd` / `varlen_fwd` / `varlen_bwd` / `fwd_kvcache` | Yes — on `flash_attn_v100_cuda` **and** `flash_attn_2_cuda` | indirectly (via python wrapper) | `same` (names) | FA2-style names replicated; `flash_attn_2_cuda` shell present | ai-bond `kernel/fused_mha_api.cpp:19-32`; verified |
| 3.2 | `torch.ops._vllm_fa2_C.varlen_fwd` | **No** | Yes (vLLM python wrapper dispatches here) | `decision` | If we shim at the **python** `flash_attn_varlen_func` level (2.1) we bypass `_vllm_fa2_C` entirely → simplest. Alternative: register ai-bond as `_vllm_fa2_C` alias (harder). **Lean: python-level shim.** | `vllm_flash_attn/flash_attn_interface.py:300`; verified |
| 3.3 | `torch.ops.flash_attn_v100.varlen_fwd` | Yes (TORCH_LIBRARY) | No | `same` | ai-bond also exposes a torch.ops path; usable if we prefer ops over pybind | `kernel/fused_mha_api.cpp:308`; verified |

**Section 3 verdict:** name parity good; the only choice is *where* to interpose (python wrapper vs op alias) — lean python-level. **Confirm 3.2 lean, then close.**

---

## Section 4 — Kernel function audit (THE varlen path vLLM needs) — **walk target**

Walk only the prefill/decode varlen path first. This is where block-size 256 + page-crossing live.

| # | ai-bond symbol | Location | Status | What to verify on the walk | Evidence so far |
|---|---|---|---|---|---|
| 4.1 | `flash_attention_varlen_forward` (host) | `kernel/fused_mha_forward_varlen.cu:383` | `kernel-divergent` (block size) | arg unpack, the `page_block_size % 256` check (:451), shape asserts, paged_KV flag | partially read; 256 check verified |
| 4.2 | `launcher_flash_attention_forward_varlen` | `…varlen.cu:293` | **`deferred → test.py`** | grid/block dims, D-dispatch — internal perf/correctness, not integration-blocking | name confirmed |
| 4.3 | fused varlen forward `__global__` kernel | `…varlen.cu:27` (`__launch_bounds__`) | **`deferred → test.py`** | overall dataflow — internal correctness, covered by test.py | confirmed entry |
| 4.4 | `BlockInfo` + `init_q()` | `include/template.h:16,36-89` | **`adapter-only` (RESOLVED T2)** | seqlen_k = `min(cu_seqlens_k diff, seqused_k)`; k_base unused in paged addressing → adapter must pass **length-valid** `cu_seqlens_k[i]=i*max_seqlen_k` (NOT a zero dummy). Kernel unchanged. | verified template.h:61-67,89 |
| 4.5 | **KV paged-load block** | `…varlen.cu:182-199` | **`kernel-divergent` (RESOLVED T2 — Route B only)** | Codex spot-check: load takes ONE `phys_page` for tile start, then stages a contiguous `valid_kv_rows` span → a tile crossing a page boundary reads the wrong physical block. So block_size=16 (tile 160 spans ~10 pages) REQUIRES page-crossing load changes = Route B. Route A (`--block-size 256`) avoids it. | layout match verified; Codex confirmed contiguous-span assumption |
| 4.6 | `WMMA_GEMM_LOAD_TILE` | `…varlen.cu:165,207,256` → `include/` | **`deferred → test.py`** (contiguous-span assumption already captured in 4.5/Route-B) | smem staging — internal; page-span issue already logged at 4.5 | callsites seen |
| 4.7 | `WMMA_GEMM_SCORES` (Q·Kᵀ + mask/alibi/softcap/window) | `include/mat_mul.h:24` | **`deferred → test.py`** | mask/alibi/softcap numerics — correctness, covered by test.py (args themselves forwarded 1:1) | def located |
| 4.8 | `WMMA_GEMM_SOFTMAX` + LSE write | `…varlen.cu:283-285` → `include/softmax.h` | **`adapter-compatible` (RESOLVED T4)** | LSE = `max + log(sum)` (`:285`), shape `[H_Q,T_Q]`=(nheads,total_q) — matches vLLM `merge_attn_states` `[NUM_HEADS,NUM_TOKENS]` + std logsumexp merge. Closes the Turn-3 LSE sub-item. | verified :285; Codex verified merge_attn_states |
| 4.9 | V-load + output accumulation (`WMMA_GEMM_GRADIENTS<dO_PV>` / `WMMA_GEMM_EPILOGUE<write_dO>`) | `…varlen.cu:267,278` | `unknown — WALK` | NOTE shared fwd/bwd macro names via `GemmType` template param — in fwd these do P·V + O-write | callsites seen |
| 4.10 | MMA primitive `WMMA_MMA_F32_F32` (m8n8k4 native / m16n16k16 fused) | `include/mma_m8n8k4.h:475`, `mma_m16n16k16.h`, sizes in `kernel.h:11-17` | `same` (sanctioned split) | the Volta tensor-core emulation; correctness is the build's `test.py` job, not integration | verified earlier |

**Section 4 status:** rows 4.1/4.5/4.10 partially known; 4.2-4.4, 4.6-4.9 are `unknown — WALK`. **Start the walk at 4.1.**

---

## Section 5 — Deferred (only after the varlen path is understood)

| # | Function | Why deferred |
|---|---|---|
| 5.1 | `flash_attention_forward` (dense, non-paged) | V1 doesn't use the dense path |
| 5.2 | `flash_attention_kvcache` | V1 never calls `with_kvcache` |
| 5.3 | `flash_attention_backward` | inference-only; no training |
| 5.4 | `flash_attention_varlen_backward` | inference-only |
| 5.5 | packed-API kernels | absent + unused (Section 1) |

---

## Walk log (append one entry per turn)

### Turn 1 — Row 4.1 `flash_attention_varlen_forward` host entry (`…varlen.cu:383-477`) — Claude (lead), pending Codex verify
Status proposed: **`kernel-divergent` (block size) + 3 new adapter constraints.** Found:
- **C1 — `cu_seqlens_k` is MANDATORY even in paged mode.** It's a non-optional `const at::Tensor&` and is asserted unconditionally (`:464-469`: int32/cuda/contig/1-D/`size==B+1`). But vLLM passes **`cu_seqlens_k=None`** on the paged path (decode + paged prefill) and uses `seqused_k` instead (`vllm_flash_attn` comment: "only used for non-paged prefill"; call site `flash_attn.py:809` omits it). → **Adapter must synthesize a `[B+1]` `cu_seqlens_k`.** Whether that's *adapter-only* (kernel ignores its values in paged mode, only needs shape/B) or *kernel-divergent* (kernel reads its values for offsets) is decided by **row 4.4** `BlockInfo::init_q` — template.h:61 shows `k_base=CU_SEQLENS_K[batch]` only in the NON-paged branch, so likely adapter-synthesizable. **This makes 4.4 the critical next row.**
- **C2 — fp16-only.** `:415` hard-requires `q == kFloat16` (k/v must match). No bf16. On V100 this is usually moot (no bf16 tensor cores; FP8-resident path already dequants to FP16, KV fp16) but **must confirm GLM-Air/Qwen serve as fp16, not bf16**.
- **C3 — `num_splits ≤ 1`.** `:434` rejects `num_splits>1`; vLLM passes `num_splits=attn_metadata.max_num_splits` (can be >1 for split-KV decode). → **Adapter forces `num_splits∈{0,1}`**; disables vLLM's split-KV decode opt but functionally fine.
- Confirmed benign: Volta-only guard (`:411`), GQA assert `H_Q%H_K==0` (`:433`), paged shape asserts match vLLM `[num_blocks,page_block_size,H_K,D]` (`:453-460`), `max_seqlen_q==1` causal-off micro-opt (`:437`).
- **Next (Claude leads):** row 4.4 `BlockInfo::init_q` (`include/template.h:36-77`) — resolve whether paged mode reads `cu_seqlens_k` values (→ C1 verdict) and how `seqlen_k` derives from `seqused_k`.
- **Codex action:** independently verify C1/C2/C3 against the cited lines; flag anything I over/under-called.

### Turn 2 — Rows 4.4 + 4.5 — Codex correction, Claude verified against source
- **C1 CORRECTED (Codex, Claude re-verified `template.h:61-67,89`):** my "shape-only dummy cu_seqlens_k" hint was WRONG. `seqlen_k = min(cu_seqlens_k_diff, seqused_k)`; a zero-diff dummy → seqlen_k=0 → no KV attended → broken. Adapter must pass **length-valid** `cu_seqlens_k[i] = i * max_seqlen_k` (diff = max_seqlen_k ≥ seqused_k[i] → clamp gives exact seqused_k[i]). `k_base` absolute base is unused in paged addressing, so only the diff matters. **C1 remains `adapter-only`** — kernel correct, adapter owes a real length-bearing tensor. → **Row 4.4 RESOLVED.**
- **C2 confirmed** (Codex): fp16-only at host check, no bf16 path.
- **C3 confirmed** (Codex): clamp `num_splits∈{0,1}` is correctness-safe, only disables split-KV decode opt.
- **Row 4.5 RESOLVED (Codex spot-check):** paged load stages one `phys_page` + contiguous `valid_kv_rows` span → block_size=16 needs page-crossing load = Route B. Route A avoids. (Decode-time decision, not a Route-A blocker.)
- **Open Section-4 rows remaining:** 4.2 (launcher), 4.3 (kernel dataflow), 4.6 (LOAD_TILE smem staging), 4.7 (scores+mask), 4.8 (softmax+LSE), 4.9 (V/output+return contract).
- **Next (Claude leads):** **the output/return contract** — does the host honor vLLM's preallocated `out=` in-place (call site passes `out=output[:num_actual_tokens]`) or allocate+return (→ adapter copy)? + LSE shape vs vLLM's `context_lse`/`suffix_lse` consumers (`flash_attn.py:854,931,966,1185,1213`). Host tail `…varlen.cu:477-578`.
- **Codex parallel option:** Row 4.7 `WMMA_GEMM_SCORES` (`mat_mul.h:24`) — confirm causal/window/alibi/softcap applied consistently with the vLLM args we forward.

### Turn 3 — Output / return contract (host tail `…varlen.cu:481-577`) — Claude (lead), pending Codex verify
Status proposed: **`adapter-only` — no copy, no blocker.** Found:
- **OUT is in-place** (`:530` `out_fp16 = out.has_value() ? out.value() : empty_like(q)`). vLLM passes `out=output[:num_actual_tokens]` → ai-bond writes directly into vLLM's preallocated buffer. `:537` asserts `out.sizes()==q.sizes()` — vLLM slices both q and out to `num_actual_tokens`, so it matches. **No output copy needed.** ✓
- **LSE shape = `[H_Q, T_Q]`** (`:531`) = (nheads, total_q) — matches Tri Dao varlen convention. **One OPEN sub-item:** vLLM's chunked-prefix / cascade paths (`flash_attn.py:931 context_lse`, `:966 query_lse`, merged via `merge_attn_states`) consume LSE; confirm `merge_attn_states` expects `[nheads, total_q]` (not transposed). **Only load-bearing if chunked-prefix/cascade active**; the unified main path (`:809`) ignores LSE entirely → irrelevant for the first PoC.
- **leftpad_k unsupported with paged** (`:484`) — fine, vLLM doesn't pass leftpad in the varlen path (it's a `with_kvcache`-only arg).
- **dmask / rng_state** always returned but empty/unused without dropout → adapter takes return `[0]`=out, `[1]`=lse, ignores `[2],[3]`. ✓
- **Return vector** `{out, softmax_lse, dmask, rng_state}` (`:577`) — adapter maps to vLLM's `(out[, lse])`. ✓
- **Net:** return contract introduces NO new blocker; lse-shape-vs-`merge_attn_states` is the only thing to confirm before enabling chunked-prefill/cascade.
- **Next (Claude leads):** Row 4.8 `WMMA_GEMM_SOFTMAX` (`softmax.h`) — confirm the LSE *value* (natural-log lse, `m + log(l)` convention) matches what `merge_attn_states` assumes, closing the lse sub-item end-to-end.
- **Codex action:** verify in-place-out reading; and (parallel) check vLLM `merge_attn_states` LSE input shape/convention to close the Turn-3 open sub-item.

### Turn 4 — Row 4.8 LSE + WALK CLOSE — Codex found, Claude verified (`varlen.cu:285`)
- **LSE sub-item CLOSED.** ai-bond `softmax_lse = sRowMax + logf(sum)` (`:285`, sum floored 1e-24) = standard `max+log(sum)`. Shape `[H_Q,T_Q]`=(nheads,total_q). vLLM `merge_attn_states` (Triton + DCP transpose evidence) expects `[NUM_HEADS,NUM_TOKENS]` + std logsumexp merge → **matches on shape AND value.** Chunked-prefix/cascade safe.
- **Rows 4.2/4.3/4.6/4.7 → `deferred → test.py`.** Internal correctness/perf, not integration-blocking. ai-bond's own `test.py` + the first microbench are the proof unless they fail.

## ✅ INTEGRATION-BLOCKING WALK COMPLETE (Claude + Codex concur)
No kernel blocker for Route A. Integration = **one python-level shim + sm80-gate patch + `--block-size 256`**, KV fp16. Every divergence found is adapter-mechanical (see Running adapter spec below). Next phase = BUILD + MICROBENCH (go/no-go), per [[feedback_handoff_scripts_shared_gpu]] hand the user a self-contained script.

## Running adapter spec (FROZEN at walk close; item 1 REFINED at T5)
1. Interpose at vLLM's `flash_attn_varlen_func` import seam, but the adapter body calls ai-bond's **LOW-LEVEL `flash_attn_v100_cuda.varlen_fwd`** — NOT ai-bond's public python wrapper, which hardcodes `seqused_k=None`/`out=None` and adds a post-hoc `.contiguous()` copy (`flash_attn_interface.py:206-219`; Codex T5 finding) — Row 3.2 + T5
2. Reorder args vLLM-fork → ai-bond upstream order — Row 2.1
3. If `block_table` set & `cu_seqlens_k is None`: synthesize `cu_seqlens_k[i]=i*max_seqlen_k` (length-valid, NOT zero dummy) — Row 4.4/C1
4. Clamp `num_splits → {0,1}` — Row 4.1/C3
5. Force `q/k/v_descale=None`; KV cache fp16 (no FP8-KV) — contract
6. `scheduler_metadata=None` (FA2 path) — Row 2.2
7. Pass `out=` straight through (in-place, no copy) — Row 4.9/T3
8. Defeat sm80 gate (patch `_is_fa2_supported`+pin fa_version, OR separate sm70 backend reg) + launch `--block-size 256` (Route A) — Row 2.4 / block-size
9. Assert fp16 serving dtype (no bf16) — Row 4.1/C2
LSE shape/value already compatible (Row 4.8) — no adapter action.

### Turn 5 — Build+microbench script review (Codex) → fixes applied (Claude)
Codex review of `tools/fa_v100_build_microbench.sh` + `fa_v100_paged_smoke.py` + `fa_v100_microbench.py` found 2 blocking issues, both verified and fixed:
- **B1 (real, spec-refining):** smoke called the PUBLIC `flash_attn_varlen_func`, but ai-bond's public wrapper **hardcodes `seqused_k=None` and `out=None`** (+ post-hoc `.contiguous()` copy) — verified `flash_attn_interface.py:206-219`. So the smoke wasn't testing in-place `out=`, `seqused_k`, or the C1 clamp at all. **FIX:** smoke + microbench now call low-level `flash_attn_v100_cuda.varlen_fwd` directly (the true adapter call path). **Spec item 1 refined accordingly** (adapter body must use low-level varlen_fwd, not the public wrapper).
- **B2 (real):** `test.py` exit code was logged but not gated; microbench exit also unpropagated. **FIX:** test.py failure → abort exit 7 before smoke/bench; bench failure → exit 8 (missing test.py = WARN+skip, failing test.py = hard abort).
- **Smoke hardened while fixing B1:** garbage-fill (100.0) of KV slots beyond `seqused_k` → any clamp failure blows up softmax → loud FAIL; `data_ptr()` identity check for in-place out; LSE `[H_Q,T_Q]` shape assert.
- **Shapes CONFIRMED (Codex, from GLM-Air config):** heads 96/8, head_dim 128, TP8 → per-rank **HQ=12 HK=1 D=128** (defaults were right); **`num_hidden_layers=46`** not 45 → LAYERS default corrected to 46 (the "45" in earlier prefill notes was the MoE-layer count).
- **Baseline decision:** SDPA anchor stays (clearly labeled); no standalone vLLM-Triton A/B for the first go/no-go — the documented ~42s residual is the bar. (Claude+Codex concur.)
- **Status: script APPROVED for run** (post-fix). Next = user launches `tools/fa_v100_build_microbench.sh` on a clean box.

### Turn 6 — Codex static re-review → 2 final fixes → STATIC REVIEW PASSED
- Codex confirmed the 22-arg low-level `varlen_fwd` order matches the C++/torchbind signature (the riskiest line) and all T5 fixes.
- **Fix 1 (required):** smoke reference built causal mask on CPU (`torch.arange(L)`) vs CUDA scores → device-mismatch crash. → `torch.arange(L, device=DEV)`. Applied.
- **Fix 2 (hardening):** driver now captures `PIP_RC=${PIPESTATUS[0]}` and aborts on build failure (exit 5) — a stale installed `flash_attn_v100` could otherwise mask a failed build via the import check alone. Applied.
- **STATIC REVIEW PASSED (Claude + Codex).** Script set is run-ready: `tools/fa_v100_build_microbench.sh` → next action = USER runs on clean box; results file `tools/fa_v100_microbench_<ts>.txt` comes back for joint go/no-go read.

### Turn 7 — Design-critique question → D=128 TILE/PAGE STRADDLE found (Claude), Codex independently verified
User asked (while waiting for GPU): "any implementation logic you'd do differently?" Answering it surfaced a likely real bug ON OUR EXACT SHAPE:
- **Finding:** main KV loop (`varlen.cu:174-200`) resolves ONE `phys_page` from each tile's START row, then `WMMA_GEMM_LOAD_TILE` stages `valid_kv_rows` LINEARLY from that pointer (Codex verified the ld.global linear staging — closing the one inferred piece). With page=256 and `BLOCK_N_128=160` (`forward.h`), tile 1 = rows 160-319 straddles the page boundary → reads the wrong physical block whenever block_table is non-contiguous. **D safety table:** 32(BLOCK_N 256)✓ 64(128)✓ 256(64)✓ — **128(160)✗ 16(512)✗**. The host `%256` check is the wrong invariant; correctness needs `page % BLOCK_N == 0`.
- Other "do differently" items logged for later (non-blocking): no split-KV/flash-decoding (`num_splits≤1`; decode parallelism = B×H_Q CTAs — host-side chunk+`merge_attn_states` is a possible cheap split); no software-pipelined global loads (no cp.async on sm70 → manual register prefetch is THE perf lever if TFLOP/s lands low; unused `DUAL_LOAD` knob exists); GQA K/V re-read ×(H_Q/H_K) per Q-head CTA (matters for decode only); forward path reuses backward-named macros (cosmetic).
- **DECISION (one-GPU-window strategy):** driver upgraded to self-contained contingency: stock build → smoke; on smoke FAIL (exit 1) auto-apply `BLOCK_N_128 160→128`, `rm -rf build` (headers not dep-tracked), rebuild, re-run test.py + smoke, bench whichever build passed. `AUTOPATCH=0` disables; `PREPATCH=1` skips the stock attempt. Backup `forward.h.bak_fa_audit` + git-revertible. One window ⇒ bug confirmed + fix validated + numbers.
- **Smoke interpretation guide:** stock FAIL→patched PASS = straddle confirmed, NOT "ai-bond hopeless". Stock PASS = our static analysis wrong somewhere (also informative). Patched FAIL = different bug, abort exit 6.

### Turn 8 — ON-GPU RUN (2026-06-11, GPU 0, ai-bond `a68f76e`, vllm-0.21 `ad7125a43`) → **VERDICT: GO**
Results file: `tools/fa_v100_microbench_20260611_084004.txt`. Three launches to green:
- **Launch 1 (vllm021-cu126 image): FAILED at build** — ai-bond `setup.py:141` pins `torch.version.cuda ≥ 12.9` (checks the TORCH WHEEL, not nvcc). **Volta torch islands:** 2.10+cu129 (last cu129 wheel w/ sm70) vs 2.11+cu126 (cu128/129 of 2.11 dropped Volta). Serving stack (0.21+cu126) and ai-bond's pin sit on DIFFERENT islands → integration needs pin-relax-and-rebuild-on-cu126 (Path A, try first) or a dedicated cu129 image (Path B; user OK with isolated variant image). cu129 image on box = `vllm-v100-py312-test:cu129` (carries vllm 0.18.0 + torch 2.10.0+cu129 + nvcc 12.9).
- **Launch 2 (test:cu129): kernels COMPILED CLEAN (~12 min), failed at pip dep step** (uninstall of debian-managed `wheel`) → fix: `pip install --no-deps` (driver updated).
- **Launch 3 (test:cu129): FULL GREEN.**
  - test.py (stock): **PASS** — ai-bond's own suite correct on V100 silicon.
  - Smoke (stock BLOCK_N_128=160): **FAIL exactly as predicted** — cos=0.004, max_abs=100.73 = the planted garbage value → wrong-physical-block read. In-place out= OK, LSE shape OK (contract mechanics fine even on stock).
  - **AUTO-CONTINGENCY: BLOCK_N_128 160→128 → rebuild → test.py PASS → smoke cos=1.000000 max_abs=0.002 PASS. STRADDLE BUG CONFIRMED + FIX VALIDATED (predicted T7 → triggered → fixed, closed loop).** Patch lives in working tree (`forward.h`, backup `.bak_fa_audit`); **worth upstreaming to ai-bond as issue/PR.**
  - **Bench (patched, 26k/HQ12/HK1/D128/block256):** ai-bond **112.6 ms/layer, 18.4 TFLOP/s, ×46 = 5.18s**. torch-SDPA anchor 59.2 ms/layer (×46=2.72s) — **anchor mislabel: V100 SDPA dispatches to cutlass mem-efficient FMHA which USES sm70 tensor cores** → it's a real fused kernel (not paged though). ai-bond = 0.53× of SDPA → headroom exists (pipelining), but paged-KV-native is the requirement and ai-bond has it.
  - **Triton A/B (`tools/fa_v100_triton_prefill_bench.py`, run in the REAL serving image vllm021-cu126): unified_attention = 945.4 ms/layer, 2.2 TFLOP/s, ×46 = 43.49s ≈ the documented ~42s residual (match within 1.5s!).** The residual was 100% the Triton prefill kernel, NOT TP all-reduce. **ai-bond is 8.4× faster.**
- **GO/NO-GO: GO.** Projected TTFT@26k: 60.2s → ~22s (2.7×); full Phase-4 journey 169→~22s (~7.7×). Next phase: implement the frozen 9-item adapter + sm70 gate; resolve Path A (relax pin, rebuild+re-gate under torch 2.11+cu126 — driver re-runnable with `AUTOPATCH`/pin patch) vs Path B (isolated cu129 image). Decode stays Triton+cudagraph (ai-bond has no split-KV; prefill-only integration).

### Turn 9 — CONVERGENCE (Claude + Codex, 2026-06-11) → AUDIT CLOSED, INTEGRATION PHASE OPENS
Codex concurred on the full causal chain (predicted straddle → confirmed → fixed → Triton 43.49s ≈ 42s residual → ai-bond 5.18s) and the GO. **Agreed integration order:**
1. **Adapter shim** — vLLM-facing `flash_attn_varlen_func` signature; body routes to LOW-LEVEL `flash_attn_v100_cuda.varlen_fwd`; frozen 9-item spec exactly (synth `cu_seqlens_k`, in-place `out=`, `num_splits≤1`, fp16-only, block 256, never the public ai-bond wrapper).
2. **sm70 gate, ENV-GATED not global** (Codex refinement, adopted): `VLLM_V100_FLASH_ATTN=1` opt-in → instant fallback to Triton, matching the project's kill-switch flag pattern.
3. **Path A first:** relax ai-bond's CUDA pin, build+test.py+smoke in the cu126 serving image (driver reusable). Pass → one serving image, no new island. Real ABI/toolchain failure → Path B isolated cu129 image is then justified.
4. **e2e GLM-Air TTFT@26k A/B** = the proof point for the 60→~22s projection.
**Explicit caveat (Codex, adopted): PREFILL-ONLY win.** Decode stays on Triton+cudagraph unless later measurements justify routing it (ai-bond lacks split-KV/paged-decode sophistication; decode at 30.7 tok/s is already in the comfortable band).
**Upstream:** BLOCK_N_128 straddle = real ai-bond correctness bug (paged KV, non-contiguous tables); file issue/PR with `fa_v100_paged_smoke.py` as reproducer.

### Turn 10 — PATH A VALIDATED + Stage-1 adapter IMPLEMENTED (2026-06-11)
- **Path A (cu126 serving image) GREEN** (`tools/fa_v100_microbench_20260611_122011.txt`): build OK under torch 2.11+cu126/nvcc 12.6, test.py PASS, smoke cos=1.000000, bench 109.9 ms/layer / 18.90 TF (≥ cu129 build). **One serving image; no cu129 island needed.**
- **The cu129 pin was REAL, not arbitrary:** ai-bond's softcap uses the `__tanhf` device intrinsic (CUDA ≥12.8/12.9 only) — under nvcc 12.6 it's host-only → compile error (`mat_mul.h:116,145`). Fix: `__tanhf`→`tanhf` (standard, all-CUDA). Also hit: cu126 image's setuptools does git introspection → needs `git config safe.directory` (cu129 image has no git → skipped it); ai-bond's pip deps clash with debian-managed `wheel` → `pip install --no-deps` (driver updated).
- **ai-bond working tree now carries 3 deliberate patches** (uncommitted): `include/forward.h` BLOCK_N_128 160→128 (correctness, upstream-worthy), `setup.py` pin 12.9→12.6 (+ tanhf makes it valid), `include/mat_mul.h` __tanhf→tanhf ×2 (compat, upstream-worthy as guarded ifdef). Built .so persists at `build/lib.linux-x86_64-cpython-312/` (PYTHONPATH-importable, no install).
- **Stage-1 integration DESIGN CHANGE (improvement over the Turn-9 plan):** vLLM uses ONE backend for prefill AND decode, so swapping in a FLASH_ATTN backend would route DECODE through ai-bond too (no split-KV → would regress the validated 30.7 tok/s cudagraph decode). Instead: **stay on TRITON_ATTN, interpose at `unified_attention`** (identical standard contract: q,k_cache,v_cache,out,cu_seqlens_q,seqused_k,block_table). Prefill batches (max_seqlen_q>1) → ai-bond low-level varlen_fwd; decode batches → original Triton (cudagraph FULL_DECODE_ONLY captures decode-only batches → never sees the ai-bond path → capture-safe by construction). NO selector/capability/version gates needed at all (gates 2.4/registry mapped but unused: `FlashAttentionBackend.supports_compute_capability` ≥(8,0) at flash_attn.py:205, `_is_fa2_supported` ≥80 — relevant only to the deferred full-backend Stage 2).
- **Implemented:** `src/fp8_w8a16_sm70/fa_v100_prefill.py` (env-gated `VLLM_V100_FLASH_ATTN`, default OFF; per-call fallback gates: dtype/block%256/dense-block-layout(handles `kv_cache.unbind(1)` strided views)/alibi/sinks/descale/kv-quant/chunk-lookback; synthesized cu_seqlens_k cached per step) + wired into `vllm_serve.py` (additive try/except). e2e A/B: `tools/fa_prefill_ttft_ab_vllm021.sh` — GLM-Air FP8 TP8 @26k, arms triton|fa identical except the flag, both `--block-size 256`, prefix-caching+chunked-prefill OFF (also keeps Sq==Sk = the smoke-validated causal alignment; Sq<Sk chunked alignment NOT yet smoke-covered — extend smoke before enabling chunked prefill).
- **RUNNING:** e2e A/B launched. Expected: triton TTFT ~60s, fa TTFT ~22s, decode tok/s IDENTICAL across arms (proves decode untouched).
### Turn 11 — e2e A/B debugging chain (2026-06-11): 3 gotchas → root cause = STRIDED Q
- **Baseline (triton arm) LOCKED:** TTFT@24,040tok = 51.85/51.78s, decode 22.96/22.95 tok/s, coherent, 0.1% trial spread. (Matches the documented curve: 60s@26k ⇒ ~52s@24k by quadratic attention scaling.)
- **fa attempt 1: silent full fallback** — my descale gate misread the contract: triton backend ALWAYS passes 1.0-filled `layer._k/_v_scale.expand(...)` tensors even for fp16 "auto" KV (`triton_attn.py` forward); presence ≠ quantization. Result: TTFT identical to baseline (51.8s), fallback banners on all ranks = the per-call-fallback safety property WORKED (wrong gate ⇒ slow, never broken). Fix: gate on **cache dtype** (k/v fp16) instead of descale presence; q_descale (truly None-or-unsupported) still gated.
- **fa attempt 2: PERF CONFIRMED, OUTPUT GARBAGE** — route banner on all 8 ranks, **TTFT 19.57/19.77s = the predicted ~19s (2.65× vs baseline)**, decode untouched (23.0), but output incoherent. Validation hole exposed: test.py+smoke ≤512 tok; the 26k microbench TIMED but never correctness-checked.
- **`tools/fa_v100_longseq_check.py`** (new keeper tool): paged sweep 512→24k, shuffled tables, separate AND vLLM-interleaved `unbind(1)` KV views → **ALL cos=1.000000** (long-seq + interleaved + Sq==Sk all correct; chunked prefill confirmed off in engine config — not the cause).
- **ROOT CAUSE (standalone-reproduced): ai-bond's low-level kernel assumes DENSE `H_Q*D` query rows.** vLLM passes q as a `.split()` view of the fused QKV projection — row stride 1792 (GLM-Air/rank) vs dense 1536. Host only checks `stride(-1)==1` → silent misread of every row >0. qkv-split-q test: cos=0.53/0.48 @512/2048. This is WHY ai-bond's public wrapper `.contiguous()`es q — and a real divergence from upstream FA2, whose varlen kernels take explicit q strides (upstream feedback item #3). **Adapter fix: densify q when `q.stride(0)!=Hq*D` (~0.2ms copy @24k vs ~5s attention) + temp-buffer/copy-back guard for non-dense out.**
### Turn 12 — ✅ E2E WIN (2026-06-11, final): TTFT@24k 51.8s → 19.45s (2.66×), decode untouched, coherent
fa arm rerun with the strided-q fix: **trial1 19.44s / trial2 19.47s TTFT** (baseline 51.85/51.78), decode 22.96/22.94 vs 22.96/22.95 (digit-identical), output coherent AND tracking the baseline's greedy thinking-trace (strongest e2e exactness signal the harness shows; trial2 correctly read the per-trial seed word from the 24k prompt). Route banner on all 8 ranks. **The full chain held: audit → microbench (kernel 8.4×) → adapter → e2e (TTFT 2.66×; remaining ~14s is non-attention prefill: MoE/linears/sampling).** Combined with Phase 4, the GLM-Air prefill journey is 169s → ~19.5s (~8.7×) at this depth.
**Remaining before promoting the flag default-ON:** (a) Sq<Sk (chunked-prefill/prefix) causal alignment still UNVALIDATED — add a prefix case to `fa_v100_longseq_check.py` and gate or validate before ever enabling chunked prefill/prefix caching with FA; (b) concurrency soak (multi-user prefill+decode mix); (c) formal exactness A/B (logprobs) if wanted for the benchmark chapters; (d) update `docs/GLM45_AIR_V100_CONFIG.md` once promoted. **Upstream package for ai-bond (3 items + reproducers):** BLOCK_N_128 straddle fix, `__tanhf`→guarded `tanhf` (+pin 12.6), strided-q (dense-row assumption vs upstream FA2's explicit q strides).

### Turn 13 — CONVERGED CLOSE-OUT (Claude + Codex): evidence hygiene + invariants frozen
Codex concurred ("integration-successful, not just promising") and requested the final artifact set; all applied:
(a) per-run trial logs now reset at arm start (no mixed-evidence files); (b) one-time q-densify log line (`densify q stride=(1792,128,1) -> contiguous`) so the root-cause fix is visible in e2e evidence; (c) `fa_v100_longseq_check.py` promoted into the driver as hard gate [6b] (qkv-split case reframed as informational probe — the raw-kernel limitation is by-design until upstream adds q strides); (d) **THE FOUR FA INTEGRATION INVARIANTS recorded at the top of this file** (.so-only exposure / dtype KV gate / BLOCK_N_128=128 / dense-Q adapter boundary); (e) durable results copied to `results/fa_ttft_ab_20260611/`. Upstream items kept separate per Codex: straddle (correctness), tanhf+pin (toolchain), strided-q (interface semantics — long-term question: adapter densify vs ai-bond learning explicit Q strides; densify is fine at ~0.2ms/24k, upstream strides is the better end state).

- **e2e GOTCHA #1 (both arms crashed at GLM4 model init, fixed):** putting ai-bond's `build/lib.../` on PYTHONPATH exposes its **`flash_attn` compatibility shim** to vLLM, whose optional flash-attn probe then half-succeeds — `import flash_attn` OK but `from flash_attn.ops.triton.rotary import apply_rotary` (vllm `rotary_embedding/common.py:138`) fails → `ModuleNotFoundError: flash_attn.ops` → worker death. **DEPLOYMENT RULE: expose ONLY `flash_attn_v100_cuda*.so` to the serving container** (the adapter uses the low-level module exclusively); never the `flash_attn`/`flash_attn_v100` python shims. Ironic but correct: the Tri-Dao-faithful shell that makes ai-bond a clean drop-in for *flash-attn users* is exactly what must be hidden from *vLLM*, which probes flash-attn as an optional extra it was built without. A/B script now stages the bare .so (`$OUT/pylib`).
