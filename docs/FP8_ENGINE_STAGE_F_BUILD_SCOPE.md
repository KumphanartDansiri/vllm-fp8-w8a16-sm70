# Stage F — engine-build scope (PLANNING ONLY — do not implement yet)

Branch `fp8-engine-stage-f-build`. Per review (Codex): scope the build before coding; implement
only after push + review. This doc answers: exact LMDeploy source, minimal subtree, build deps,
resulting op names, and how we avoid importing the whole TurboMind runtime.

Goal: make `turbomind_fp8_backend.ops_available()` True in our own vLLM image so the Stage-E
adapter's `turbomind` path goes live — sourcing the engine from **upstream lmdeploy**, not 1catai.

## 1. Exact source — PINNED
- **InternLM/lmdeploy `v0.14.0`** (latest stable; `Config_E4M3` verified present in
  `src/turbomind/kernels/gemm/arch/config_sm70_s884.h` AND registered in `kernel/sm70_884_8.cu`).
  Pin the tag/commit, not `main`. (Stage-E source audit established the engine is upstream.)

## 2. Sourcing method — RECOMMEND: copied minimal subtree (not full vendor, not submodule)
| option | verdict |
|---|---|
| **copied minimal subtree** (~1 MB, `third_party/turbomind_gemm_sm70/` + LICENSE + PROVENANCE) | **RECOMMENDED** — self-contained, we control exactly what compiles, ai-bond-style minimalism, easy to audit; cost = manual re-sync on lmdeploy bumps (rare) |
| git submodule @ v0.14.0, sparse-build gemm only | clean provenance but pulls whole lmdeploy repo + submodule/build friction |
| full vendor (1catai-style) | rejected — inherits the sprawl we chose to avoid |
Attribution: keep upstream Apache-2.0 headers; add `PROVENANCE.md` (exact tag + file list + re-sync steps).

## 3. Minimal subtree (avoids the whole TurboMind runtime)
The gemm subsystem is **880 KB** + core **180 KB**; the sm70 path is nearly self-contained. Runtime
coupling is a **single header** (`kernels/attention/quantization.h`, pulled by `convert.cuh` +
`transform.h`) — NOT the llama model (the `models/llama/*` includes are only in `test/` files we drop).

**INCLUDE:**
- `kernels/gemm/` shared: `gemm.cu`, `registry.{h,cu}` (trimmed — see below), `convert.cuh`,
  `convert_v3.cu`, `cast.h`, `types.h`, `utils.h`, `desc.h`, `arch.h`, `transform.h`, `epilogue*`,
  `context/tuner/dispatch` headers the sm70 config needs.
- sm70-specific: `mainloop_sm70.h`, `iterator_sm70.h`, `scheduler_sm70.cuh`,
  `arch/{mma_sm70.h, operand_sm70_s884.h, config_sm70_s884.h}`, `kernel/sm70_884_{4,8,16}.cu`.
- `kernels/core/*` (common, meta, math, array_ops, data_type, layout, smem, array, mma, sync).
- `core/*` (data_type, core, check, cuda_data_type, allocator, tensor).
- `kernels/attention/quantization.h` (the one coupling header) + whatever *it* pulls (expected: core only — VERIFY).
- CUTLASS **headers only** (header-only lib) for `mainloop_sm70.h`.

**EXCLUDE (prune):**
- All non-sm70 kernels: `kernel/sm{75,80,90}_*`, `gemm_universal_sm90*`, `mainloop_sm80_v2.h`,
  `kernel_impl_sm90.h`, `tma.cu` (Hopper TMA) — these are what drag in heavy CUTLASS/sm90.
- `test/` (testbed_v3.h, quantization_impl.h — the only `models/llama` referencers).
- **Trim `registry.cu`** to register ONLY `sm70_884_{4,8,16}()` (drop sm75/sm80/sm90 registrations →
  no sm90 kernels compiled → CUTLASS surface shrinks to what mainloop_sm70.h needs).

## 4. Build deps + how it slots into our image
- Base: our `vllm-v100:vllm021-cu126` (already a source-built vLLM w/ toolchain). Add: nothing new if
  it has nvcc+cmake; else copy CUDA toolkit like `Dockerfile.fp8builder` did (that used cu128 — we
  target **cu126** to match our image; VERIFY the sm70 s884 templates compile under CUDA 12.6, expected
  yes: HMMA m8n8k4 is Volta, no cu128-only feature).
- CUTLASS: header-only; vendor the needed headers or FetchContent (needs `git` in the image).
- Compile the subtree + our bindings into a small extension (`_fp8_sm70_C`) loaded alongside our JIT
  ext, OR extend the existing ext. Keep it OFF the default build until validated (flag-gated).

## 5. Op bindings we expose (our OWN thin csrc — informed by 1catai, never `_auto`)
Matching the contract the Stage-E adapter already calls:
- `fp8_sm70_prepare(qweight[N,K] fp8, scales[N/128,K/128] fp32, group_size) -> (tm_w, tm_s, meta[k_ld,q_ld])`
- `fp8_gemm_sm70_out(out, x, tm_w, tm_s, group_size, k_ld, q_ld)`   ← explicit ld only; **no `_auto`**
- `fp8_moe_gemm_sm70_out(out, sorted_x, expert_offsets, ptrs_w, ptrs_s, E, k, n, group_size, gated_silu)`
- `awq_moe_build_strided_ptrs(tm_w, tm_s, k_ld, q_ld, E) -> (ptrs_w, ptrs_s)`
These wrap the gemm public API (`gemm.h`, `convert.h`, `cast.h`, `types.h`, `utils.h`,
`core/data_type.h`). We deliberately do NOT port `_auto`/`_meta` — `prepare→meta→gemm` is the contract.
(Result op names must match `turbomind_fp8_backend.ops_available()`'s checks.)

## 6. Avoiding the whole TurboMind runtime — the crux
- Only `kernels/attention/quantization.h` couples gemm→attention; vendor that one header + verify its
  transitive deps stay within `kernels/core`/`core`. If it pulls more, stub the one helper it uses.
- Registry trim to sm70 means NO sm90/tma/CUTLASS-heavy compilation.
- We link NOTHING from `models/`, `engine/`, or the turbomind server — only the gemm kernels.

## 7. Open risks / verify-first (before/at implementation)
- [ ] `attention/quantization.h` transitive includes (confirm core-only; else stub).
- [ ] sm70 s884 templates compile under **CUDA 12.6** (our image) not just 12.8.
- [ ] CUTLASS header set actually needed by `mainloop_sm70.h` after sm90 prune (minimize).
- [ ] ABI/toolchain match with our torch (cu126) so `import` + op registration works.
- [ ] Full diff vendored-1catai-lmdeploy ↔ v0.14.0 to confirm no essential 1catai *engine* patches
  (Stage-E residual; expected none — deltas were in the vLLM wrapper).

## 8. Stage F validation (after the build lands)
1. `ops_available()` True → adapter self-check picks turbomind for block-128.
2. Re-run `stage_d_format_gate.py` + `stage_d_full_moe.py` in OUR image (not 1catai's) → cos=1.0000.
3. Wire `select_backend()` into `compressed_tensors_v100.py` (turbomind prepare at load; ours else).
4. Serving: TP≤4 (block-128 clean), resolve TP8 MoE sharding (expert-parallel vs intermediate vs pad
   `I/tp`→128), eager + cudagraph, real-prompt numerical sanity + throughput vs current ours-only.

## Estimate
Bounded: ~1 MB subtree, one coupling header, sm70-only compile. The risk is build-integration
(CUTLASS prune + CUDA 12.6 + ABI), not algorithmic. Recommend a throwaway build spike first
(compile the trimmed subtree + a single `fp8_gemm_sm70_out` binding, run the Stage-D gate in our
image) before wiring the loader.
