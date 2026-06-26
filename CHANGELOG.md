# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/) (`0.x`: minor = features/behavior,
patch = fixes).

## [Unreleased]

## [0.6.0] - 2026-06-24

The June optimization release: the headline V100 capabilities (coalesced +
branchless-dequant FP8, the FlashAttention-V100 bridge, the FP16-MoE fix, and
dual-engine 0.19/0.21 support) all land here. The companion benchmark story is
the `v100-vllm-2026` write-up.

### Added
- **FlashAttention-V100 prefill + MLA bridge** (`VLLM_V100_FLASH_ATTN=1`) — wraps
  the [ai-bond/flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)
  kernel for MHA/GQA and MLA prefill on `sm_70`; GLM-Air TTFT@24k 51.8s → 19.45s
  (2.66×), 8-model fleet verified. Credited in the README Acknowledgements.
- **vLLM 0.21 sm_70 source-build + FP8 W8A16 plugin port** (no code changes) —
  dual-engine 0.19 + 0.21 support; CT-FP8 MoE handles both engine layouts.
- **Coalesced FP8 W8A16 decode GEMV** (`VLLM_V100_FP8_COALESCED_GEMV=1`),
  including grouped-coalesced MoE-w13 — FP8-resident decode at ~FP16 speed, half
  the memory (GLM-4.5-Air 30.7 → 56.6 tok/s).
- **MTP `k>1` for Qwen MoE FP8** — `VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=512`
  fixes the cudagraph-capture crash; 122B optimum k=3 (1.68×), 35B optimum k=2 (1.24×).
- env-gated V100 MLA prefill hook.
- Performance Experiment v2 harness + dual-engine benchmark matrix
  (`results/perf_v2_COMBINED.csv`, snapshot tag `fp8-v100-2026-matrix`); Qwen3.5
  dense+MoE exact triad (`results/qwen35_triad_matrix_20260624.csv`).

### Changed
- **Branchless E4M3→FP16 dequant promoted to canonical.** The branchy converter
  (not `sm_70` itself) was the dense-FP8 limiter; with the branchless path,
  **dense FP8 now beats FP16 at 1–2 users and ties at 4** (was ~0.31× FP16).
  Sparse MoE wins at every concurrency.
- README narrowed to the plugin artifact; the V100 story, full benchmark matrix,
  and MTP deep-dive moved to the `v100-vllm-2026` write-up. Engine docs
  re-anchored to **0.19 + 0.21** (0.18 → legacy).
- Canonical serve example no longer disables chunked prefill (leave it ON).

### Fixed
- **Volta FP16-MoE config fix** — the default decode `BLOCK_K=128` made FP16 MoE
  slower than dense on V100; a default-on heuristic + bundled autotuned JSONs
  restore it (35B-A3B 15.6 → 65.9 tok/s, 4–9× e2e). FP8 path untouched.
- `REQUIREMENTS.md`: corrected the chunked-prefill row (it wrongly *required* the
  crash-causing `--no-enable-chunked-prefill`) and the engine baseline (0.18
  "production anchor" → 0.19/0.21 are the FP8 baselines).

### Docs
- Credit the ai-bond FlashAttention-V100 fork (Acknowledgements).
- gitignore scratch `results/` (published-evidence dirs stay tracked).

## [0.5.0] - 2026-06-07
- **vLLM 0.19 source-built `sm_70` baseline.** FP8 W8A16 + cudagraph + MTP port
  at 0.18 parity (Qwen3.6-35B-A3B-FP8: 52.27 cudagraph / 62.54 +MTP tok/s).

## [0.4.2] - 2026-05-30
- ns=8 re-verified release — headline numbers re-measured at the supported
  `--max-num-seqs 8` (35B-A3B-FP8 TP4 = 52.4 tok/s). `ns=1` crashes hybrid models
  under cudagraph (upstream vLLM bug).

## [0.4.1] - 2026-05-30
- Opt-in Qwen3.5 MTP speculative decoding (`ENABLE_QWEN_MTP=1`, default OFF);
  122B-A10B-FP8 TP8 1.36×.

## [0.4.0] - 2026-05-29
- 122B-A10B-FP8 at 34.76 tok/s on Python 3.12 + cudagraph.

## [0.3.2] - 2026-05-25
- Stage 2D close-out: GDN attribution + cross-cutting TP all-reduce finding
  (instrumentation only; no behavior change).

## [0.3.1] - 2026-05-25
- Stage 2D: measurement-only MoE wrapper attribution (instrumentation only).

## [0.3.0] - 2026-05-25
- Stage 2C: profile hygiene + layer caches (122B-A10B-FP8 TP8: 5.09 tok/s).

## [0.2.0] - 2026-05-24
- Stage 2A active-list + Stage 2B grouped routed GEMM, default-on for V100 —
  1.86× long-run decode on Qwen3.5-122B-A10B at TP8 (2.62 → 4.87 tok/s).

[Unreleased]: https://github.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/compare/v0.6.0...main
[0.6.0]: https://github.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/releases/tag/v0.2.0
