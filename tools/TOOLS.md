# Tools catalogue — production vs experimental

Quick map of what's a reusable harness vs a one-shot investigation script, so the next person
(or session) knows what to run and what was scaffolding. Focus = the 2026-06-12/13 MoE+vision arc;
see `docs/V100_OPTIMIZATION_FINDINGS.md` for the findings these produced.

## Production / reusable harnesses

| tool | what it does | when to use |
|---|---|---|
| `moe_volta_tune.py` | Single-GPU Volta fused-MoE tuner: a-priori SMEM feasibility prune + shell-walk early-stop; emits canonical `E=,N=,device_name=Tesla_V100.json`. Env: `TUNE_E/TOPK/K/NSHARD/M`. | Re-tune a NEW (model, TP) MoE shape for V100. |
| `moe_volta_tune_fleet.sh` | Fans `moe_volta_tune.py` across all 8 GPUs (18 (shape,M) jobs), merges per-M → the two canonical JSONs. ~50 min. | Bulk re-tune both reference shapes (or extend `SHAPES`). |
| `moe_stages_ab_vllm021.sh` | e2e A/B harness. Arms: `base` (stock), `s2/s3/s4` (num_stages JSON), `kbest` (hand ladder), `auto` (bundled tuned JSON), `plugin` (the shipped monkey-patch). `NUSERS=N` for concurrent-stream decode. Self-verifies config pickup. | Validate any MoE config/patch change on the real model, 1- or N-user. |
| `moe_fp8_profile_decode.sh` | Serves an FP8 MoE model with the built-in per-section profiler on (eager; `NUSERS` for batch M). Dumps route/GEMV/scatter GPU-time buckets. | Profile where FP8 MoE decode time goes for a shape. |

## One-shot investigation scripts (kept for provenance; rerun only to re-confirm)

| tool | the question it answered | result |
|---|---|---|
| `moe_decode_microbench.py` | Where does FP16 MoE decode time go? | `fused_moe_kernel` = 98.9%, 90× off floor |
| `moe_decode_tile_sweep.py` | Which tile axis is the lever? (`SWEEP_M`, `SWEEP_E/K/NSHARD`) | `BLOCK_K`; 64 vs 128 = 2.3–9.6× |
| `vit_fa_v100_d72_microbench.py` / `.sh` | Can padded FA-V100 (D=72→128) beat SDPA for ViT? | No — 0.37–0.47× SDPA |

## Shipped implementation (not a tool — for reference)

- `src/fp8_w8a16_sm70/vllm_serve.py::_patch_volta_moe_default_config` — the default-ON sm<80 MoE
  heuristic + auto-load of the bundled configs (env `VLLM_V100_MOE_FP16_TUNED`, default 1).
- `src/fp8_w8a16_sm70/moe_configs/` — bundled autotuned per-M JSONs (q35b-TP4, g26b-TP4),
  auto-loaded via `VLLM_TUNED_CONFIG_FOLDER` when unset.

## Results index (durable artifacts)

- `results/moe_stages_ab_*` — e2e A/B runs (num_stages null, kbest 4×, auto wins, 8-user 7×).
- `results/moe_decode_msweep_*` — M=1..16 tile sweeps.
- `results/moe_volta_tune_{q35b,g26b}/` — the merged canonical tuned JSONs (= what's bundled).
- `results/moe_volta_fleet_20260613_034003/` — per-(shape,M) fleet-tune outputs.
- `results/moe_fp8_profile_20260613_072405/` — FP8 decode profile (M=1 and M=8 buckets).
- `results/vit_fa_v100_d72_2026061308*` — vision FA no-go microbench.
