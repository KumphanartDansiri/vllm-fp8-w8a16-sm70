# V100 (sm_70) fused-MoE tuned configs

Per-M Triton fused_moe configs autotuned on Tesla V100-SXM2-32GB (vLLM 0.21, Triton 3.6)
by `tools/moe_volta_tune_fleet.sh` (feasibility-pruned shell-walk). Loaded automatically by
`fp8_w8a16_sm70.vllm_serve` via VLLM_TUNED_CONFIG_FOLDER (it sets this to this dir if you
haven't). Names encode (E, N=per-rank-shard-intermediate, device) — so they are TP-SPECIFIC:

- E=256,N=128 : Qwen3.6-35B-A3B  @ TP4 (moe_intermediate 512 / 4)
- E=128,N=176 : gemma-4-26B-A4B   @ TP4 (704 / 4)

Other models/TP sizes that don't match a file here fall back to the plugin's sm<80
get_default_config heuristic (BLOCK_K=64), which is itself ~4-9x over stock. Re-tune per
(model, TP) with tools/moe_volta_tune.py if you want exact configs for a new shape.
Unquantized (fp16/bf16) MoE only; our FP8 path uses its own kernels and ignores these.
