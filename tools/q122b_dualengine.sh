#!/usr/bin/env bash
# Qwen3.5-122B-A10B dual-engine CONSISTENCY run: FP8 + GPTQ-Int4 on vLLM 0.21 AND 0.19,
# TP8, 4096 ctx / 512 tok / ns=8 / cudagraph — the SAME harness + params as the 27B/35B triad,
# so the whole Qwen3.5 family (27B, 35B-A3B, 122B) is measured identically across both engines.
# 122B = TP8 (all 8 GPUs), so the two engines run SEQUENTIALLY. Box must be idle.
#
# 0.21 side: tools/tp_concurrency_sweep.sh (q122b:fp8 / q122b:int4), image vllm-v100:vllm021-cu126.
# 0.19 side: tools/triad_compat_vllm019.sh (MODELS=q122b, GPU_BASE=0 for TP8), image vllm019-cu126.
# Results: results/q122b_021_{fp8,int4}_<stamp>/SUMMARY.txt + results/q122b_019_<stamp>/SUMMARY.txt
#
# Launch:
#   tmux new-session -d -s q122b 'cd /home/kumphanartd/vllm-fp8-w8a16-sm70 && \
#     ./tools/q122b_dualengine.sh 2>&1 | tee ~/q122b_dual.log; touch ~/q122b_dual.done'
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STAMP="$(date -u +%Y%m%d_%H%M%S)"
note(){ echo "[q122b-dual $(date -u +%FT%TZ)] $*"; }

# clean-box guard — 122B needs every GPU
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1}END{print s+0}')
if [[ "${used:-9999}" -gt 4000 ]]; then
  note "ABORT: GPUs busy (${used}MiB used) — 122B TP8 needs an idle box."; exit 1
fi
note "clean box (${used}MiB). Qwen3.5-122B-A10B dual-engine consistency run (4096 ctx, TP8)."

# ---- vLLM 0.21 + cu126 (FP8 then Int4, sequential) ----
for prec in fp8 int4; do
  note "=== 0.21/cu126 q122b $prec TP8 ==="
  MODEL_KEY=q122b PREC="$prec" TPS=8 USERS="1 2 4 8" MAXLEN=4096 GENTOK=512 NRUN=2 \
    GPUMEM=0.90 HEALTH_TIMEOUT=2400 PORT=8040 OUT="results/q122b_021_${prec}_${STAMP}" \
    ./tools/tp_concurrency_sweep.sh
done

# ---- vLLM 0.19 + cu126 (FP8 + Int4 in one invocation; GPU_BASE=0 for TP8) ----
note "=== 0.19/cu126 q122b fp8+int4 TP8 ==="
MODELS=q122b PRECS="fp8 int4" TPS=8 GPU_BASE=0 USERS="1 2 4 8" MAXLEN=4096 GENTOK=512 NRUN=2 \
  GPUMEM=0.90 HEALTH_TIMEOUT=2400 PORT=8050 OUT="results/q122b_019_${STAMP}" \
  ./tools/triad_compat_vllm019.sh

note "DONE. Results: results/q122b_021_fp8_${STAMP} , results/q122b_021_int4_${STAMP} , results/q122b_019_${STAMP}"
