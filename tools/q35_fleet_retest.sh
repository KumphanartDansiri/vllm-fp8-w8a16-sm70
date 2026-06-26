#!/usr/bin/env bash
# Bring the Qwen3.5 featured pair (27B + 35B-A3B) into the FLEET MATRIX condition.
# Runs each at the SAME condition as every flagship/fleet model (via perf_v2_cell.sh):
#   32768 ctx, chunked-prefill ON, decode = 256 tok median-of-5, correctness battery
#   (quality_status), short+long TTFT (+FA-on arm on 0.21), TP4, cudagraph, ns=8.
# FULL precision set FP16 / FP8 / GPTQ-Int4 (user override of the fleet's FP16+FP8-only
# tier rule — Int4 paths added to perf_v2_cell.sh for q27b35/q35b35), BOTH engines 0.19+0.21.
# 12 cells, SEQUENTIAL (TP4 = GPUs 0-3; clean single-cell measurement). ~4-5h. Box must be idle.
#
# Launch:
#   tmux new-session -d -s q35fleet 'cd /home/kumphanartd/vllm-fp8-w8a16-sm70 && \
#     ./tools/q35_fleet_retest.sh 2>&1 | tee ~/q35fleet.log; touch ~/q35fleet.done'
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
note(){ echo "[q35fleet $(date -u +%FT%TZ)] $*"; }

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1}END{print s+0}')
if [[ "${used:-9999}" -gt 4000 ]]; then note "ABORT: GPUs busy (${used}MiB)"; exit 1; fi
note "clean box (${used}MiB). Qwen3.5 27B+35B -> FLEET condition (32768/256tok/median5/chunked-on), FP16/FP8/Int4 x 0.19+0.21."

n=0; ok=0
for ENGINE in 021 019; do
  for MK in q27b35 q35b35; do
    for PREC in fp16 fp8 int4; do
      n=$((n+1))
      note "=== cell $n/12: $MK $PREC engine=$ENGINE ==="
      if MODEL_KEY="$MK" PREC="$PREC" ENGINE="$ENGINE" bash tools/perf_v2_cell.sh; then
        ok=$((ok+1)); note "cell $MK/$PREC/$ENGINE OK"
      else
        note "cell $MK/$PREC/$ENGINE rc=$? (infeasible/fail — continuing)"
      fi
    done
  done
done
note "DONE: $ok/$n cells OK. Results: results/perf_v2_q{27b35,35b35}_{fp16,fp8,int4}_{019,021}_*/rows.csv"
