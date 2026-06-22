#!/usr/bin/env bash
# Finish the glm47 (GLM-4.7-Flash, MLA / Glm4MoeLite, BF16/fp16-only) perf_v2 cell on
# BOTH engines. Produces, per engine, the two dirs the consolidation expects:
#   - main     (default phase): correctness battery + decode C1..C8 + cold TTFT (ttft_long)
#   - ttftboth (TTFT_BOTH=1)  : cold+warm TTFT reconcile (ttft_long_cold / _warm)
# => 4 cell runs total: {021,019} x {main, ttftboth}. glm47 is NOT FA-eligible (MLA, not
# MHA/GQA) so there is no FA-on arm. 019 uses the inline MLACommonImpl MLA-prefill hook;
# 021 uses the dedicated FlashAttnPrefillBackend path (both env-gated, see fa_v100_mla_prefill.py).
#
# Clean-box batch — LAUNCH YOURSELF on an idle box; each serve self-guards (skips if the
# target GPUs hold >2GB, i.e. son's training is live) and captures to results/ files.
#
# Usage:   bash tools/perf_v2_glm47_finish.sh
# Knobs:   ENGINES (default "021 019")  USERS (default "1 2 4 8")  NRUN (5)  TTFT_REPS (3)
#   If C8 MLA decode-cudagraph capture destabilizes, re-run with: USERS="1 2 4" bash ...
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENGINES="${ENGINES:-021 019}"
export USERS="${USERS:-1 2 4 8}"
export NRUN="${NRUN:-5}"
export TTFT_REPS="${TTFT_REPS:-3}"
export MODEL_KEY=glm47 PREC=fp16

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results/perf_v2_glm47_finish_${STAMP}.log"
echo "[glm47-finish] engines='$ENGINES' users='$USERS' nrun=$NRUN ttft_reps=$TTFT_REPS -> $LOG"

run(){ # $1=ENGINE  $2=phase-label  $3...=extra env (KEY=VAL)
  local eng="$1" label="$2"; shift 2
  echo "=== glm47 fp16 $eng [$label] ===" | tee -a "$LOG"
  env ENGINE="$eng" "$@" bash tools/perf_v2_cell.sh 2>&1 | tee -a "$LOG"
  echo "=== glm47 fp16 $eng [$label] rc=${PIPESTATUS[0]} ===" | tee -a "$LOG"
}

for eng in $ENGINES; do
  run "$eng" main                          # battery + decode + cold TTFT
  run "$eng" ttftboth   TTFT_BOTH=1        # cold+warm TTFT reconcile
done

echo "[glm47-finish] DONE. Fold into the matrix: python3 tools/perf_v2_consolidate.py" | tee -a "$LOG"
