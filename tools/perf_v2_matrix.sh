#!/usr/bin/env bash
# Performance Experiment v2 — MATRIX DRIVER. Iterates the fleet x precision x engine,
# calling perf_v2_cell.sh per cell. Infeasible cells (missing model / OOM / never
# healthy) are logged and skipped, never zeroed. Clean-box batch — launch yourself.
#
# Default matrix = the 6-model fleet on BOTH engines (0.19 + 0.21), fp8 everywhere +
# the fitting reference (fp16 dense/MoE, int4 for 122B). Override CELLS to subset.
#
# Usage:
#   bash tools/perf_v2_matrix.sh                       # full matrix
#   CELLS="q27b:fp8:021 q27b:fp16:021" bash tools/perf_v2_matrix.sh   # subset
#   ENGINES="021" bash tools/perf_v2_matrix.sh         # one engine
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENGINES="${ENGINES:-019 021}"
# model:prec pairs (engine appended per ENGINES). int4 only for 122B; fp16 where it fits.
PAIRS_DEFAULT="q27b:fp8 q27b:fp16 q35b:fp8 q35b:fp16 q122b:fp8 q122b:int4 glm:fp8 g31b:fp8 g31b:fp16 g26b:fp8 g26b:fp16"
PAIRS="${PAIRS:-$PAIRS_DEFAULT}"

# Build CELLS unless explicitly given
if [[ -z "${CELLS:-}" ]]; then
  CELLS=""
  for e in $ENGINES; do for p in $PAIRS; do CELLS+="$p:$e "; done; done
fi

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="results/perf_v2_matrix_${STAMP}.log"; mkdir -p results
echo "[matrix] cells: $CELLS" | tee "$LOG"
echo "[matrix] start $(date -u +%FT%TZ)" | tee -a "$LOG"

for cell in $CELLS; do
  IFS=':' read -r mk prec eng <<<"$cell"
  echo "" | tee -a "$LOG"
  echo "========================================================" | tee -a "$LOG"
  echo "[matrix] CELL $mk:$prec:$eng  $(date -u +%FT%TZ)" | tee -a "$LOG"
  MODEL_KEY="$mk" PREC="$prec" ENGINE="$eng" bash tools/perf_v2_cell.sh 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  case "$rc" in
    0) echo "[matrix] $cell OK" | tee -a "$LOG" ;;
    2) echo "[matrix] $cell SKIP (infeasible combo / missing model)" | tee -a "$LOG" ;;
    3) echo "[matrix] $cell SKIP (serve failed / OOM at min TP)" | tee -a "$LOG" ;;
    *) echo "[matrix] $cell ERROR rc=$rc" | tee -a "$LOG" ;;
  esac
done
echo "" | tee -a "$LOG"
echo "[matrix] done $(date -u +%FT%TZ). Aggregate rows:" | tee -a "$LOG"
find results -name rows.csv -newermt "@$(( $(date +%s) - 86400 ))" 2>/dev/null | while read -r f; do
  echo "  $f ($(wc -l <"$f") rows)" | tee -a "$LOG"
done
echo "[matrix] cat results/perf_v2_*/rows.csv > combined for the SSOT rebuild." | tee -a "$LOG"
