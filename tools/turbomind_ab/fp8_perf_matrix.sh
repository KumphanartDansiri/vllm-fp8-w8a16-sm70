#!/usr/bin/env bash
# FP8 perf MATRIX: TurboMind vs our FP8 dequant. Dense + MoE, TP2/TP4, cudagraph (headline).
# SEQUENTIAL — one serve at a time, isolated cache (never parallel during capture). ~45-60 min.
#   bash tools/turbomind_ab/fp8_perf_matrix.sh          # full 8-cell matrix
#   TPS=2 BES="ours turbomind" MODELS_ONLY=q35b ...     # subset via env
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-results/fp8_perf_${STAMP}}"; mkdir -p "$OUT"
CSV="$OUT/perf.csv"
export OUT CSV
MODE="${MODE:-cudagraph}"; CONC="${CONC:-1 2 4 8}"; GENTOK="${GENTOK:-256}"
export MODE CONC GENTOK
TPS="${TPS:-2 4}"; BES="${BES:-ours turbomind}"

declare -a MODELS=(
  "/mnt/models/Qwen/Qwen3.5-27B-FP8|q27b"
  "/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8|q35b"
)
echo "[matrix] OUT=$OUT MODE=$MODE CONC=[$CONC] GENTOK=$GENTOK TPS=[$TPS] BES=[$BES]"
for row in "${MODELS[@]}"; do
  IFS='|' read -r M S <<<"$row"
  [[ -n "${MODELS_ONLY:-}" && "${MODELS_ONLY}" != "$S" ]] && continue
  [[ ! -d "$M" ]] && { echo "[matrix] SKIP $S (missing $M)"; continue; }
  for TP in $TPS; do
    for BE in $BES; do
      echo "===================== $S TP=$TP BACKEND=$BE MODE=$MODE ====================="
      MODEL="$M" SERVED="$S" TP_SIZE="$TP" BACKEND="$BE" \
        bash tools/turbomind_ab/fp8_turbomind_vs_ours_perf.sh \
        || echo "[matrix] cell FAILED ($S TP=$TP $BE) — continuing"
    done
  done
done
echo; echo "===================== RAW CSV: $CSV ====================="
column -t -s, "$CSV" 2>/dev/null || cat "$CSV"
echo; echo "===================== TURBOMIND vs OURS ====================="
python3 tools/turbomind_ab/fp8_perf_render.py "$CSV" || true
echo "[matrix] done -> $OUT"
