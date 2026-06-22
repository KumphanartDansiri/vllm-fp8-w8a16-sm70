#!/usr/bin/env bash
# q27b FP8 pinned to TP4 (= the q27b-fp16 TP) for a PURE precision delta, to sit alongside
# the canonical q27b:fp8 @ TP2 (half-the-GPUs deployment point). Lets us judge whether
# "FP8 at 1/2 the FP16 TP" is worth pursuing fleet-wide.
#
# Mechanics: runs MODEL_KEY=q27b PREC=fp8 TP=4 FA_ARM=0 (no registry edit, no FA arm) and
# RENAMES each output dir perf_v2_q27b_fp8_<eng>_<stamp> -> perf_v2_q27b4_fp8_<eng>_<stamp>
# so it does NOT clobber the canonical TP2 rows in consolidation (which keys on dir prefix).
# The model field inside rows.csv is rewritten q27b->q27b4 for self-consistency.
#
# Runs CONCURRENTLY with the glm47 run on the OTHER 4 GPUs: glm47@TP4 uses GPUs 0-3, this
# uses GPUs 4-7 (GPU_OFFSET=4) on a distinct host port (8060). Disjoint device sets + the
# per-serve clean_guard keep them isolated. LAUNCH in tmux:
#   tmux new -d -s q27b4 'bash tools/perf_v2_q27b4_tp4.sh'
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENGINES="${ENGINES:-021 019}"
export USERS="${USERS:-1 2 4 8}" NRUN="${NRUN:-5}" TTFT_REPS="${TTFT_REPS:-3}"
export MODEL_KEY=q27b PREC=fp8 TP=4 FA_ARM=0
export GPU_OFFSET="${GPU_OFFSET:-4}" PORT="${PORT:-8060}"   # GPUs 4-7, separate host port

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results/perf_v2_q27b4_finish_${STAMP}.log"
echo "[q27b4] q27b FP8 @ TP4 on GPUs ${GPU_OFFSET}-$((GPU_OFFSET+3)) port ${PORT} (concurrent with glm47 on 0-3)" | tee -a "$LOG"

rename_latest(){ # $1=engine ; rename newest perf_v2_q27b_fp8_<eng>_* -> q27b4 prefix
  local eng="$1" nd tag dest
  nd=$(ls -dt results/perf_v2_q27b_fp8_${eng}_* 2>/dev/null | head -1)
  [ -n "$nd" ] || { echo "[q27b4] WARN: no q27b_fp8_${eng} dir to rename" | tee -a "$LOG"; return; }
  tag=${nd##*/perf_v2_q27b_fp8_${eng}_}
  dest=results/perf_v2_q27b4_fp8_${eng}_${tag}
  mv "$nd" "$dest" && sed -i 's/^q27b,/q27b4,/' "$dest/rows.csv" 2>/dev/null
  echo "[q27b4] $nd -> $dest" | tee -a "$LOG"
}

run(){ # $1=engine $2=label $3...=extra env
  local eng="$1" label="$2"; shift 2
  echo "=== q27b4 (q27b fp8 TP4) $eng [$label] ===" | tee -a "$LOG"
  env ENGINE="$eng" "$@" bash tools/perf_v2_cell.sh 2>&1 | tee -a "$LOG"
  echo "=== q27b4 $eng [$label] rc=${PIPESTATUS[0]} ===" | tee -a "$LOG"
  rename_latest "$eng"
}

for eng in $ENGINES; do
  run "$eng" main
  run "$eng" ttftboth   TTFT_BOTH=1
done

echo "Q27B4_FINISH_EXIT=done" | tee -a results/perf_v2_q27b4_finish_LAST.exit
echo "[q27b4] DONE. Fold in: python3 tools/perf_v2_consolidate.py" | tee -a "$LOG"
