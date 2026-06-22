#!/usr/bin/env bash
# FP8 @ TP2 "half-the-GPUs" indicator for q35b/g31b/g26b: DECODE throughput on HALF the
# fp16 GPU count, at a REDUCED max-model-len (MAXLEN, default 8192) because a full 32k KV
# cache does not fit at TP2 (half the GPUs to shard KV across). long-TTFT is N/A at the
# reduced len (the 24k prompt over-runs it -> graceful nan); decode + short-TTFT are valid.
# q27b is the exception (small enough to keep 32k at TP2). 021-only (g26b:fp8:019 = gemma4.py gap).
#
# Runs on GPUs 4-7 (offsets 4 & 6) so it does NOT collide with a glm47 run on 0-3. Two
# concurrent, then the third. Renames out dirs perf_v2_<m>_fp8_021_* -> perf_v2_<m>2_fp8_021_*.
# LAUNCH:  tmux new -d -s tp2 'bash tools/perf_v2_fp8_tp2_indicator.sh'
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export USERS="${USERS:-1 2 4 8}" NRUN="${NRUN:-5}" TTFT_REPS="${TTFT_REPS:-3}"
export MAXLEN="${MAXLEN:-8192}"          # reduced max-model-len so KV fits at TP2
ENGINES="${ENGINES:-021}"; PHASES="${PHASES:-main}"

rename_latest(){ local m="$1" eng="$2" nd tag dest
  nd=$(ls -dt results/perf_v2_${m}_fp8_${eng}_* 2>/dev/null | head -1)
  [ -n "$nd" ] || { echo "[tp2] WARN no ${m}_fp8_${eng} dir"; return; }
  tag=${nd##*/perf_v2_${m}_fp8_${eng}_}; dest=results/perf_v2_${m}2_fp8_${eng}_${tag}
  mv "$nd" "$dest" && sed -i "s/^${m},/${m}2,/" "$dest/rows.csv" 2>/dev/null
  echo "[tp2] $nd -> $dest"
}

run_model(){ # $1=model $2=gpu_offset ; own log
  local m="$1" off="$2"
  local log="results/perf_v2_${m}2_finish_$(date +%Y%m%d_%H%M%S).log"
  echo "[tp2] $m fp8 TP2 maxlen=$MAXLEN GPUs ${off}-$((off+1)) port 80$((50+off))" | tee -a "$log"
  for eng in $ENGINES; do for phase in $PHASES; do
    local ttft=0; [[ "$phase" == ttftboth ]] && ttft=1
    echo "=== ${m}2 fp8 $eng [$phase] ===" | tee -a "$log"
    env MODEL_KEY="$m" PREC=fp8 TP=2 FA_ARM=0 GPU_OFFSET="$off" PORT="80$((50+off))" ENGINE="$eng" TTFT_BOTH="$ttft" \
        bash tools/perf_v2_cell.sh 2>&1 | tee -a "$log"
    echo "=== ${m}2 fp8 $eng [$phase] rc=${PIPESTATUS[0]} ===" | tee -a "$log"
    rename_latest "$m" "$eng" | tee -a "$log"
  done; done
  echo "[tp2] $m DONE" | tee -a "$log"
}

run_model q35b 4 &      # GPUs 4-5
run_model g31b 6 &      # GPUs 6-7
wait
run_model g26b 4        # GPUs 4-5 (freed)
echo "TP2_INDICATOR_EXIT=done" > results/perf_v2_tp2_indicator_LAST.exit
echo "[tp2] ALL DONE -> python3 tools/perf_v2_consolidate.py"
