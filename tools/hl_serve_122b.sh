#!/usr/bin/env bash
# Load + serve 122B-A10B-FP8 (TP=8, ns=8 cudagraph). Runs in FOREGROUND.
# When you see "Application startup complete", run tools/hl_measure_122b.sh
# in another terminal. Ctrl-C here when done.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | awk '{s+=$1} END{print s+0}')
if [[ "$apps" -ne 0 || "$used" -gt 2000 ]]; then
    echo "[hl] ABORT: GPUs busy (compute_apps=$apps, used=${used}MiB). Wait until idle."
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | head
    exit 1
fi

MODEL=/mnt/models/Qwen3.5-122B-A10B-FP8 PORT=8002 GPUS=0,1,2,3,4,5,6,7 TP_SIZE=8 \
  ./tools/bench_v100.sh serve hl_122b
