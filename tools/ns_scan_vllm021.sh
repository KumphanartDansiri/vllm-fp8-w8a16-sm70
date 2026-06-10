#!/usr/bin/env bash
# Overnight max-num-seqs sweep on vLLM 0.21 sm_70 FP8 (image vllm-v100:vllm021-cu126).
#
# WHY: ns=1 turned out FASTER than ns=8 for single-stream decode on 0.21 (the
# 0.18 "ns=8 faster" lore was an artifact of the now-fixed ns=1 crash). This scan
# maps tok/s vs ns ∈ {1,2,4,8} across all FP8 models so we can pick the optimal
# single-stream ns per model AND see whether ns=1 erases the ~14% vs-0.19 gap.
#
# Method: cudagraph (FULL_DECODE_ONLY), --no-async-scheduling (ruled out, held
# constant), 200-token ignore_eos timed decode (matches the 0.19 head-to-head).
# Each cell = a full serve via tools/fp8_smoke_vllm021.sh (which JIT-loads the
# kernel, captures cudagraph, measures, tears down).
#
# Shared-box rule: clean-box guard BEFORE each cell; if the trainer resumes,
# remaining cells are marked SKIP and the scan stops (no collision). Model-major,
# fastest-first ordering so complete per-model data lands before any interruption.
#
# Launch and walk away:  ./tools/ns_scan_vllm021.sh
#   Results table -> /tmp/v100_ns_scan/RESULTS.txt ; per-cell logs alongside.
# Env: NS_LIST="1 2 4 8"  MODELS="q35b-fp8 q122b-fp8 q27b-fp8"  TIMETOK=200

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NS_LIST="${NS_LIST:-1 2 4 8}"
# Fastest-useful first: 35B (TP4 MoE, clearest ns signal) → 122B (TP8 flagship,
# the regression case) → 27B (TP4 dense). Edit to add non-FP8 models if desired.
MODELS="${MODELS:-q35b-fp8 q122b-fp8 q27b-fp8}"
TIMETOK="${TIMETOK:-200}"
FP8_SUMMARY=/tmp/v100_fp8_021/SUMMARY_cudagraph.txt

OUT=/tmp/v100_ns_scan
mkdir -p "$OUT"
RESULTS="$OUT/RESULTS.txt"
note() { echo "[ns-scan] $*"; }

clean_box_guard() {
    local apps used
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1} END{print s+0}')
    [[ "$apps" -eq 0 && "$used" -le 2000 ]]
}

: > "$RESULTS"
echo "vLLM 0.21 sm_70 FP8 — max-num-seqs sweep [cudagraph, async-off, ${TIMETOK}-tok] — $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RESULTS"
echo "baselines: 0.21 122B-FP8 ns=8=23.93 | 0.19 122B-FP8 ns=8=27.74 | 0.21 35B-FP8 ns=1=36.74" | tee -a "$RESULTS"
echo "----------------------------------------------------------------" | tee -a "$RESULTS"

docker image inspect vllm-v100:vllm021-cu126 >/dev/null 2>&1 || { note "image missing"; exit 1; }

busy=0
for m in $MODELS; do
    for ns in $NS_LIST; do
        if ! clean_box_guard; then
            echo "$m ns=$ns | SKIP (box busy — trainer)" | tee -a "$RESULTS"; busy=1; break
        fi
        note "=== $m  ns=$ns ==="
        docker rm -f "fp8021_${m}_cudagraph" >/dev/null 2>&1 || true
        NS="$ns" MODE=cudagraph ONLY="$m" TIMETOK="$TIMETOK" EXTRA="--no-async-scheduling" \
            ./tools/fp8_smoke_vllm021.sh > "$OUT/${m}_ns${ns}.run.log" 2>&1 || true
        # scrape the verdict line the fp8 harness wrote
        line=$(grep -E "^${m} \[cudagraph\]:" "$FP8_SUMMARY" 2>/dev/null | tail -1)
        [[ -z "$line" ]] && line="(no verdict — see $OUT/${m}_ns${ns}.run.log)"
        echo "$m ns=$ns | $line" | tee -a "$RESULTS"
        # preserve the serve log for this cell (variant counts / capture / errors)
        cp "/tmp/v100_fp8_021/${m}_cudagraph_serve.log" "$OUT/${m}_ns${ns}_serve.log" 2>/dev/null || true
    done
    [[ "$busy" == 1 ]] && { note "box busy — stopping scan"; break; }
done

echo "----------------------------------------------------------------" | tee -a "$RESULTS"
echo "==== NS SCAN DONE $(date -u +%H:%M:%SZ) ====" | tee -a "$RESULTS"
note "results -> $RESULTS"