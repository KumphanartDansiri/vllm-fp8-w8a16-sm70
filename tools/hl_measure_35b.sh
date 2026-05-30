#!/usr/bin/env bash
# Measure 35B-A3B-FP8 — run AFTER tools/hl_serve_35b.sh shows "Application
# startup complete". 1 warmup + 5 timed curls; result -> file + stdout.
set -uo pipefail
PORT="${PORT:-8002}"
MODEL=/mnt/models/Qwen3.6-35B-A3B-FP8
LABEL="35B-A3B-FP8 ns=8 TP=4"
RES=/tmp/v100_bench/RESULT_hl_35b.txt
URL="http://localhost:${PORT}/v1/completions"
BODY=$(printf '{"model":"%s","prompt":"The capital of France is","max_tokens":200,"temperature":0,"ignore_eos":true}' "$MODEL")

if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[hl] no healthy serve on :$PORT — start tools/hl_serve_35b.sh first"; exit 1
fi
echo "[hl] warmup..."; curl -s "$URL" -H 'Content-Type: application/json' -d "$BODY" >/dev/null
tot_t=0; tot_tok=0; ok=1
for i in 1 2 3 4 5; do
    s=$(date +%s.%N); r=$(curl -s "$URL" -H 'Content-Type: application/json' -d "$BODY"); e=$(date +%s.%N)
    ct=$(echo "$r" | python3 -c "import sys,json;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null) || ok=0
    [[ -z "${ct:-}" ]] && ok=0
    echo "  curl $i: ${ct:-?} tok in $(python3 -c "print(round($e-$s,2))")s"
    tot_t=$(python3 -c "print($tot_t+($e-$s))"); tot_tok=$((tot_tok+${ct:-0}))
done
if [[ "$ok" == 1 && "$tot_tok" -gt 0 ]]; then
    python3 -c "print(f'$LABEL: {$tot_tok/$tot_t:.2f} tok/s ({$tot_tok} tok / {$tot_t:.2f}s)')" | tee "$RES"
else
    echo "$LABEL: MEASURE_FAILED" | tee "$RES"
fi
echo "[hl] wrote $RES"
