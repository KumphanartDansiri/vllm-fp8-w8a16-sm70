#!/usr/bin/env bash
# Measure 122B-A10B-FP8 + MTP — run AFTER tools/hl_serve_122b_mtp.sh shows
# "Application startup complete". 1 warmup + 5 timed curls; result -> file.
# MTP is workload-dependent, so this also tries a 2nd prompt for a range.
set -uo pipefail
PORT="${PORT:-8002}"
MODEL=/mnt/models/Qwen3.5-122B-A10B-FP8
LABEL="122B-A10B-FP8+MTP ns=8 TP=8"
RES=/tmp/v100_bench/RESULT_hl_122b_mtp.txt
URL="http://localhost:${PORT}/v1/completions"

if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[hl] no healthy serve on :$PORT — start tools/hl_serve_122b_mtp.sh first"; exit 1
fi

run_prompt() {  # $1 = prompt text ; echoes "tok/s (..)"
    local body; body=$(printf '{"model":"%s","prompt":"%s","max_tokens":200,"temperature":0,"ignore_eos":true}' "$MODEL" "$1")
    curl -s "$URL" -H 'Content-Type: application/json' -d "$body" >/dev/null   # warmup
    local tot_t=0 tot_tok=0 ok=1 ct s e
    for i in 1 2 3 4 5; do
        s=$(date +%s.%N); r=$(curl -s "$URL" -H 'Content-Type: application/json' -d "$body"); e=$(date +%s.%N)
        ct=$(echo "$r" | python3 -c "import sys,json;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null) || ok=0
        [[ -z "${ct:-}" ]] && ok=0
        tot_t=$(python3 -c "print($tot_t+($e-$s))"); tot_tok=$((tot_tok+${ct:-0}))
    done
    if [[ "$ok" == 1 && "$tot_tok" -gt 0 ]]; then python3 -c "print(f'{$tot_tok/$tot_t:.2f}')"; else echo "FAIL"; fi
}

echo "[hl] prompt A (capital of France)..."; A=$(run_prompt "The capital of France is")
echo "[hl] prompt B (explain tensor parallelism)..."; B=$(run_prompt "Explain tensor parallelism in large language models.")
{ echo "$LABEL"; echo "  prompt A: $A tok/s"; echo "  prompt B: $B tok/s"; echo "  (MTP is workload-dependent; report the range)"; } | tee "$RES"
echo "[hl] wrote $RES"
