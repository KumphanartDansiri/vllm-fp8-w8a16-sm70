#!/usr/bin/env bash
# Reddit-matched throughput measure for gemma-4-31B on vLLM 0.19 tf5, to compare
# apples-to-apples with the public "10x V100" report (Gemma 4 31B = 21.6 tok/s).
#
# Matches his stated methodology EXACTLY:
#   FP16, enforce-eager, max-model-len 8192, FIVE different prompts,
#   256 max tokens with NATURAL EOS (no ignore_eos), and reports both:
#     Avg tok/s    = all 5 requests (first includes warmup overhead)
#     Steady tok/s = requests 2..5 (warmup excluded)
#
#   MODE=eager     ./tools/measure_gemma_reddit.sh   # default — matches reddit
#   MODE=cudagraph ./tools/measure_gemma_reddit.sh   # same method, cudagraph
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${MODE:-eager}"
IMAGE="${IMAGE:-vllm-v100-py312:vllm019-tf5}"
MODEL="${MODEL:-/mnt/models/google/gemma-4-31B-it}"
GPUS="${GPUS:-0,1,2,3}"; TP="${TP:-4}"; PORT="${PORT:-8003}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
MAXTOK="${MAXTOK:-256}"
SERVED=gemma4; CNAME=measure_gemma_reddit
OUT=/tmp/v100_measure; mkdir -p "$OUT"
LOG="$OUT/gemma_reddit_${MODE}_serve.log"; RES="$OUT/RESULT_gemma_reddit_${MODE}.txt"
note(){ echo "[reddit] $*"; }

case "$MODE" in
  eager)     MODE_ARGS=(--enforce-eager) ;;
  cudagraph) MODE_ARGS=() ;;
  *) note "MODE must be eager|cudagraph"; exit 1 ;;
esac

# 5 different prompts (mimic "five prompts per model"); chosen to elicit
# substantial multi-paragraph answers so generation approaches 256 tok.
PROMPTS=(
  "Explain, step by step, how a neural network learns from data."
  "Write a short story about a lighthouse keeper who finds a message in a bottle."
  "Describe the major causes and events of the French Revolution."
  "Compare and contrast renewable and non-renewable energy sources in detail."
  "Explain the rules of chess to a complete beginner, covering each piece."
)

apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1} END{print s+0}')
[[ "$apps" -ne 0 || "$used" -gt 2000 ]] && { note "ABORT: GPUs busy (apps=$apps used=${used}MiB)"; exit 1; }
note "clean-box OK. MODE=$MODE, ${#PROMPTS[@]} prompts, max_tokens=$MAXTOK (natural EOS)."

docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing — build-gemma first"; exit 1; }
docker rm -f "$CNAME" >/dev/null 2>&1 || true

note "serving..."
CONTAINER_NAME="$CNAME" GPUS="$GPUS" PORT="$PORT" IMAGE="$IMAGE" \
    ./docker/run_docker_vllm019_py312.sh serve \
        --model "$MODEL" --served-model-name "$SERVED" \
        --tensor-parallel-size "$TP" --dtype float16 \
        --attention-backend TRITON_ATTN "${MODE_ARGS[@]}" \
        --no-enable-chunked-prefill --max-model-len 8192 --max-num-seqs 8 \
        --gpu-memory-utilization 0.90 --host 0.0.0.0 --port "$PORT" \
    </dev/null >"$LOG" 2>&1 &
lpid=$!

healthy=0; waited=0
while (( waited < HEALTH_TIMEOUT )); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
    kill -0 "$lpid" 2>/dev/null || { note "server process died"; break; }
    sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "  ...loading (${waited}s)"
done
[[ "$healthy" == 1 ]] || { note "never healthy. tail:"; tail -n 25 "$LOG"; docker stop "$CNAME" >/dev/null 2>&1; exit 1; }
note "healthy after ${waited}s. running 5 prompts (NO discard — first carries warmup, per reddit)..."

URL="http://localhost:${PORT}/v1/chat/completions"
declare -a TOKS TIMES
i=0
for p in "${PROMPTS[@]}"; do
    i=$((i+1))
    body=$(python3 -c "import json,sys;print(json.dumps({'model':'$SERVED','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':$MAXTOK,'temperature':0}))" "$p")
    s=$(date +%s.%N); r=$(curl -s "$URL" -H 'Content-Type: application/json' -d "$body"); e=$(date +%s.%N)
    ct=$(echo "$r" | python3 -c "import sys,json;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
    dt=$(python3 -c "print(round($e-$s,3))")
    TOKS+=("${ct:-0}"); TIMES+=("$dt")
    note "  prompt $i: ${ct:-?} tok in ${dt}s"
done

: > "$RES"
python3 - "$MODE" "$TP" "${#PROMPTS[@]}" "${TOKS[@]}" "---" "${TIMES[@]}" <<'PY' | tee -a "$RES"
import sys
mode, tp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rest = sys.argv[4:]
sep = rest.index("---")
toks = [int(x) for x in rest[:sep]]
times = [float(x) for x in rest[sep+1:]]
avg = sum(toks)/sum(times) if sum(times) else 0
steady = sum(toks[1:])/sum(times[1:]) if sum(times[1:]) else 0
print(f"gemma-4-31B | vLLM 0.19 tf5 | {mode} | FP16 TP={tp} | reddit-method (5 prompts, 256 max tok, natural EOS)")
print(f"  per-prompt tokens : {toks}")
print(f"  Avg tok/s    (incl warmup, all 5) : {avg:.2f}")
print(f"  Steady tok/s (excl 1st request)   : {steady:.2f}")
print(f"  (reddit 10xV100 gemma-4 eager: Avg 21.6 / Steady 21.6)")
PY
note "stopping $CNAME..."; docker stop "$CNAME" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
note "result -> $RES"
