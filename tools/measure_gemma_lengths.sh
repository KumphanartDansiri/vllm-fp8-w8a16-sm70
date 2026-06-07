#!/usr/bin/env bash
# Realistic decode-rate-vs-CONTEXT-LENGTH sweep for gemma-4-31B on vLLM 0.19 tf5.
#
# WHY: short-generation benchmarks (e.g. 256 tok) report near PEAK decode rate
# and overstate real-world throughput. Decode slows as context grows (each new
# token attends over the whole prior context; KV cache read scales with length).
# Real usage (long documents / legal work) lives at thousands of tokens, where
# the sustained rate is what matters.
#
# METHOD (isolates pure decode rate at a given context depth L):
#   serve ONCE (one torch.compile + cudagraph capture), then for each L:
#     tA = wall(prefill_L + 4 decode);  tB = wall(prefill_L + 132 decode)
#     decode_rate@L = 128 / (tB - tA)        # the shared prefill cancels out
#   Requires --no-enable-prefix-caching, else tB's prefill is cache-served and
#   the subtraction is wrong.
#
#   MODE=cudagraph CONTEXTS="512 2048 4096" ./tools/measure_gemma_lengths.sh
#   MODE=eager     ...                                      # eager comparison
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${MODE:-cudagraph}"
IMAGE="${IMAGE:-vllm-v100-py312:vllm019-tf5}"
MODEL="${MODEL:-/mnt/models/google/gemma-4-31B-it}"
GPUS="${GPUS:-0,1,2,3}"; TP="${TP:-4}"; PORT="${PORT:-8003}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
CONTEXTS="${CONTEXTS:-512 2048 4096}"
SERVED=gemma4; CNAME=measure_gemma_len
OUT=/tmp/v100_measure; mkdir -p "$OUT"
LOG="$OUT/gemma_len_${MODE}_serve.log"; RES="$OUT/RESULT_gemma_len_${MODE}.txt"
note(){ echo "[len] $*"; }

case "$MODE" in
  eager)     MODE_ARGS=(--enforce-eager) ;;
  cudagraph) MODE_ARGS=() ;;
  *) note "MODE must be eager|cudagraph"; exit 1 ;;
esac

apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1} END{print s+0}')
[[ "$apps" -ne 0 || "$used" -gt 2000 ]] && { note "ABORT: GPUs busy (apps=$apps used=${used}MiB)"; exit 1; }
note "clean-box OK. MODE=$MODE contexts=[$CONTEXTS]"

docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing — build-gemma first"; exit 1; }
docker rm -f "$CNAME" >/dev/null 2>&1 || true

note "serving (one compile/capture, then sweep; prefix-caching OFF for clean isolation)..."
CONTAINER_NAME="$CNAME" GPUS="$GPUS" PORT="$PORT" IMAGE="$IMAGE" \
    ./docker/run_docker_vllm019_py312.sh serve \
        --model "$MODEL" --served-model-name "$SERVED" \
        --tensor-parallel-size "$TP" --dtype float16 \
        --attention-backend TRITON_ATTN "${MODE_ARGS[@]}" \
        --no-enable-chunked-prefill --no-enable-prefix-caching \
        --max-model-len 8192 --max-num-seqs 8 \
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
note "healthy after ${waited}s."

URL="http://localhost:${PORT}/v1/chat/completions"
SENT="The quick brown fox jumps over the lazy dog near the quiet riverbank. "

# ctx_request <approx_ctx_tokens> <max_tokens> -> "completion_tokens prompt_tokens elapsed_s"
ctx_request() {
    local ctx="$1" mt="$2" reps prompt body s e r
    reps=$(( ctx / 13 + 1 ))
    prompt=$(python3 -c "print('$SENT'*$reps + 'Now continue the narrative in vivid detail.')")
    body=$(python3 -c "import json,sys;print(json.dumps({'model':'$SERVED','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':$mt,'temperature':0,'ignore_eos':True}))" "$prompt")
    s=$(date +%s.%N); r=$(curl -s "$URL" -H 'Content-Type: application/json' -d "$body"); e=$(date +%s.%N)
    python3 -c "import sys,json;d=json.load(sys.stdin);u=d['usage'];print(u['completion_tokens'],u['prompt_tokens'],round($e-$s,3))" <<<"$r" 2>/dev/null \
        || echo "ERR ERR ERR"
}

note "warmup (also forces any residual capture)..."; ctx_request 256 8 >/dev/null 2>&1

: > "$RES"
echo "gemma-4-31B | vLLM 0.19 tf5 | $MODE | FP16 TP=$TP  —  decode rate vs context depth" | tee -a "$RES"
printf '%-12s  %-14s\n' "ctx_tokens" "decode_tok/s" | tee -a "$RES"
for ctx in $CONTEXTS; do
    read -r cA pA tA <<<"$(ctx_request "$ctx" 4)"
    read -r cB pB tB <<<"$(ctx_request "$ctx" 132)"
    out=$(python3 -c "
try:
    dn=$cB-$cA; dt=$tB-$tA
    print(f'{$pB:<12d}  {dn/dt:<14.2f}') if dt>0 and dn>0 else print(f'{$pB:<12}  ERR(dt={dt:.2f},dn={dn})')
except Exception as ex:
    print(f'{\"$ctx\":<12}  ERR({ex})')
" 2>/dev/null || echo "$ctx  ERR")
    echo "$out" | tee -a "$RES"
done
note "stopping $CNAME..."; docker stop "$CNAME" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
note "result -> $RES"
