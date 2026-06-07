#!/usr/bin/env bash
# Measure gemma-4-31B throughput on vLLM 0.19 (tf5 = transformers-5.x image) on
# V100. Self-contained: serves, warms up, times N requests, tears down, writes
# the tok/s to file. Clean-box-guarded (GPU only — a concurrent aphrodite CPU
# build is fine).
#
#   MODE=eager     ./tools/measure_gemma_vllm019.sh   # --enforce-eager (default)
#   MODE=cudagraph ./tools/measure_gemma_vllm019.sh   # drop --enforce-eager (vLLM
#                                                      # default torch.compile + cudagraph)
#
# Env: GPUS, TP, PORT, MAXTOK (gen tokens/req, default 200), N (timed reqs, 5),
#      IMAGE, MODEL, HEALTH_TIMEOUT.
#
# NOTE: eager is CPU-launch-bound, so a number measured WHILE aphrodite is
# compiling (8 nvcc jobs) will be pessimistic. Re-run on an idle box for a clean
# figure. cudagraph's first run includes a long one-time torch.compile warmup.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${MODE:-eager}"
IMAGE="${IMAGE:-vllm-v100-py312:vllm019-tf5}"
MODEL="${MODEL:-/mnt/models/google/gemma-4-31B-it}"
GPUS="${GPUS:-0,1,2,3}"; TP="${TP:-4}"; PORT="${PORT:-8003}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
MAXTOK="${MAXTOK:-200}"; N="${N:-5}"
SERVED=gemma4; CNAME=measure_gemma_vllm019
OUT=/tmp/v100_measure; mkdir -p "$OUT"
LOG="$OUT/gemma_${MODE}_serve.log"; RES="$OUT/RESULT_gemma_${MODE}.txt"

note(){ echo "[measure] $*"; }

case "$MODE" in
  eager)     MODE_ARGS=(--enforce-eager) ;;
  cudagraph) MODE_ARGS=() ;;   # drop --enforce-eager -> vLLM default compile+cudagraph
  *) note "MODE must be 'eager' or 'cudagraph'"; exit 1 ;;
esac

# clean-box guard: GPU compute only (the aphrodite docker build uses no GPU).
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1} END{print s+0}')
if [[ "$apps" -ne 0 || "$used" -gt 2000 ]]; then
    note "ABORT: GPUs busy (compute_apps=$apps, used=${used}MiB)."
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | head
    exit 1
fi
note "clean-box OK (${used}MiB). MODE=$MODE, TP=$TP on GPUs $GPUS."

docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing — run './tools/smoke_vllm019.sh build-gemma' first"; exit 1; }
docker rm -f "$CNAME" >/dev/null 2>&1 || true

note "serving gemma-4-31B (this loads the model; ~2-3 min cold)..."
CONTAINER_NAME="$CNAME" GPUS="$GPUS" PORT="$PORT" IMAGE="$IMAGE" \
    ./docker/run_docker_vllm019_py312.sh serve \
        --model "$MODEL" --served-model-name "$SERVED" \
        --tensor-parallel-size "$TP" --dtype float16 \
        --attention-backend TRITON_ATTN \
        "${MODE_ARGS[@]}" \
        --no-enable-chunked-prefill \
        --max-model-len 8192 --max-num-seqs 8 \
        --gpu-memory-utilization 0.90 \
        --host 0.0.0.0 --port "$PORT" \
    </dev/null >"$LOG" 2>&1 &
lpid=$!

healthy=0; waited=0
while (( waited < HEALTH_TIMEOUT )); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then healthy=1; break; fi
    if ! kill -0 "$lpid" 2>/dev/null; then note "server process exited before healthy"; break; fi
    sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "  ...loading (${waited}s)"
done
if [[ "$healthy" != 1 ]]; then
    note "FAILED to become healthy in ${HEALTH_TIMEOUT}s. Tail:"; tail -n 30 "$LOG"
    docker stop "$CNAME" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; exit 1
fi
note "healthy after ${waited}s."

URL="http://localhost:${PORT}/v1/chat/completions"
BODY=$(printf '{"model":"%s","messages":[{"role":"user","content":"Write a detailed multi-paragraph essay about the history, geography, and culture of France."}],"max_tokens":%d,"temperature":0,"ignore_eos":true}' "$SERVED" "$MAXTOK")

note "warmup request (also triggers any compile/cudagraph capture)..."
curl -s "$URL" -H 'Content-Type: application/json' -d "$BODY" >/dev/null

tot_t=0; tot_tok=0; ok=1
for i in $(seq 1 "$N"); do
    s=$(date +%s.%N); r=$(curl -s "$URL" -H 'Content-Type: application/json' -d "$BODY"); e=$(date +%s.%N)
    ct=$(echo "$r" | python3 -c "import sys,json;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null) || ok=0
    [[ -z "${ct:-}" ]] && ok=0
    dt=$(python3 -c "print(round($e-$s,2))")
    note "  run $i: ${ct:-?} tok in ${dt}s"
    tot_t=$(python3 -c "print($tot_t+($e-$s))"); tot_tok=$((tot_tok+${ct:-0}))
done

if [[ "$ok" == 1 && "$tot_tok" -gt 0 ]]; then
    python3 -c "print(f'gemma-4-31B | vLLM 0.19 tf5 | $MODE | FP16 TP=$TP : {$tot_tok/$tot_t:.2f} tok/s ({$tot_tok} tok / {$tot_t:.2f}s over $N reqs)')" | tee "$RES"
    echo "(reddit 10xV100 vLLM eager reference: 21.6 tok/s)" | tee -a "$RES"
else
    echo "gemma-4-31B vLLM 0.19 tf5 ($MODE): MEASURE_FAILED — see $LOG" | tee "$RES"
fi
note "stopping $CNAME..."; docker stop "$CNAME" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
note "result -> $RES"
