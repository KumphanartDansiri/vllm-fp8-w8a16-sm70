#!/usr/bin/env bash
# Measure Qwen3.6-35B-A3B-FP8 (block-FP8 MoE) on vLLM 0.19 via the serve-fp8
# monkey-patch path. Default = cudagraph (mode=0 + FULL_DECODE_ONLY, ns=8 — the
# project's proven V100 FP8 config). Optional Qwen MTP speculative decoding.
# Self-contained: serves, warms up, times N requests, tears down, writes tok/s.
#
#   ./tools/measure_qwen35b_vllm019.sh             # cudagraph
#   MTP=1 ./tools/measure_qwen35b_vllm019.sh       # cudagraph + MTP spec-decode
#   MODE=eager ./tools/measure_qwen35b_vllm019.sh  # eager comparison
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${MODE:-cudagraph}"; MTP="${MTP:-0}"
IMAGE="${IMAGE:-vllm-v100-py312:vllm019}"   # base image (Qwen needs no transformers-5.x)
MODEL="${MODEL:-/mnt/models/Qwen/Qwen3.6-35B-A3B-FP8}"
GPUS="${GPUS:-0,1,2,3}"; TP="${TP:-4}"; PORT="${PORT:-8002}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"; MAXTOK="${MAXTOK:-200}"; N="${N:-5}"
SERVED=qwen35b; CNAME=measure_qwen35b
OUT=/tmp/v100_measure; mkdir -p "$OUT"
tag="${MODE}$([[ $MTP == 1 ]] && echo _mtp)"
LOG="$OUT/qwen35b_${tag}_serve.log"; RES="$OUT/RESULT_qwen35b_${tag}.txt"
note(){ echo "[qwen] $*"; }

case "$MODE" in
  eager)     CG_ARGS=(--enforce-eager) ;;
  cudagraph) CG_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}') ;;
  *) note "MODE must be eager|cudagraph"; exit 1 ;;
esac

apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1} END{print s+0}')
[[ "$apps" -ne 0 || "$used" -gt 2000 ]] && { note "ABORT: GPUs busy (apps=$apps used=${used}MiB)"; exit 1; }
note "clean-box OK. MODE=$MODE MTP=$MTP TP=$TP"

docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }
docker rm -f "$CNAME" >/dev/null 2>&1 || true

note "serving Qwen3.6-35B-A3B-FP8 (serve-fp8; cold start can be long — Triton autotune + capture)..."
ENABLE_QWEN_MTP="$MTP" CONTAINER_NAME="$CNAME" GPUS="$GPUS" PORT="$PORT" IMAGE="$IMAGE" \
    ./docker/run_docker_vllm019_py312.sh serve-fp8 \
        --model "$MODEL" --served-model-name "$SERVED" \
        --quantization fp8 --dtype float16 \
        --attention-backend TRITON_ATTN \
        --tensor-parallel-size "$TP" \
        --max-num-seqs 8 --gpu-memory-utilization 0.85 \
        --max-model-len 8192 --no-enable-chunked-prefill \
        "${CG_ARGS[@]}" \
        --host 0.0.0.0 --port "$PORT" \
    </dev/null >"$LOG" 2>&1 &
lpid=$!

healthy=0; waited=0
while (( waited < HEALTH_TIMEOUT )); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
    kill -0 "$lpid" 2>/dev/null || { note "server process died early"; break; }
    sleep 10; waited=$((waited+10)); (( waited % 120 == 0 )) && note "  ...loading (${waited}s)"
done
if [[ "$healthy" != 1 ]]; then
    note "FAILED to become healthy in ${HEALTH_TIMEOUT}s. Tail:"; tail -n 30 "$LOG"
    echo "qwen35b ($tag): SERVE_FAILED — see $LOG" | tee "$RES"
    docker stop "$CNAME" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; exit 1
fi
note "healthy after ${waited}s."

URL="http://localhost:${PORT}/v1/completions"
BODY=$(printf '{"model":"%s","prompt":"The capital of France is","max_tokens":%d,"temperature":0,"ignore_eos":true}' "$SERVED" "$MAXTOK")
note "warmup (triggers capture/autotune)..."; curl -s "$URL" -H 'Content-Type: application/json' -d "$BODY" >/dev/null

tot_t=0; tot_tok=0; ok=1
for i in $(seq 1 "$N"); do
    s=$(date +%s.%N); r=$(curl -s "$URL" -H 'Content-Type: application/json' -d "$BODY"); e=$(date +%s.%N)
    ct=$(echo "$r" | python3 -c "import sys,json;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null) || ok=0
    [[ -z "${ct:-}" ]] && ok=0
    note "  run $i: ${ct:-?} tok in $(python3 -c "print(round($e-$s,2))")s"
    tot_t=$(python3 -c "print($tot_t+($e-$s))"); tot_tok=$((tot_tok+${ct:-0}))
done

if [[ "$ok" == 1 && "$tot_tok" -gt 0 ]]; then
    python3 -c "print(f'Qwen3.6-35B-A3B-FP8 | vLLM 0.19 serve-fp8 | $MODE${MTP:+ MTP=$MTP} | TP=$TP : {$tot_tok/$tot_t:.2f} tok/s ({$tot_tok} tok / {$tot_t:.2f}s over $N reqs)')" | tee "$RES"
    echo "(vLLM 0.18 baseline ref: 35B-A3B-FP8 cudagraph ns=8 = 52.4 tok/s; gemma-4-31B vLLM0.19 cudagraph = 29.22)" | tee -a "$RES"
else
    echo "Qwen3.6-35B-A3B-FP8 ($tag): MEASURE_FAILED — see $LOG" | tee "$RES"
fi
note "stopping $CNAME..."; docker stop "$CNAME" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
note "result -> $RES"
