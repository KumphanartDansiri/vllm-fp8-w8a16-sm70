#!/usr/bin/env bash
# ======================================================================================
# GLM-4.7-Flash MLA on V100 — OUTPUT CORRECTNESS verification (not just "it runs").
#
# The A/B probes used /v1/completions with a RAW prompt at temp=0 -> an instruct model
# just continues/loops the text (the "quick brown fox" repetition was that, not an MLA
# bug). This tool uses /v1/chat/completions (applies the chat template) and asks
# VERIFIABLE questions, so a correct model produces checkable answers. Pairs the
# numcheck (kernel math cos=1.0) with end-to-end semantic correctness — per the
# "acceptance/coherence alone isn't proof" rule.
#
# Also runs a greedy DETERMINISM check (same prompt x2 -> identical = run-to-run exact)
# and prints full answers for human judgement.
#
# Launch config = the validated usable one: VLLM_V100_MLA_PREFILL=1 +
# VLLM_V100_MLA_DECODE_CUDAGRAPH=1 + cudagraph (FULL_DECODE_ONLY).
#
# Usage:  ./tools/glm47_flash_mla_v100_correctness.sh
# Env:    IMAGE FA_DIR MODEL PORT TP MAXLEN TEMP
# ======================================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
FA_DIR="${FA_DIR:-/home/kumphanartd/flash-attention-v100}"
MODEL="${MODEL:-/mnt/models/zai-org/GLM-4.7-Flash}"
SERVED="glm47flash"
PORT="${PORT:-8021}"
TP="${TP:-4}"
MAXLEN="${MAXLEN:-8192}"
TEMP="${TEMP:-0.0}"
MAXTOK="${MAXTOK:-320}"          # raise for this reasoning model (<think> eats budget)
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
CACHE_TAG="${CACHE_TAG:-021}"
CG_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
SO_GLOB="$FA_DIR/build/lib.linux-x86_64-cpython-312/flash_attn_v100_cuda.*.so"

OUT=/tmp/v100_glm47_mla_correctness
mkdir -p "$OUT"; for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
SUMMARY="$OUT/SUMMARY.txt"; : > "$SUMMARY"
SLOG="$OUT/serve.log"
note() { echo "[glm47-correct] $*"; }
log()  { echo "$*" | tee -a "$SUMMARY"; }

clean_box_guard() {
    local any=0 i used pids
    for i in $(seq 0 $((TP-1))); do
        used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
        pids=$(nvidia-smi --id="$i" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        [[ "${used:-9999}" -gt 2000 || "${pids:-1}" -gt 0 ]] && any=1
    done
    [[ "$any" -eq 0 ]]
}

log "GLM-4.7-Flash MLA OUTPUT CORRECTNESS [V100 TP=$TP, chat-template, temp=$TEMP, cudagraph] — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "ABORT: image $IMAGE missing"; exit 1; }
# shellcheck disable=SC2086
ls $SO_GLOB >/dev/null 2>&1 || { log "ABORT: .so not built"; exit 1; }
clean_box_guard || { log "SKIP: box busy"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 1; }

rm -rf "$OUT/pylib"; mkdir -p "$OUT/pylib"
# shellcheck disable=SC2086
cp $SO_GLOB "$OUT/pylib/" || { log "ABORT: stage .so failed"; exit 1; }
gpus="$(seq -s, 0 $((TP-1)))"

docker rm -f glm47_correct >/dev/null 2>&1 || true
docker run --rm -i --name glm47_correct --gpus "\"device=$gpus\"" \
    -v /mnt/models:/mnt/models:ro -v "$PROJECT_ROOT":/work -w /work \
    -v "$OUT/pylib":/falib:ro -e PYTHONPATH=/work/src:/falib \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
    -p ${PORT}:${PORT} --shm-size=16g \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
    -e VLLM_V100_MLA_PREFILL=1 -e VLLM_V100_MLA_DECODE_CUDAGRAPH=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "$IMAGE" \
    python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
        --tensor-parallel-size "$TP" --dtype float16 \
        --compilation-config "$CG_CONFIG" \
        --max-model-len "$MAXLEN" --max-num-seqs 8 --gpu-memory-utilization 0.90 \
        --host 0.0.0.0 --port "$PORT" \
    </dev/null >"$SLOG" 2>&1 &
lpid=$!

healthy=0 waited=0
while (( waited < HEALTH_TIMEOUT )); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
    kill -0 "$lpid" 2>/dev/null || { log "server exited before healthy"; break; }
    sleep 5; waited=$((waited+5))
done
(( healthy )) || { log "NOT HEALTHY in ${waited}s"; grep -iE "error|ampere|traceback" "$SLOG" | tail -8 | tee -a "$SUMMARY"; docker rm -f glm47_correct >/dev/null 2>&1; exit 1; }
log "server healthy in ${waited}s"

# chat-completions helper: $1=question -> prints assistant text
chat() {
    local q="$1"
    curl -sf -m 300 "http://localhost:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c "import json,sys;print(json.dumps({'model':'$SERVED','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':int('$MAXTOK'),'temperature':float('$TEMP')}))" "$q")" \
        2>>"$OUT/curl.err" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['choices'][0]['message']['content'].strip())" 2>/dev/null
}

# verifiable Q&A: "label|question|expected-substring (case-insensitive; empty=judge manually)"
declare -a CASES=(
  "capital|What is the capital of France? Answer in one word.|paris"
  "arith|Compute 17 * 23. Give only the number.|391"
  "primes|List the first five prime numbers, comma-separated.|2, 3, 5, 7, 11"
  "speed|A train travels 60 km in 1.5 hours. What is its average speed in km/h? Give the number.|40"
  "instr|Reply with exactly one word in all caps: BANANA|BANANA"
  "reason|Alice has 3 apples and buys 2 more, then gives 1 away. How many apples does she have? Give the number.|4"
  "code|Write a Python one-liner using a list comprehension that returns squares of 0..4. Output only the code.|"
  "fact|Who wrote the play 'Romeo and Juliet'? Answer with the name only.|shakespeare"
)

log ""
log "================== OUTPUT CORRECTNESS (chat-template, temp=$TEMP) =================="
pass=0; total=0
warmup=$(chat "Say hi." ); note "warmup -> ${warmup:0:40}"
for c in "${CASES[@]}"; do
    IFS='|' read -r label q expect <<<"$c"
    ans="$(chat "$q")"
    total=$((total+1))
    log ""
    log "[$label] Q: $q"
    log "[$label] A: $ans"
    if [[ -z "$expect" ]]; then
        log "[$label] -> (judge manually)"
    elif grep -qiF "$expect" <<<"$ans"; then
        log "[$label] -> PASS (contains '$expect')"; pass=$((pass+1))
    else
        log "[$label] -> CHECK (expected to contain '$expect')"
    fi
done

# greedy determinism: same prompt twice must be identical (run-to-run "Exact")
if [[ "$TEMP" == "0.0" || "$TEMP" == "0" ]]; then
    log ""
    log "-- greedy determinism (same prompt x2 -> identical?) --"
    d1="$(chat "In one sentence, what is a tensor core?")"
    d2="$(chat "In one sentence, what is a tensor core?")"
    if [[ "$d1" == "$d2" && -n "$d1" ]]; then log "determinism: EXACT (identical)"; else log "determinism: DIFFERS"; log "  run1: $d1"; log "  run2: $d2"; fi
fi

log ""
log "SCORE: $pass/$total auto-verifiable cases passed (others judge manually above)."
log "serve.log: $SLOG"
docker rm -f glm47_correct >/dev/null 2>&1 || true
