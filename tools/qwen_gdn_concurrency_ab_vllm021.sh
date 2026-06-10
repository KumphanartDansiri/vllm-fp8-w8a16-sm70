#!/usr/bin/env bash
# Qwen3.6-27B-FP8 GDN concurrency A/B.
#
# Purpose: after Gemma dense + Gemma MoE both scaled under concurrency on the
# same coalesced GEMV stack, Qwen's C=4 cliff is narrowed to GatedDeltaNet.
# This script keeps the coalesced M<=8 kernel ON and sweeps the Qwen-only
# packed recurrent GDN decode flag, plus eager/profile variants for attribution.
#
# Usage:
#   ./tools/qwen_gdn_concurrency_ab_vllm021.sh
#   ONLY=packed_cg ./tools/qwen_gdn_concurrency_ab_vllm021.sh
#
# Outputs:
#   /tmp/v100_qwen_gdn_ab/SUMMARY.txt
#
# Env:
#   IMAGE MODEL GPUS TP PORT MAXTOK CONC CONFIGS ONLY HEALTH_TIMEOUT

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
MODEL="${MODEL:-/mnt/models/Qwen/Qwen3.6-27B-FP8}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
PORT="${PORT:-8028}"
MAXTOK="${MAXTOK:-128}"
CONC="${CONC:-1 2 4}"
MAXLEN="${MAXLEN:-4096}"
GPUMEM="${GPUMEM:-0.80}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
ONLY="${ONLY:-}"
CACHE_TAG="${CACHE_TAG:-021}"
SERVED="${SERVED:-qwen-gdn-ab}"
OUT="${OUT:-/tmp/v100_qwen_gdn_ab}"

# label | packed_recurrent_decode | enforce_eager | breakdown_profile
CONFIGS_DEFAULT=(
  "packed_cg|1|0|0"
  "unpacked_cg|0|0|0"
  "packed_eager_profile|1|1|1"
  "unpacked_eager_profile|0|1|1"
)

IFS=' ' read -r -a CONFIGS_OVERRIDE <<<"${CONFIGS:-}"
if [[ "${#CONFIGS_OVERRIDE[@]}" -gt 0 && -n "${CONFIGS_OVERRIDE[0]}" ]]; then
    CONFIGS_TO_RUN=("${CONFIGS_OVERRIDE[@]}")
else
    CONFIGS_TO_RUN=("${CONFIGS_DEFAULT[@]}")
fi

mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
for s in torchext triton torch inductor; do
    mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"
done

note() { echo "[qwen-gdn-ab] $*"; }

clean_gpu_guard() {
    local gpu used pids busy=0
    IFS=',' read -ra ids <<<"$GPUS"
    for gpu in "${ids[@]}"; do
        used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{print $1+0}')
        pids=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '/^[0-9]+$/ {n++} END{print n+0}')
        if [[ "${used:-9999}" -gt 2000 || "${pids:-1}" -gt 0 ]]; then
            echo "[qwen-gdn-ab] GPU $gpu busy: used=${used:-?}MiB pids=${pids:-?}" >&2
            busy=1
        fi
    done
    [[ "$busy" -eq 0 ]]
}

run_config() {
    local label="$1" packed="$2" eager="$3" profile="$4"
    local cname="qwen_gdn_${label}" slog="$OUT/${label}_serve.log"
    local compile_args=()

    [[ -f "$MODEL/config.json" ]] || {
        echo "$label: SKIP missing model $MODEL" | tee -a "$SUMMARY"
        return 1
    }
    clean_gpu_guard || {
        echo "$label: SKIP GPUs busy: $GPUS" | tee -a "$SUMMARY"
        return 1
    }

    docker rm -f "$cname" >/dev/null 2>&1 || true
    note "=== $label packed=$packed eager=$eager profile=$profile gpus=$GPUS port=$PORT ==="

    if [[ "$eager" == "1" ]]; then
        compile_args=(--enforce-eager)
    else
        compile_args=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
    fi

    docker run --rm -i --name "$cname" --gpus "\"device=$GPUS\"" \
        -v /mnt/models:/mnt/models:ro \
        -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
        -p "${PORT}:${PORT}" --shm-size=16g \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
        -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE="$packed" \
        -e VLLM_V100_FP8_COALESCED_GEMV=1 \
        -e VLLM_V100_FP8_COALESCED_UNROLL=4 \
        -e VLLM_V100_FP8_COALESCED_M_UNROLL=8 \
        -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8 \
        -e VLLM_V100_FP8_DECODE_BREAKDOWN="$profile" \
        -e VLLM_V100_FP8_DECODE_BREAKDOWN_GDN_SUBS=1 \
        -e VLLM_V100_FP8_DECODE_BREAKDOWN_MOE_SUBS=1 \
        -e VLLM_V100_FP8_MOE_OTHER_PROFILE="$profile" \
        -e VLLM_V100_FP8_ROW_PARALLEL_AR_PROFILE="$profile" \
        "$IMAGE" \
        python3 -m fp8_w8a16_sm70.vllm_serve \
            --model "$MODEL" --served-model-name "$SERVED" \
            --tensor-parallel-size "$TP" --dtype float16 --quantization fp8 \
            "${compile_args[@]}" \
            --attention-backend TRITON_ATTN \
            --max-model-len "$MAXLEN" --max-num-seqs 8 \
            --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill \
            --host 0.0.0.0 --port "$PORT" \
        </dev/null >"$slog" 2>&1 &
    local spid=$!

    local healthy=0 waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$spid" 2>/dev/null || break
        sleep 10
        waited=$((waited + 10))
        (( waited % 60 == 0 )) && note "  ...loading $label (${waited}s)"
    done
    if [[ "$healthy" != "1" ]]; then
        echo "$label: FAIL not healthy after ${waited}s; see $slog" | tee -a "$SUMMARY"
        grep -nE "Traceback|Error|CUDA|out of memory|assert" "$slog" | tail -20 | tee -a "$SUMMARY"
        docker stop "$cname" >/dev/null 2>&1 || true
        wait "$spid" 2>/dev/null || true
        return 1
    fi

    note "  healthy after ${waited}s; warmup"
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c "import json; print(json.dumps({'model':'$SERVED','messages':[{'role':'user','content':'Say hi.'}],'max_tokens':16,'temperature':0,'stream':False}))")" \
        >/dev/null 2>&1 || true

    python3 - "$PORT" "$SERVED" "$MAXTOK" "$label" "$CONC" <<'PY' | tee -a "$SUMMARY"
import json, sys, threading, time, urllib.request

port, served, maxtok, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
concs = [int(x) for x in sys.argv[5].split()]
topics = ["France", "Japan", "Brazil", "Egypt", "Canada", "India", "Norway", "Kenya"]

def stream_one(i, out):
    body = json.dumps({
        "model": served,
        "stream": True,
        "max_tokens": maxtok,
        "temperature": 0,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
        "messages": [{
            "role": "user",
            "content": (
                "Write a detailed technical explanation of GatedDeltaNet, "
                f"linear attention, and serving concurrency using {topics[i % len(topics)]} "
                "as the example topic."
            ),
        }],
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    first = last = None
    chunks = 0
    usage_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens"):
                    usage_tokens = int(usage["completion_tokens"])
                choices = obj.get("choices") or []
                delta = choices[0].get("delta", {}).get("content") if choices else None
                if delta:
                    now = time.time()
                    if first is None:
                        first = now
                    last = now
                    chunks += 1
    except Exception:
        pass
    out[i] = (t0, first, last, usage_tokens or chunks)

for C in concs:
    res = [None] * C
    threads = []
    wall0 = time.time()
    for i in range(C):
        t = threading.Thread(target=stream_one, args=(i, res))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall1 = time.time()
    ok = [r for r in res if r and r[1] and r[2] and r[3] > 1]
    total_tokens = sum(r[3] for r in ok)
    aggregate = total_tokens / (wall1 - wall0) if wall1 > wall0 else float("nan")
    per_stream = [(r[3] - 1) / (r[2] - r[1]) for r in ok if r[2] > r[1]]
    ttft = [r[1] - r[0] for r in ok]
    mean_stream = sum(per_stream) / len(per_stream) if per_stream else float("nan")
    mean_ttft = sum(ttft) / len(ttft) if ttft else float("nan")
    print(
        f"{label}: C={C} aggregate={aggregate:.2f} tok/s "
        f"per_stream_decode={mean_stream:.2f} tok/s "
        f"ttft={mean_ttft:.2f}s ok={len(ok)}/{C}"
    )
PY

    {
        echo ""
        echo "---- $label serve extract ----"
        grep -E "DECODE-BREAKDOWN|Qwen3NextGatedDeltaNet|gdn_|row_parallel_ar|Coalesced GEMV|VLLM_ENABLE_FLA_PACKED|packed recurrent|Throughput:" "$slog" | tail -80 || true
    } | tee -a "$SUMMARY"

    note "  stopping $label"
    docker stop "$cname" >/dev/null 2>&1 || true
    wait "$spid" 2>/dev/null || true
}

main() {
    : > "$SUMMARY"
    {
        echo "Qwen GDN concurrency A/B $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "model=$MODEL image=$IMAGE gpus=$GPUS tp=$TP maxtok=$MAXTOK conc=[$CONC]"
        echo "coalesced: M_MAX=8 unroll=4 m_unroll=8"
        echo ""
    } >> "$SUMMARY"

    docker image inspect "$IMAGE" >/dev/null 2>&1 || {
        note "missing docker image $IMAGE"
        exit 1
    }

    local row label packed eager profile
    for row in "${CONFIGS_TO_RUN[@]}"; do
        IFS='|' read -r label packed eager profile <<<"$row"
        [[ -n "$ONLY" && "$label" != *"$ONLY"* ]] && continue
        run_config "$label" "$packed" "$eager" "$profile" || true
        echo "" | tee -a "$SUMMARY"
    done
    note "==== SUMMARY: $SUMMARY ===="
    cat "$SUMMARY"
}

main "$@"
