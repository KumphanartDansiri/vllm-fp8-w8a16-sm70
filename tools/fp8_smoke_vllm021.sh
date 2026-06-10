#!/usr/bin/env bash
# FP8 W8A16 sm_70 smoke on the STOCK vLLM 0.21 source build
# (image vllm-v100:vllm021-cu126), via the fp8_w8a16_sm70 monkey-patches mounted
# at /work (PYTHONPATH=/work/src), kernel JIT-compiled in-container and cached.
#
# This is the FP8 port validation: load a block-FP8 (DeepSeek-style [128,128])
# model on V100 through the patched Fp8LinearMethod/Fp8MoEMethod and confirm it
# generates coherent text. Eager (--enforce-eager) for correctness isolation.
#
# Order is dense-first (Linear path only) -> MoE -> flagship, so a Linear-path
# bug is caught on the cheap 27B before the slow MoE/TP8 runs:
#   q27b-fp8     Qwen3.6-27B-FP8        dense block-FP8   TP4   (Linear path)
#   q35b-fp8     Qwen3.6-35B-A3B-FP8    MoE   block-FP8   TP4   (MoE path)
#   q122b-fp8    Qwen3.5-122B-A10B-FP8  MoE   block-FP8   TP8   (flagship)
#
# Usage:  ONLY=q27b-fp8 ./tools/fp8_smoke_vllm021.sh    # one model (default: all)
# Env:    IMAGE PORT HEALTH_TIMEOUT MAXTOK GPUMEM  + any VLLM_V100_FP8_* knobs

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
PORT="${PORT:-8021}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3000}"
MAXTOK="${MAXTOK:-200}"
GPUMEM="${GPUMEM:-0.85}"
ONLY="${ONLY:-}"
MODE="${MODE:-eager}"        # eager | cudagraph
NRUN="${NRUN:-3}"            # timed ignore_eos runs for tok/s
TIMETOK="${TIMETOK:-128}"    # tokens per timed run
CG_CONFIG="${CG_CONFIG:-{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}}"
NS="${NS:-8}"               # --max-num-seqs (historical ns=1 bug lever)

OUT=/tmp/v100_fp8_021
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY_${MODE}.txt"
# CACHE_TAG isolates the JIT kernel / triton caches per engine ABI. The 0.21
# image is torch 2.11+cu126; the 0.19 image is a different torch — they must NOT
# share a torch_extensions cache (ABI-incompatible .so). Override for 0.19 runs.
CACHE_TAG="${CACHE_TAG:-021}"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note() { echo "[fp8-021] $*"; }

# label|model|served|tp
MATRIX=(
  "q27b-fp8|/mnt/models/Qwen/Qwen3.6-27B-FP8|q27bfp8|4"
  "q35b-fp8|/mnt/models/Qwen/Qwen3.6-35B-A3B-FP8|q35bfp8|4"
  "q122b-fp8|/mnt/models/Qwen/Qwen3.5-122B-A10B-FP8|q122bfp8|8"
)

clean_box_guard() {
    local apps used
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1} END{print s+0}')
    [[ "$apps" -eq 0 && "$used" -le 2000 ]]
}

run_one() {
    local label="$1" model="$2" served="$3" tp="$4" gpus
    if (( tp >= 8 )); then gpus="0,1,2,3,4,5,6,7"; else gpus="0,1,2,3"; fi
    local cname="fp8021_${label}_${MODE}"
    local slog="$OUT/${label}_${MODE}_serve.log" rfile="$OUT/${label}_${MODE}_response.json" sfile="$OUT/${label}_${MODE}_sample.txt"

    # mode -> serve args. cudagraph = mode:0 (no torch.compile) + FULL_DECODE_ONLY
    # capture; the open question is whether our W8A16 kernel + MoE grouped dispatch
    # are capturable on Volta.
    local MARGS=()
    case "$MODE" in
        eager)     MARGS=(--enforce-eager) ;;
        cudagraph) MARGS=(--compilation-config "$CG_CONFIG") ;;
        *) note "unknown MODE=$MODE"; return 1 ;;
    esac

    [[ -f "$model/config.json" ]] || { echo "$label: SKIP (missing $model)" | tee -a "$SUMMARY"; return; }
    clean_box_guard || { echo "$label: SKIP (box busy)" | tee -a "$SUMMARY"; return 1; }

    note "=== $label : $model (TP=$tp on $gpus) [FP8 W8A16 sm_70 | $MODE] ==="
    docker rm -f "$cname" >/dev/null 2>&1 || true
    docker run --rm -i --name "$cname" --gpus "\"device=$gpus\"" \
        -v /mnt/models:/mnt/models:ro \
        -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
        -p ${PORT}:${PORT} --shm-size=16g \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e VLLM_V100_FP8_MOE_FALLBACK="${VLLM_V100_FP8_MOE_FALLBACK:-1}" \
        -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM="${VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM:-1}" \
        -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS="${VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS:-128}" \
        -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP="${VLLM_V100_FP8_MOE_FAST_ROUTE_PREP:-1}" \
        "$IMAGE" \
        python3 -m fp8_w8a16_sm70.vllm_serve --model "$model" --served-model-name "$served" \
            --tensor-parallel-size "$tp" --dtype float16 "${MARGS[@]}" \
            --max-model-len 4096 --max-num-seqs "$NS" \
            --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill \
            --host 0.0.0.0 --port "$PORT" ${EXTRA:-} \
        </dev/null >"$slog" 2>&1 &
    local lpid=$!

    local healthy=0 waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$lpid" 2>/dev/null || { note "  server exited before healthy"; break; }
        sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "  ...loading $label (${waited}s)"
    done
    if [[ "$healthy" != 1 ]]; then
        echo "$label: FAIL (never healthy in ${HEALTH_TIMEOUT}s) — see $slog" | tee -a "$SUMMARY"
        note "patch banner / first error:"; grep -E "Patches applied|Error|NotImplementedError|Traceback|raise|assert" "$slog" | head -8
        tail -n 20 "$slog"; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
        return 1
    fi
    note "  healthy after ${waited}s. Generating..."

    local url="http://localhost:${PORT}/v1/chat/completions"
    local body
    body=$(python3 -c "import json,sys;print(json.dumps({'model':'$served','messages':[{'role':'user','content':'Write a detailed multi-paragraph essay about the history, geography, and culture of France.'}],'max_tokens':$MAXTOK,'temperature':0}))")
    curl -s "$url" -H 'Content-Type: application/json' -d "$body" >"$rfile" 2>&1
    local verdict
    verdict=$(python3 - "$rfile" "$sfile" <<'PY'
import json, sys, re
rfile, sfile = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(rfile)); text = d["choices"][0]["message"]["content"]
    ntok = d.get("usage", {}).get("completion_tokens", 0); s = text.strip()
    open(sfile, "w").write(text)
    words = s.split()
    rep = (max((words.count(w) for w in set(words)), default=0)/len(words)) if words else 1.0
    bang = (s.count("!")/len(s)) if s else 1.0
    ok = bool(s) and ntok >= 20 and bang < 0.3 and rep < 0.35
    print(("OK" if ok else "BAD") + f"\t{ntok}\t{rep:.2f}\t" + re.sub(r'\s+',' ',s)[:140])
except Exception as e:
    print(f"BAD\t0\t1.00\tparse-error: {e}")
PY
)
    local tag ntok rep snip
    tag=$(printf '%s' "$verdict" | cut -f1); ntok=$(printf '%s' "$verdict" | cut -f2)
    rep=$(printf '%s' "$verdict" | cut -f3); snip=$(printf '%s' "$verdict" | cut -f4-)

    # tok/s: NRUN timed ignore_eos runs of TIMETOK tokens (the headline number,
    # esp. for cudagraph mode).
    local tbody tot_t=0 tot_tok=0 i s_t e_t ct
    tbody=$(python3 -c "import json;print(json.dumps({'model':'$served','messages':[{'role':'user','content':'Continue this story in vivid detail.'}],'max_tokens':$TIMETOK,'temperature':0,'ignore_eos':True}))")
    curl -s "$url" -H 'Content-Type: application/json' -d "$tbody" >/dev/null 2>&1   # warmup / capture
    for i in $(seq 1 "$NRUN"); do
        s_t=$(date +%s.%N); curl -s "$url" -H 'Content-Type: application/json' -d "$tbody" >"$OUT/.t.json" 2>&1; e_t=$(date +%s.%N)
        ct=$(python3 -c "import json;print(json.load(open('$OUT/.t.json'))['usage']['completion_tokens'])" 2>/dev/null || echo 0)
        tot_tok=$((tot_tok+${ct:-0})); tot_t=$(python3 -c "print($tot_t+($e_t-$s_t))")
    done
    local toks; toks=$(python3 -c "print(f'{$tot_tok/$tot_t:.2f}')" 2>/dev/null || echo "n/a")

    # kernel-variant banner (proves our GEMM actually ran)
    local kern; kern=$(grep -oE "kernel variant counts after [0-9]+ calls: [^\"]*" "$slog" | tail -1)
    if [[ "$tag" == "OK" ]]; then
        echo "$label [$MODE]: PASS  (${ntok} tok, rep=$rep, ~${toks} tok/s) [$kern] | \"$snip\"" | tee -a "$SUMMARY"
    else
        echo "$label [$MODE]: FAIL  (tag=$tag tok=$ntok rep=$rep) | \"$snip\" — see $slog" | tee -a "$SUMMARY"
    fi
    note "  stopping $cname..."; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
}

main() {
    : > "$SUMMARY"
    echo "vLLM 0.21.0 sm_70 FP8 W8A16 smoke [MODE=$MODE] — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }
    local row label model served tp
    for row in "${MATRIX[@]}"; do
        IFS='|' read -r label model served tp <<<"$row"
        [[ -n "$ONLY" && "$label" != *"$ONLY"* ]] && continue
        run_one "$label" "$model" "$served" "$tp" || true
    done
    echo; note "==== FP8 SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
}

main "$@"
