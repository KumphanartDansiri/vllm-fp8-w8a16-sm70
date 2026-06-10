#!/usr/bin/env bash
# gemma-4 compatibility + PROVISIONAL perf check on vLLM 0.21 sm_70
# (image vllm-v100:vllm021-cu126). Launch and walk away; results ->
# /tmp/v100_gemma4/SUMMARY.txt, generated text -> *_sample.txt, logs -> *_serve.log.
#
# ⚠️ PERF IS PROVISIONAL: this box is CPU-contended (son's aiagent VSCode +
# possibly a build). Single-stream decode tok/s here carries ±~10% noise — treat
# the numbers as "does it run and roughly how fast", NOT as clean benchmarks.
# The COMPATIBILITY verdict (loads + generates coherently) is contention-immune.
#
# Runner is AUTO-DETECTED per model from config.json quant_method:
#   (none)              -> stock `vllm serve`              [BF16/FP16]
#   fp8 + weight_block_size -> our kernel (python -m fp8_w8a16_sm70.vllm_serve)
#   fp8, no block       -> our module (FAIL-CLOSED: kernel is block-only)
#   compressed-tensors  -> stock `vllm serve`  *** EXPECTED FAIL on sm_70 ***
#       RedHatAI FP8 models use compressed-tensors (per-channel weight + per-token
#       dynamic act). Our patch hooks the `fp8` method, NOT compressed-tensors, so
#       it does not engage; stock compressed-tensors FP8 needs sm_80+ (cutlass/
#       Marlin). This run DOCUMENTS that gap (what a future patch must hook).
#
# Usage:  ./tools/gemma4_compat_vllm021.sh           # all present models
#         ONLY=g26b-a4b-bf16 ./tools/gemma4_compat_vllm021.sh
# Env: IMAGE PORT HEALTH_TIMEOUT MAXTOK GPUMEM TP_OVERRIDE

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
PORT="${PORT:-8021}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
MAXTOK="${MAXTOK:-200}"
GPUMEM="${GPUMEM:-0.85}"
ONLY="${ONLY:-}"

OUT=/tmp/v100_gemma4
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-021-$s"; done
note() { echo "[gemma4] $*"; }

# label|path-or-GLOB|tp   (GLOB: resolved at runtime; skipped if no match yet)
MODELS=(
  "g26b-a4b-bf16|/mnt/models/google/gemma-4-26B-A4B-it|4"
  "g31b-fp8blk|/mnt/models/RedHatAI/gemma-4-31B-it-FP8-block|4"
  "g31b-fp8dyn|/mnt/models/RedHatAI/gemma-4-31B-it-FP8-Dynamic|4"
  "g26b-a4b-fp8dyn|/mnt/models/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic|4"
)

clean_box_guard() {  # GPU only; CPU contention is accepted (perf provisional)
    local apps used
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1} END{print s+0}')
    [[ "$apps" -eq 0 && "$used" -le 2000 ]]
}

# detect_runner <model_dir> -> prints "stock" | "fp8" | "fp8-nonblock" | "ct" ; also sets QDESC
detect_runner() {
    python3 - "$1" <<'PY'
import json, sys, os
d = sys.argv[1]
try:
    c = json.load(open(os.path.join(d, "config.json")))
except Exception as e:
    print("stock\tno-config"); sys.exit(0)
tc = c.get("text_config", c)
q = c.get("quantization_config") or tc.get("quantization_config")
if not q:
    print("stock\tbf16/fp16 (no quant)"); sys.exit(0)
qm = (q.get("quant_method") or "").lower()
wbs = q.get("weight_block_size")
if qm == "fp8":
    if wbs:
        print(f"fp8\tfp8 block={wbs} (our kernel)")
    else:
        print("fp8-nonblock\tfp8 per-tensor/channel (kernel is block-only)")
elif qm == "compressed-tensors":
    # summarize the scheme
    grp = q.get("config_groups", {}) or {}
    strat = "?"
    for g, v in grp.items():
        w = (v.get("weights") or {})
        strat = f"{w.get('strategy')}/dyn-act"
        break
    print(f"ct\tcompressed-tensors {strat} (V100 hook: dequant->FP16)")
else:
    print(f"stock\t{qm or 'unknown'}")
PY
}

run_one() {
    local label="$1" model="$2" tp="$3" gpus runner qdesc
    if (( tp >= 8 )); then gpus="0,1,2,3,4,5,6,7"; else gpus="0,1,2,3"; fi
    local cname="gemma4_${label}" slog="$OUT/${label}_serve.log" rfile="$OUT/${label}_response.json" sfile="$OUT/${label}_sample.txt"

    if [[ ! -f "$model/config.json" ]]; then
        echo "$label: SKIP (not on disk yet: $model)" | tee -a "$SUMMARY"; return; fi
    clean_box_guard || { echo "$label: SKIP (GPU busy)" | tee -a "$SUMMARY"; return 1; }

    IFS=$'\t' read -r runner qdesc < <(detect_runner "$model")
    note "=== $label : $model  [$qdesc] runner=$runner (TP=$tp) ==="

    # build the serve invocation per runner
    local SERVE_BIN=() ENVS=()
    case "$runner" in
        fp8|fp8-nonblock|ct)
            SERVE_BIN=(python3 -m fp8_w8a16_sm70.vllm_serve)
            ENVS=(-v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src
                  -v "$HOME/.cache/vllm-v100-021-torchext:/root/.cache/torch_extensions"
                  -e VLLM_V100_FP8_MOE_FALLBACK=1 -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1
                  -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1 -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=128) ;;
        *)  SERVE_BIN=(vllm serve) ;;   # stock (bf16 and compressed-tensors)
    esac

    docker rm -f "$cname" >/dev/null 2>&1 || true
    docker run --rm -i --name "$cname" --gpus "\"device=$gpus\"" \
        -v /mnt/models:/mnt/models:ro \
        -v "$HOME/.cache/vllm-v100-021-triton:/root/.triton" \
        -v "$HOME/.cache/vllm-v100-021-torch:/root/.cache/torch" \
        -v "$HOME/.cache/vllm-v100-021-inductor:/tmp/torchinductor_root" \
        "${ENVS[@]}" \
        -p ${PORT}:${PORT} --shm-size=16g \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        "$IMAGE" \
        "${SERVE_BIN[@]}" --model "$model" --served-model-name "$label" \
            --tensor-parallel-size "$tp" --dtype float16 --enforce-eager \
            --max-model-len 4096 --max-num-seqs 8 \
            --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill \
            --host 0.0.0.0 --port "$PORT" \
        </dev/null >"$slog" 2>&1 &
    local lpid=$!

    local healthy=0 waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$lpid" 2>/dev/null || { note "  server exited before healthy"; break; }
        sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "  ...loading $label (${waited}s)"
    done
    if [[ "$healthy" != 1 ]]; then
        local reason; reason=$(grep -oE "no kernel image|NotImplementedError[^\"]*|not support[^\"]*|capability[^\"]*|Unknown quantization|ValueError[^\"]*|KeyError[^\"]*|min.*capability[^\"]*" "$slog" 2>/dev/null | head -1)
        echo "$label: FAIL [$qdesc] (never healthy) reason=\"${reason:-see log}\" — $slog" | tee -a "$SUMMARY"
        tail -n 20 "$slog"; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; return 1
    fi
    note "  healthy after ${waited}s. generating..."

    local url="http://localhost:${PORT}/v1/chat/completions" body
    body=$(python3 -c "import json;print(json.dumps({'model':'$label','messages':[{'role':'user','content':'Write a detailed multi-paragraph essay about the history, geography, and culture of France.'}],'max_tokens':$MAXTOK,'temperature':0}))")
    curl -s "$url" -H 'Content-Type: application/json' -d "$body" >"$rfile" 2>&1
    local verdict; verdict=$(python3 - "$rfile" "$sfile" <<'PY'
import json,sys,re
try:
    d=json.load(open(sys.argv[1])); t=d["choices"][0]["message"]["content"]; n=d.get("usage",{}).get("completion_tokens",0)
    s=t.strip(); open(sys.argv[2],"w").write(t); w=s.split()
    rep=(max((w.count(x) for x in set(w)),default=0)/len(w)) if w else 1.0
    ok=bool(s) and n>=20 and (s.count("!")/len(s) if s else 1)<0.3 and rep<0.35
    print(("OK" if ok else "BAD")+f"\t{n}\t{rep:.2f}\t"+re.sub(r'\s+',' ',s)[:130])
except Exception as e: print(f"BAD\t0\t1.0\tparse-error: {e}")
PY
)
    local tag ntok rep snip; tag=$(printf '%s' "$verdict"|cut -f1); ntok=$(printf '%s' "$verdict"|cut -f2)
    rep=$(printf '%s' "$verdict"|cut -f3); snip=$(printf '%s' "$verdict"|cut -f4-)
    # provisional tok/s (single timed run; contended box)
    local s_t e_t ct toks; s_t=$(date +%s.%N)
    curl -s "$url" -H 'Content-Type: application/json' -d "$(python3 -c "import json;print(json.dumps({'model':'$label','messages':[{'role':'user','content':'Continue this story.'}],'max_tokens':128,'temperature':0,'ignore_eos':True}))")" >"$OUT/.t.json" 2>&1; e_t=$(date +%s.%N)
    ct=$(python3 -c "import json;print(json.load(open('$OUT/.t.json'))['usage']['completion_tokens'])" 2>/dev/null||echo 0)
    toks=$(python3 -c "print(f'{$ct/($e_t-$s_t):.1f}')" 2>/dev/null||echo "n/a")
    local kern; kern=$(grep -oE "kernel variant counts after [0-9]+ calls: [^\"]*" "$slog"|tail -1)
    if [[ "$tag" == "OK" ]]; then
        echo "$label: PASS [$qdesc] (${ntok} tok, rep=$rep, ~${toks} tok/s PROVISIONAL) ${kern:+[$kern]} | \"$snip\"" | tee -a "$SUMMARY"
    else
        echo "$label: FAIL [$qdesc] (tag=$tag tok=$ntok) | \"$snip\" — $slog" | tee -a "$SUMMARY"
    fi
    note "  stopping $cname..."; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
}

main() {
    : > "$SUMMARY"
    echo "gemma-4 compat + PROVISIONAL perf on vLLM 0.21 sm_70 [$IMAGE] — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
    echo "(perf provisional: box CPU-contended; compat verdict is reliable)" >> "$SUMMARY"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }
    local row label spec tp model
    for row in "${MODELS[@]}"; do
        IFS='|' read -r label spec tp <<<"$row"
        [[ -n "$ONLY" && "$label" != *"$ONLY"* ]] && continue
        if [[ "$spec" == GLOB:* ]]; then
            model=$(compgen -G "${spec#GLOB:}" 2>/dev/null | head -1)
            [[ -z "$model" ]] && { echo "$label: SKIP (block variant not downloaded yet)" | tee -a "$SUMMARY"; continue; }
        else model="$spec"; fi
        TP_OVERRIDE="${TP_OVERRIDE:-}"; [[ -n "$TP_OVERRIDE" ]] && tp="$TP_OVERRIDE"
        run_one "$label" "$model" "$tp" || true
    done
    echo; note "==== GEMMA4 SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
}
main "$@"
