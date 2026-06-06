#!/usr/bin/env bash
# Self-contained sm_70 smoke test for the vLLM 0.19 baseline. Launch it and
# walk away — it captures everything to /tmp/v100_smoke019/ and prints a
# PASS/FAIL summary. Respects the shared-box rule: a clean-box guard refuses to
# run if the GPUs are already busy (e.g. a training job), and each track stops
# its server before the next starts.
#
# Two independent things get validated on the 0.19 source build:
#   Track A (gemma) : stock `vllm serve` loads google/gemma-4-31B-it and
#                     generates coherent text  ->  proves 0.19 BUILDS + RUNS on
#                     sm_70 and the NEW model is supported. (The actual goal.)
#   Track B (fp8)   : `serve-fp8` loads Qwen3.6-35B-A3B-FP8 (block-FP8 MoE)
#                     through the fp8_w8a16_sm70 monkey-patches  ->  proves the
#                     0.18-era FP8 W8A16 patches PORT to 0.19. Also greps the
#                     server log for residual-risk #1 (an unexpected
#                     get_fused_moe_quant_config / NotImplementedError path).
#
# Smokes run with --enforce-eager on purpose: this checks CORRECTNESS (does it
# load + generate) on the new baseline, decoupled from the cudagraph/ns=8 perf
# path. Perf re-measurement is a separate follow-up once these pass.
#
# Usage:
#   ./tools/smoke_vllm019.sh build        # one-time sm_70 source build (~30-90 min)
#   ./tools/smoke_vllm019.sh qwen         # load-verify ALL 6 Qwen 3.5/3.6 models
#   ./tools/smoke_vllm019.sh fp8          # quick single-model FP8 check (35B-A3B)
#   ./tools/smoke_vllm019.sh all          # build-if-needed + full Qwen sweep
#   ./tools/smoke_vllm019.sh build-gemma  # build the transformers-5.x variant image
#   ./tools/smoke_vllm019.sh gemma        # load-verify Gemma 4 (needs build-gemma first)
#
# Env:
#   GPUS=0,1,2,3   devices to use (default 0,1,2,3 — TP=4)
#   TP=4           tensor-parallel size (default 4)
#   PORT=8003      host port
#   HEALTH_TIMEOUT=2400  seconds to wait for "server healthy" (cold Triton
#                        autotune on first run can take 10-20 min)
#   GEMMA_MODEL / FP8_MODEL  override model paths

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LAUNCHER=./docker/run_docker_vllm019_py312.sh
IMAGE="${IMAGE:-vllm-v100-py312:vllm019}"
# Gemma 4 needs transformers 5.x (its checkpoint is transformers_version
# 5.5.0.dev0), which violates vLLM 0.19's `transformers<5` pin. We build a
# SEPARATE image for it so the FP8-validated default image is never disturbed.
GEMMA_IMAGE="${GEMMA_IMAGE:-vllm-v100-py312:vllm019-tf5}"
GEMMA_TRANSFORMERS="${GEMMA_TRANSFORMERS:-5.5.4}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
PORT="${PORT:-8003}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
GEMMA_MODEL="${GEMMA_MODEL:-/mnt/models/google/gemma-4-31B-it}"
FP8_MODEL="${FP8_MODEL:-/mnt/models/Qwen/Qwen3.6-35B-A3B-FP8}"

OUT=/tmp/v100_smoke019
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"

note() { echo "[smoke019] $*"; }

clean_box_guard() {
    local apps used
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1} END{print s+0}')
    if [[ "$apps" -ne 0 || "$used" -gt 2000 ]]; then
        note "ABORT: GPUs busy (compute_apps=$apps, used=${used}MiB). Wait until idle."
        nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | head
        exit 1
    fi
    note "clean-box guard OK (no compute apps, ${used}MiB used)."
}

image_exists() { docker image inspect "$IMAGE" >/dev/null 2>&1; }

do_build() {
    if image_exists; then
        note "image $IMAGE already present — skipping build (delete it to force a rebuild)."
        return 0
    fi
    note "building $IMAGE from source (slow, one-time)..."
    if ! "$LAUNCHER" build 2>&1 | tee "$OUT/build.log"; then
        note "BUILD FAILED — see $OUT/build.log"
        echo "BUILD: FAILED" >> "$SUMMARY"
        exit 1
    fi
    note "build OK."
}

# Build the Gemma variant image: same sm_70 vLLM 0.19 source build, but with
# transformers overridden to 5.x (unsupported by vLLM's pin — Gemma 4 needs it).
do_build_gemma() {
    if docker image inspect "$GEMMA_IMAGE" >/dev/null 2>&1; then
        note "gemma image $GEMMA_IMAGE already present — skipping build."
        return 0
    fi
    note "building $GEMMA_IMAGE (transformers=$GEMMA_TRANSFORMERS, UNSUPPORTED override)..."
    if ! IMAGE="$GEMMA_IMAGE" TRANSFORMERS_VERSION="$GEMMA_TRANSFORMERS" \
            "$LAUNCHER" build 2>&1 | tee "$OUT/build_gemma.log"; then
        note "GEMMA BUILD FAILED — see $OUT/build_gemma.log"
        echo "build-gemma: FAILED" >> "$SUMMARY"
        exit 1
    fi
    note "gemma build OK."
}

# Qwen 3.5/3.6 family load-verification matrix. These archs (qwen3_5,
# qwen3_5_moe) are in vLLM 0.19's bundled _CONFIG_REGISTRY, so they load with
# the supported (pinned) transformers — unaffected by the Gemma/transformers-5.x
# issue. FP8 rows go through the monkey-patches (serve-fp8); the FP16/Int4 rows
# are stock serve. quant_method is auto-detected from config.json, so stock rows
# need no --quantization flag. Row fields: label|mode|model|served|tp|extra-args
QWEN_MATRIX=(
  "35b-a3b-fp8|serve-fp8|/mnt/models/Qwen/Qwen3.6-35B-A3B-FP8|q35bfp8|4|--quantization fp8 --attention-backend TRITON_ATTN"
  "35b-a3b-fp16|serve|/mnt/models/Qwen/Qwen3.6-35B-A3B|q35bf16|4|--attention-backend TRITON_ATTN"
  "27b-fp8|serve-fp8|/mnt/models/Qwen/Qwen3.6-27B-FP8|q27bfp8|4|--quantization fp8 --attention-backend TRITON_ATTN"
  "27b-fp16|serve|/mnt/models/Qwen/Qwen3.6-27B|q27bf16|4|--attention-backend TRITON_ATTN"
  "122b-a10b-fp8|serve-fp8|/mnt/models/Qwen/Qwen3.5-122B-A10B-FP8|q122bfp8|8|--quantization fp8 --attention-backend TRITON_ATTN"
  "122b-a10b-int4|serve|/mnt/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4|q122bint4|8|--attention-backend TRITON_ATTN"
)

run_qwen_matrix() {
    local row label mode model served tp extra gpus
    for row in "${QWEN_MATRIX[@]}"; do
        IFS='|' read -r label mode model served tp extra <<<"$row"
        if [[ ! -f "$model/config.json" ]]; then
            note "skip $label — no config.json at $model"
            echo "$label: SKIP (model missing)" | tee -a "$SUMMARY"; continue
        fi
        if (( tp >= 8 )); then gpus="0,1,2,3,4,5,6,7"; else gpus="0,1,2,3"; fi
        # word-split $extra deliberately into separate serve flags.
        TP="$tp" GPUS="$gpus" run_track "$label" "$mode" "$model" "$served" completions $extra
    done
}

# run_track <label> <mode> <model> <served_name> <endpoint> <extra serve args...>
# Launches the server in the background, waits for health, sends one request,
# validates the generated text, stops the server, and appends PASS/FAIL.
run_track() {
    local label="$1" mode="$2" model="$3" served="$4" endpoint="$5"; shift 5
    local cname="smoke019_${label}"
    local slog="$OUT/${label}_serve.log"
    local rfile="$OUT/${label}_response.json"

    note "=== Track '$label' : $mode $model (TP=$TP on GPUs $GPUS) ==="
    docker rm -f "$cname" >/dev/null 2>&1 || true

    CONTAINER_NAME="$cname" GPUS="$GPUS" PORT="$PORT" IMAGE="$IMAGE" \
        "$LAUNCHER" "$mode" \
            --model "$model" \
            --served-model-name "$served" \
            --tensor-parallel-size "$TP" \
            --dtype float16 \
            --enforce-eager \
            --max-model-len 4096 \
            --max-num-seqs 8 \
            --gpu-memory-utilization 0.85 \
            --no-enable-chunked-prefill \
            --host 0.0.0.0 --port "$PORT" \
            "$@" \
        </dev/null >"$slog" 2>&1 &
    local lpid=$!

    # Wait for health (or early server death).
    local healthy=0 waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then healthy=1; break; fi
        if ! kill -0 "$lpid" 2>/dev/null; then note "server process exited before healthy"; break; fi
        sleep 10; waited=$((waited+10))
        (( waited % 60 == 0 )) && note "  ...waiting for $label health (${waited}s)"
    done

    if [[ "$healthy" != 1 ]]; then
        note "Track '$label' FAILED to become healthy in ${HEALTH_TIMEOUT}s. Tail of log:"
        tail -n 30 "$slog"
        echo "$label: FAILED (server never healthy) — see $slog" | tee -a "$SUMMARY"
        docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
        return 1
    fi
    note "Track '$label' healthy after ${waited}s. Sending request..."

    # Build request body for the right endpoint.
    local url body
    if [[ "$endpoint" == "chat" ]]; then
        url="http://localhost:${PORT}/v1/chat/completions"
        body=$(printf '{"model":"%s","messages":[{"role":"user","content":"In one sentence, what is the capital of France?"}],"max_tokens":64,"temperature":0}' "$served")
    else
        url="http://localhost:${PORT}/v1/completions"
        body=$(printf '{"model":"%s","prompt":"The capital of France is","max_tokens":64,"temperature":0}' "$served")
    fi
    curl -s "$url" -H 'Content-Type: application/json' -d "$body" >"$rfile" 2>&1

    # Validate: must have generated >=5 tokens of non-empty, non-"!"-spam text.
    local verdict
    verdict=$(python3 - "$rfile" "$endpoint" <<'PY'
import json, sys
path, endpoint = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
    if endpoint == "chat":
        text = d["choices"][0]["message"]["content"]
    else:
        text = d["choices"][0]["text"]
    ntok = d.get("usage", {}).get("completion_tokens", 0)
    stripped = text.strip()
    bang_ratio = (stripped.count("!") / len(stripped)) if stripped else 1.0
    ok = bool(stripped) and ntok >= 5 and bang_ratio < 0.5
    snippet = stripped.replace(chr(10), " ")[:120]
    print(("OK" if ok else "BAD") + f"\t{ntok}\t{snippet}")
except Exception as e:
    print(f"BAD\t0\tparse-error: {e}")
PY
)
    local tag ntok snip
    tag=$(printf '%s' "$verdict" | cut -f1)
    ntok=$(printf '%s' "$verdict" | cut -f2)
    snip=$(printf '%s' "$verdict" | cut -f3-)

    # FP8 track: extra log assertions — patch banner present, no fallback fault.
    local extra=""
    if [[ "$mode" == "serve-fp8" ]]; then
        grep -q "serve_fp8_v100" "$slog" && grep -q "Patches applied" "$slog" \
            && extra="patches=OK" || extra="patches=MISSING"
        if grep -Eq "get_fused_moe_quant_config|NotImplementedError: serve_fp8_v100" "$slog"; then
            extra="$extra residual-risk#1=HIT"
        else
            extra="$extra residual-risk#1=clear"
        fi
    fi

    if [[ "$tag" == "OK" && ( "$mode" != "serve-fp8" || "$extra" == "patches=OK"* ) ]]; then
        note "Track '$label' PASS (${ntok} tok). ${extra}"
        echo "$label: PASS  (${ntok} tok) ${extra} | \"$snip\"" | tee -a "$SUMMARY"
    else
        note "Track '$label' FAIL (tag=$tag tok=$ntok). ${extra}"
        echo "$label: FAIL  (tag=$tag tok=$ntok) ${extra} | \"$snip\" — see $slog / $rfile" | tee -a "$SUMMARY"
    fi

    note "stopping $cname..."
    docker stop "$cname" >/dev/null 2>&1 || true
    wait "$lpid" 2>/dev/null || true
}

main() {
    local cmd="${1:-all}"
    : > "$SUMMARY"
    echo "vLLM 0.19 sm_70 smoke — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"

    case "$cmd" in
        build)       do_build ;;
        build-gemma) do_build_gemma ;;
        gemma) clean_box_guard
               docker image inspect "$GEMMA_IMAGE" >/dev/null 2>&1 || {
                   note "gemma image $GEMMA_IMAGE missing — run '$0 build-gemma' first"; exit 1; }
               IMAGE="$GEMMA_IMAGE" run_track gemma serve "$GEMMA_MODEL" gemma4 chat ;;
        fp8)   clean_box_guard; image_exists || { note "image $IMAGE missing — run '$0 build' first"; exit 1; }
               run_track fp8 serve-fp8 "$FP8_MODEL" qwen35bfp8 completions \
                   --quantization fp8 --attention-backend TRITON_ATTN ;;
        # Full Qwen 3.5/3.6 family load-confidence sweep (6 models, sequential).
        qwen)  clean_box_guard; image_exists || { note "image $IMAGE missing — run '$0 build' first"; exit 1; }
               run_qwen_matrix ;;
        # 'all' = build-if-needed + the full Qwen family sweep. Gemma is NOT
        # included because it needs the unsupported transformers-5.x variant —
        # run it explicitly via 'build-gemma' + 'gemma'.
        all)   clean_box_guard; do_build; run_qwen_matrix ;;
        *)     note "usage: $0 {build|all|qwen|fp8|build-gemma|gemma}"; exit 1 ;;
    esac

    echo; note "==== SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
}

main "$@"
