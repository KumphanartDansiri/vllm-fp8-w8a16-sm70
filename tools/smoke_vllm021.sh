#!/usr/bin/env bash
# Overnight sm_70 GATE-4 smoke for the STOCK vLLM 0.21 source build
# (image vllm-v100:vllm021-cu126, built via docker/Dockerfile.vllm021_cu126).
#
# Launch it and walk away. It loads each model on the V100s, generates real
# text, eyeball-saves that text (coherence — NOT just a tok/s number, per the
# "acceptance alone isn't proof" rule), records an eager tok/s baseline, and
# prints a PASS/FAIL summary to /tmp/v100_smoke021/SUMMARY.txt.
#
# WHAT THIS PROVES (gate 4): stock vLLM 0.21 — with NO FP8 patches, NO arch
# patch (cu126/CUDA-12.6 base already lists sm_70), NO separate transformers
# image (0.21 bundles transformers 5.10.2, which natively defines gemma4 /
# qwen3_5 / qwen3_5_moe) — loads and correctly generates for four models on
# Tesla V100. FP8 W8A16 is the SEPARATE next step (tomorrow), deliberately
# excluded here: every model below is FP16 or GPTQ-Int4.
#
# Models (sequential, fastest->slowest so a brief clean-box window still yields
# the most results; the big TP8 Int4 flagship is last):
#   q27b-fp16     Qwen3.6-27B            dense  FP16  TP4
#   q35b-a3b-fp16 Qwen3.6-35B-A3B        MoE    FP16  TP4
#   gemma4-31b    google/gemma-4-31B-it  dense  FP16  TP4   (the headline gate-4 arch)
#   q122b-int4    Qwen3.5-122B-A10B-GPTQ-Int4  MoE GPTQ-Int4 TP8  (flagship via Int4)
#
# Eager on purpose: --enforce-eager isolates CORRECTNESS (does it load + generate
# coherently on the new baseline) from the cudagraph/ns=8 perf path. The tok/s
# here is therefore an EAGER baseline, not a headline number — headline perf is a
# separate cudagraph measure once these pass.
#
# Shared-box rule: a clean-box guard runs BEFORE EACH model and refuses to
# collide with the training job. If the box goes busy mid-run, remaining models
# are marked SKIPPED (box busy) rather than crashing into the trainer.
#
# Usage:   ./tools/smoke_vllm021.sh            # run all four
#          ONLY=gemma4-31b ./tools/smoke_vllm021.sh   # just one (substring match)
# Env:     IMAGE, PORT, HEALTH_TIMEOUT, MAXTOK, NRUN, GPUMEM  (see defaults below)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
PORT="${PORT:-8021}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"   # cold triton autotune on first model can be slow
MAXTOK="${MAXTOK:-200}"                     # coherence sample length
NRUN="${NRUN:-2}"                           # timed ignore_eos runs for tok/s
TIMETOK="${TIMETOK:-128}"                   # tokens per timed run
GPUMEM="${GPUMEM:-0.85}"
ONLY="${ONLY:-}"

OUT=/tmp/v100_smoke021
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
# triton/torch caches persist across models so only the FIRST pays autotune cost.
for s in triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-021-$s"; done

note() { echo "[smoke021] $*"; }

# label|model|served|tp
MATRIX=(
  "q27b-fp16|/mnt/models/Qwen/Qwen3.6-27B|q27bf16|4"
  "q35b-a3b-fp16|/mnt/models/Qwen/Qwen3.6-35B-A3B|q35bf16|4"
  "gemma4-31b|/mnt/models/google/gemma-4-31B-it|gemma4|4"
  "q122b-int4|/mnt/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4|q122bint4|8"
)

clean_box_guard() {   # returns 0 if idle, 1 if busy
    local apps used
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1} END{print s+0}')
    if [[ "$apps" -ne 0 || "$used" -gt 2000 ]]; then
        note "box BUSY (compute_apps=$apps, used=${used}MiB)"
        return 1
    fi
    return 0
}

# run_one <label> <model> <served> <tp>
run_one() {
    local label="$1" model="$2" served="$3" tp="$4"
    local cname="smoke021_${label}" gpus
    local slog="$OUT/${label}_serve.log"
    local rfile="$OUT/${label}_response.json"
    local sfile="$OUT/${label}_sample.txt"
    if (( tp >= 8 )); then gpus="0,1,2,3,4,5,6,7"; else gpus="0,1,2,3"; fi

    if [[ ! -f "$model/config.json" ]]; then
        echo "$label: SKIP (model missing at $model)" | tee -a "$SUMMARY"; return
    fi
    if ! clean_box_guard; then
        echo "$label: SKIP (box busy — trainer running)" | tee -a "$SUMMARY"; return 1
    fi

    note "=== $label : $model  (TP=$tp on GPUs $gpus) ==="
    docker rm -f "$cname" >/dev/null 2>&1 || true

    docker run --rm -i --name "$cname" --gpus "\"device=$gpus\"" \
        -v /mnt/models:/mnt/models:ro \
        -v "$HOME/.cache/vllm-v100-021-triton:/root/.triton" \
        -v "$HOME/.cache/vllm-v100-021-torch:/root/.cache/torch" \
        -v "$HOME/.cache/vllm-v100-021-inductor:/tmp/torchinductor_root" \
        -p ${PORT}:${PORT} --shm-size=16g \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
        -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        "$IMAGE" \
        vllm serve "$model" --served-model-name "$served" \
            --tensor-parallel-size "$tp" --dtype float16 --enforce-eager \
            --max-model-len 4096 --max-num-seqs 8 \
            --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill \
            --host 0.0.0.0 --port "$PORT" \
        </dev/null >"$slog" 2>&1 &
    local lpid=$!

    local healthy=0 waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$lpid" 2>/dev/null || { note "  server process exited before healthy"; break; }
        sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "  ...loading $label (${waited}s)"
    done
    if [[ "$healthy" != 1 ]]; then
        echo "$label: FAIL (never healthy in ${HEALTH_TIMEOUT}s) — see $slog" | tee -a "$SUMMARY"
        tail -n 25 "$slog"
        docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
        return
    fi
    note "  healthy after ${waited}s. Generating coherence sample..."

    local url="http://localhost:${PORT}/v1/chat/completions"
    local prompt="Write a detailed multi-paragraph essay about the history, geography, and culture of France."
    local body
    body=$(python3 -c "import json,sys;print(json.dumps({'model':'$served','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':$MAXTOK,'temperature':0}))" "$prompt")
    curl -s "$url" -H 'Content-Type: application/json' -d "$body" >"$rfile" 2>&1

    # Coherence verdict + save the generated text for human eyeballing.
    local verdict
    verdict=$(python3 - "$rfile" "$sfile" <<'PY'
import json, sys, re
rfile, sfile = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(rfile))
    text = d["choices"][0]["message"]["content"]
    ntok = d.get("usage", {}).get("completion_tokens", 0)
    s = text.strip()
    open(sfile, "w").write(text)
    # repetition guard: most-common token shouldn't dominate (catches " the the the")
    words = s.split()
    rep = (max((words.count(w) for w in set(words)), default=0) / len(words)) if words else 1.0
    bang = (s.count("!") / len(s)) if s else 1.0
    ok = bool(s) and ntok >= 20 and bang < 0.3 and rep < 0.35
    print(("OK" if ok else "BAD") + f"\t{ntok}\t{rep:.2f}\t" + re.sub(r'\s+', ' ', s)[:140])
except Exception as e:
    print(f"BAD\t0\t1.00\tparse-error: {e}")
PY
)
    local tag ntok rep snip
    tag=$(printf '%s' "$verdict" | cut -f1); ntok=$(printf '%s' "$verdict" | cut -f2)
    rep=$(printf '%s' "$verdict" | cut -f3); snip=$(printf '%s' "$verdict" | cut -f4-)

    # Eager tok/s baseline: NRUN timed ignore_eos runs of TIMETOK tokens each.
    local tbody tot_t=0 tot_tok=0 i s_t e_t ct
    tbody=$(python3 -c "import json;print(json.dumps({'model':'$served','messages':[{'role':'user','content':'Continue this story in vivid detail.'}],'max_tokens':$TIMETOK,'temperature':0,'ignore_eos':True}))")
    curl -s "$url" -H 'Content-Type: application/json' -d "$tbody" >/dev/null 2>&1   # warmup
    for i in $(seq 1 "$NRUN"); do
        s_t=$(date +%s.%N); curl -s "$url" -H 'Content-Type: application/json' -d "$tbody" >"$OUT/.t.json" 2>&1; e_t=$(date +%s.%N)
        ct=$(python3 -c "import json;print(json.load(open('$OUT/.t.json'))['usage']['completion_tokens'])" 2>/dev/null || echo 0)
        tot_tok=$((tot_tok+${ct:-0})); tot_t=$(python3 -c "print($tot_t+($e_t-$s_t))")
    done
    local toks
    toks=$(python3 -c "print(f'{$tot_tok/$tot_t:.2f}')" 2>/dev/null || echo "n/a")

    if [[ "$tag" == "OK" ]]; then
        echo "$label: PASS  (${ntok} tok, rep=${rep}, ~${toks} tok/s eager) | \"$snip\"" | tee -a "$SUMMARY"
    else
        echo "$label: FAIL  (tag=$tag tok=$ntok rep=$rep) | \"$snip\" — see $slog / $sfile" | tee -a "$SUMMARY"
    fi

    note "  stopping $cname..."
    docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
}

main() {
    : > "$SUMMARY"
    echo "vLLM 0.21.0 sm_70 GATE-4 smoke — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
    echo "image=$IMAGE | eager | FP16/Int4 (no FP8) | TP per-model" >> "$SUMMARY"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; echo "ABORT: image missing" >>"$SUMMARY"; cat "$SUMMARY"; exit 1; }

    local row label model served tp
    for row in "${MATRIX[@]}"; do
        IFS='|' read -r label model served tp <<<"$row"
        [[ -n "$ONLY" && "$label" != *"$ONLY"* ]] && continue
        run_one "$label" "$model" "$served" "$tp" || {
            note "box busy — stopping remaining models to avoid colliding with the trainer."
            # mark the rest as skipped
            local r2 l2 rest
            for r2 in "${MATRIX[@]}"; do IFS='|' read -r l2 rest <<<"$r2"
                grep -q "^$l2:" "$SUMMARY" || echo "$l2: SKIP (box busy)" | tee -a "$SUMMARY"; done
            break
        }
    done

    echo; note "==== SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
    note "per-model generated text saved as $OUT/<label>_sample.txt ; serve logs as <label>_serve.log"
}

main "$@"
