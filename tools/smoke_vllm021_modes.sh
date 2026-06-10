#!/usr/bin/env bash
# Phase 2+3 of the vLLM 0.21 sm_70 overnight validation: cudagraph and
# cudagraph+MTP, GATED on a single-model canary.
#
# Companion to tools/smoke_vllm021.sh (phase 1 = eager × all 4). This script:
#   PHASE 2 (canary): run Qwen3.6-27B in cudagraph, then cudagraph+MTP.
#   GATE:    27B must PASS eager (phase 1) AND cudagraph AND cudagraph+MTP.
#   PHASE 3 (batch):  only if the gate passes, run the REMAINING models —
#            cudagraph × {35B-A3B, gemma-4, 122B-Int4}, and
#            cudagraph+MTP × {35B-A3B, 122B-Int4}  (gemma-4 has no MTP head).
#
# Why gate: cudagraph capture and MTP speculative decoding are unproven on
# 0.21/sm_70. Confirm both on the cheap 27B (TP4) before spending the night on
# the slow 122B (TP8). If the canary fails, the batch is skipped and the log
# tells you which capability broke.
#
# MTP correctness is verified by EXACTNESS, not acceptance rate: the
# cudagraph+MTP greedy output must equal the plain-cudagraph greedy output
# token-for-token (speculative decode is lossless at temperature 0). Acceptance
# rate is recorded too, but only as a perf signal. (Lesson from the 1catai
# TRITON+MTP "97.5% acceptance but garbage output" incident.)
#
# Shared-box rule: clean-box guard before every model; if the trainer resumes,
# remaining models are SKIPPED, not crashed into.
#
# Usage:
#   ./tools/smoke_vllm021_modes.sh              # wait for eager batch, then run
#   WAIT_EAGER=0 ./tools/smoke_vllm021_modes.sh # don't wait (GPUs already free)
#   CANARY_ONLY=1 ./tools/smoke_vllm021_modes.sh # phase 2 only, skip the batch

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
PORT="${PORT:-8021}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3000}"   # cudagraph capture adds time on top of load
MAXTOK="${MAXTOK:-200}"
NRUN="${NRUN:-2}"
TIMETOK="${TIMETOK:-128}"
GPUMEM="${GPUMEM:-0.85}"
WAIT_EAGER="${WAIT_EAGER:-1}"
CANARY_ONLY="${CANARY_ONLY:-0}"
# Proven on the 0.19 MTP work; 0.21 also accepts method "qwen3_5_mtp". No spaces
# in the JSON so it word-splits into a single argv token cleanly.
MTP_SPEC="${MTP_SPEC:-{\"method\":\"mtp\",\"num_speculative_tokens\":1}}"
CG_CONFIG="${CG_CONFIG:-{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}}"

OUT=/tmp/v100_smoke021
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY_modes.txt"
EAGER_SUMMARY="$OUT/SUMMARY.txt"
for s in triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-021-$s"; done

note() { echo "[modes021] $*"; }

# label|model|served|tp|has_mtp
ALL=(
  "q27b-fp16|/mnt/models/Qwen/Qwen3.6-27B|q27bf16|4|1"
  "q35b-a3b-fp16|/mnt/models/Qwen/Qwen3.6-35B-A3B|q35bf16|4|1"
  "gemma4-31b|/mnt/models/google/gemma-4-31B-it|gemma4|4|0"
  "q122b-int4|/mnt/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4|q122bint4|8|1"
)
row_for() { local l="$1" r; for r in "${ALL[@]}"; do [[ "$r" == "$l|"* ]] && { echo "$r"; return; }; done; }

clean_box_guard() {
    local apps used
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1} END{print s+0}')
    [[ "$apps" -eq 0 && "$used" -le 2000 ]]
}

# run_one <label> <mode>   mode = cudagraph | cudagraph-mtp
# returns 0 on PASS, 1 otherwise. Writes <label>_<mode>_*.{log,sample.txt}.
run_one() {
    local label="$1" mode="$2"
    local r model served tp has_mtp gpus
    r="$(row_for "$label")"; IFS='|' read -r _ model served tp has_mtp <<<"$r"
    if (( tp >= 8 )); then gpus="0,1,2,3,4,5,6,7"; else gpus="0,1,2,3"; fi
    local cname="smoke021_${label}_${mode}"
    local slog="$OUT/${label}_${mode}_serve.log"
    local sfile="$OUT/${label}_${mode}_sample.txt"
    local rfile="$OUT/${label}_${mode}_response.json"

    if [[ "$mode" == "cudagraph-mtp" && "$has_mtp" != "1" ]]; then
        echo "$label [$mode]: SKIP (no MTP head)" | tee -a "$SUMMARY"; return 0
    fi
    if [[ ! -f "$model/config.json" ]]; then
        echo "$label [$mode]: SKIP (model missing)" | tee -a "$SUMMARY"; return 1
    fi
    if ! clean_box_guard; then
        echo "$label [$mode]: SKIP (box busy — trainer running)" | tee -a "$SUMMARY"; return 2
    fi

    # mode -> serve args
    local MARGS=()
    case "$mode" in
        cudagraph)     MARGS=(--compilation-config "$CG_CONFIG") ;;
        cudagraph-mtp) MARGS=(--compilation-config "$CG_CONFIG" --speculative-config "$MTP_SPEC") ;;
        *) note "unknown mode $mode"; return 1 ;;
    esac

    note "=== $label [$mode] : $model (TP=$tp on $gpus) ==="
    docker rm -f "$cname" >/dev/null 2>&1 || true
    docker run --rm -i --name "$cname" --gpus "\"device=$gpus\"" \
        -v /mnt/models:/mnt/models:ro \
        -v "$HOME/.cache/vllm-v100-021-triton:/root/.triton" \
        -v "$HOME/.cache/vllm-v100-021-torch:/root/.cache/torch" \
        -v "$HOME/.cache/vllm-v100-021-inductor:/tmp/torchinductor_root" \
        -p ${PORT}:${PORT} --shm-size=16g \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        "$IMAGE" \
        vllm serve "$model" --served-model-name "$served" \
            --tensor-parallel-size "$tp" --dtype float16 \
            --max-model-len 4096 --max-num-seqs 8 \
            --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill \
            "${MARGS[@]}" --host 0.0.0.0 --port "$PORT" \
        </dev/null >"$slog" 2>&1 &
    local lpid=$!

    local healthy=0 waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$lpid" 2>/dev/null || { note "  server exited before healthy"; break; }
        sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "  ...loading $label/$mode (${waited}s)"
    done
    if [[ "$healthy" != 1 ]]; then
        echo "$label [$mode]: FAIL (never healthy in ${HEALTH_TIMEOUT}s) — see $slog" | tee -a "$SUMMARY"
        tail -n 25 "$slog"; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
        return 1
    fi
    note "  healthy after ${waited}s. Generating..."

    local url="http://localhost:${PORT}/v1/chat/completions"
    local prompt="Write a detailed multi-paragraph essay about the history, geography, and culture of France."
    local body
    body=$(python3 -c "import json,sys;print(json.dumps({'model':'$served','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':$MAXTOK,'temperature':0}))" "$prompt")
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
    rep = (max((words.count(w) for w in set(words)), default=0) / len(words)) if words else 1.0
    bang = (s.count("!") / len(s)) if s else 1.0
    ok = bool(s) and ntok >= 20 and bang < 0.3 and rep < 0.35
    print(("OK" if ok else "BAD") + f"\t{ntok}\t{rep:.2f}\t" + re.sub(r'\s+', ' ', s)[:120])
except Exception as e:
    print(f"BAD\t0\t1.00\tparse-error: {e}")
PY
)
    local tag ntok rep snip
    tag=$(printf '%s' "$verdict" | cut -f1); ntok=$(printf '%s' "$verdict" | cut -f2)
    rep=$(printf '%s' "$verdict" | cut -f3); snip=$(printf '%s' "$verdict" | cut -f4-)

    # MTP-specific: exactness vs the cudagraph reference + acceptance rate.
    local mtp_note=""
    if [[ "$mode" == "cudagraph-mtp" ]]; then
        local ref="$OUT/${label}_cudagraph_sample.txt"
        if [[ -f "$ref" ]]; then
            if diff -q "$ref" "$sfile" >/dev/null 2>&1; then mtp_note="exact=MATCH"
            else mtp_note="exact=DIFF(!)"; fi
        else mtp_note="exact=NO-REF"; fi
        curl -s "http://localhost:${PORT}/metrics" 2>/dev/null > "$OUT/${label}_${mode}_metrics.txt" 2>/dev/null || true
        local acc
        acc=$(curl -s "http://localhost:${PORT}/metrics" 2>/dev/null | python3 -c "
import sys,re
acc=draft=0.0
for ln in sys.stdin:
    if ln.startswith('vllm:spec_decode_num_accepted_tokens') and 'total' in ln:
        try: acc=float(ln.split()[-1])
        except: pass
    if ln.startswith('vllm:spec_decode_num_draft_tokens') and 'total' in ln:
        try: draft=float(ln.split()[-1])
        except: pass
print(f'{acc/draft:.2%}' if draft>0 else 'n/a')
" 2>/dev/null || echo "n/a")
        mtp_note="$mtp_note accept=$acc"
    fi

    # eager tok/s baseline (timed ignore_eos)
    local tbody tot_t=0 tot_tok=0 i s_t e_t ct
    tbody=$(python3 -c "import json;print(json.dumps({'model':'$served','messages':[{'role':'user','content':'Continue this story in vivid detail.'}],'max_tokens':$TIMETOK,'temperature':0,'ignore_eos':True}))")
    curl -s "$url" -H 'Content-Type: application/json' -d "$tbody" >/dev/null 2>&1
    for i in $(seq 1 "$NRUN"); do
        s_t=$(date +%s.%N); curl -s "$url" -H 'Content-Type: application/json' -d "$tbody" >"$OUT/.t.json" 2>&1; e_t=$(date +%s.%N)
        ct=$(python3 -c "import json;print(json.load(open('$OUT/.t.json'))['usage']['completion_tokens'])" 2>/dev/null || echo 0)
        tot_tok=$((tot_tok+${ct:-0})); tot_t=$(python3 -c "print($tot_t+($e_t-$s_t))")
    done
    local toks; toks=$(python3 -c "print(f'{$tot_tok/$tot_t:.2f}')" 2>/dev/null || echo "n/a")

    local rc=0
    if [[ "$tag" == "OK" && ( "$mode" != "cudagraph-mtp" || "$mtp_note" == "exact=MATCH"* ) ]]; then
        echo "$label [$mode]: PASS  (${ntok} tok, rep=$rep, ~${toks} tok/s) $mtp_note | \"$snip\"" | tee -a "$SUMMARY"
    else
        echo "$label [$mode]: FAIL  (tag=$tag tok=$ntok rep=$rep) $mtp_note | \"$snip\" — see $slog" | tee -a "$SUMMARY"; rc=1
    fi
    note "  stopping $cname..."; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
    return $rc
}

wait_for_eager() {
    [[ "$WAIT_EAGER" != "1" ]] && { note "WAIT_EAGER=0 — not waiting for the eager batch."; return; }
    note "waiting for the eager batch (phase 1) to finish before touching the GPUs..."
    local w=0
    while ! grep -q "==== SUMMARY" "$OUT/run.log" 2>/dev/null; do
        # also break if no eager container AND eager summary already has all 4
        sleep 30; w=$((w+30)); (( w % 300 == 0 )) && note "  ...still waiting on eager (${w}s)"
    done
    # ensure GPUs actually freed
    while ! clean_box_guard; do note "  eager done but GPUs still busy; waiting..."; sleep 20; done
    note "eager batch complete; GPUs free. Starting phase 2."
}

main() {
    : > "$SUMMARY"
    echo "vLLM 0.21.0 sm_70 — cudagraph + cudagraph-MTP (gated) — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }

    wait_for_eager

    # ---- PHASE 2: 27B canary ----
    note "PHASE 2 — 27B canary (cudagraph, then cudagraph+MTP)"
    run_one q27b-fp16 cudagraph;     local cg=$?
    run_one q27b-fp16 cudagraph-mtp; local mtp=$?

    # eager verdict from phase 1
    local eager_ok=1
    grep -qE "^q27b-fp16: PASS" "$EAGER_SUMMARY" 2>/dev/null || eager_ok=0

    echo "" | tee -a "$SUMMARY"
    echo "GATE: 27B eager=$([[ $eager_ok == 1 ]] && echo PASS || echo FAIL) cudagraph=$([[ $cg == 0 ]] && echo PASS || echo FAIL) cudagraph-mtp=$([[ $mtp == 0 ]] && echo PASS || echo FAIL)" | tee -a "$SUMMARY"

    if [[ "$CANARY_ONLY" == "1" ]]; then note "CANARY_ONLY=1 — stopping after canary."; cat "$SUMMARY"; return; fi

    if [[ "$eager_ok" == 1 && "$cg" == 0 && "$mtp" == 0 ]]; then
        note "GATE PASSED — running phase 3 batch for the remaining models."
        echo "--- PHASE 3 batch (gate passed) ---" | tee -a "$SUMMARY"
        for label in q35b-a3b-fp16 gemma4-31b q122b-int4; do
            run_one "$label" cudagraph || true
        done
        for label in q35b-a3b-fp16 q122b-int4; do
            run_one "$label" cudagraph-mtp || true
        done
    else
        note "GATE FAILED — skipping phase 3 batch. Inspect 27B logs above."
        echo "PHASE 3 SKIPPED (canary gate failed)" | tee -a "$SUMMARY"
    fi

    echo; note "==== MODES SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
}

main "$@"
