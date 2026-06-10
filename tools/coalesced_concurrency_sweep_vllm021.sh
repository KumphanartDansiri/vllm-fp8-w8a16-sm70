#!/usr/bin/env bash
# Coalesced FP8 GEMV — CONCURRENCY SWEEP / cudagraph-cliff discriminator (vLLM 0.21 sm_70).
# Codex's Qwen M<=8 e2e showed 17.26 tok/s aggregate at concurrency=4 — BELOW the 37.24
# single-stream M=1. Concurrency should RAISE aggregate, not lower it. Hypothesis: the new
# M<=8 tiled kernel doesn't cudagraph-capture, so batch>=2 silently falls to EAGER (the
# ~3x eager penalty we've measured repeatedly), and the kernel's speed can't show.
#
# This run is the DISCRIMINATOR, on Gemma-31B PURE DENSE (no GDN, no MoE — removes the
# Qwen-hybrid confound). Two server configs, each swept over concurrency {1,2,4}:
#   m8 : coalesced ON, GEMV_M_MAX=8 -> batch>=2 uses the new M<=8 tiled kernel
#   m1 : coalesced ON, GEMV_M_MAX=1 -> batch>=2 falls back to A.3 (the control)
# READ:
#   - if m8 aggregate at C=4 < its own C=1  -> the Gemma stack reproduces the cliff
#     => it's cudagraph/scheduling (COMMON to both stacks), NOT Qwen GDN/MoE-specific.
#   - if m8 (fast kernel) < m1 (slow A.3) at C>=2 -> SMOKING GUN: M<=8 kernel breaks
#     capture (running eager costs more than its kernel speed saves).
#   - if m8 SCALES UP with C and beats m1 -> no cliff on Gemma => Qwen collapse is
#     GDN/MoE-specific, a different problem.
# Each config starts the server ONCE; the sweep reuses it (no reload per concurrency).
#
# Usage:  ./tools/coalesced_concurrency_sweep_vllm021.sh
# Env: IMAGE PORT TP MODEL MAXTOK UNROLL CONC HEALTH_TIMEOUT GPUMEM MAXLEN ONLY

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
PORT="${PORT:-8021}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
MAXTOK="${MAXTOK:-256}"          # longer gen amortizes TTFT/warmup (measure steady-state decode)
GPUMEM="${GPUMEM:-0.90}"
MAXLEN="${MAXLEN:-4096}"
TP="${TP:-4}"
MODEL="${MODEL:-/mnt/models/RedHatAI/gemma-4-31B-it-FP8-Dynamic}"
UNROLL="${UNROLL:-4}"
CONC="${CONC:-1 2 4}"
ONLY="${ONLY:-}"
CACHE_TAG="${CACHE_TAG:-021}"
SERVED="coalsweep"

# config-label | COALESCED | GEMV_M_MAX
CONFIGS=(
  "m8|1|8"
  "m1|1|1"
)

OUT=/tmp/v100_coalesced_sweep
mkdir -p "$OUT"; SUMMARY="$OUT/SUMMARY.txt"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note() { echo "[coal-sweep] $*"; }
gpu_list_for_tp() { local n="$1" i out=""; for ((i=0;i<n;i++)); do out+="${out:+,}$i"; done; echo "$out"; }
clean_box_guard() {
    local gpus="$1" used pids any=0; IFS=',' read -ra idxs <<<"$gpus"
    for i in "${idxs[@]}"; do
        used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
        pids=$(nvidia-smi --id="$i" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        [[ "${used:-9999}" -gt 2000 || "${pids:-1}" -gt 0 ]] && any=1
    done
    [[ "$any" -eq 0 ]]
}

run_config() {
    local label="$1" coalesced="$2" mmax="$3" gpus cname slog
    gpus=$(gpu_list_for_tp "$TP"); cname="coalsweep_${label}"; slog="$OUT/${label}_serve.log"
    [[ -f "$MODEL/config.json" ]] || { echo "$label: SKIP (missing $MODEL)" | tee -a "$SUMMARY"; return; }
    clean_box_guard "$gpus" || { echo "$label: SKIP (GPUs $gpus busy)" | tee -a "$SUMMARY"; return 1; }

    note "=== config=$label COALESCED=$coalesced M_MAX=$mmax UNROLL=$UNROLL (TP=$TP gpus=$gpus) — starting server ==="
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
        -e VLLM_V100_FP8_COALESCED_GEMV="$coalesced" \
        -e VLLM_V100_FP8_COALESCED_UNROLL="$UNROLL" \
        -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX="$mmax" \
        -e VLLM_V100_CT_FP8_RESIDENT=1 -e VLLM_V100_CT_FP8_RESIDENT_SELFCHECK=1 \
        -e VLLM_V100_CT_MOE_W13_RESIDENT=1 -e VLLM_V100_CT_MOE_W13_FREE_FP16=1 \
        -e VLLM_V100_CT_MOE_W2_GROUPED=1 -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        "$IMAGE" \
        python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
            --tensor-parallel-size "$TP" --dtype float16 \
            --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --max-model-len "$MAXLEN" --max-num-seqs 8 \
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
        echo "$label: FAIL (never healthy) — $slog" | tee -a "$SUMMARY"
        grep -nE "Error|Traceback|out of memory|assert" "$slog" | head -8
        docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; return 1
    fi
    note "  healthy after ${waited}s. warmup + concurrency sweep [$CONC]..."
    # warmup (JIT + cudagraph capture across batch sizes)
    curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(python3 -c "import json;print(json.dumps({'model':'$SERVED','messages':[{'role':'user','content':'Say hi.'}],'max_tokens':16,'temperature':0}))")" >/dev/null 2>&1 || true

    python3 - "$PORT" "$SERVED" "$MAXTOK" "$label" "$CONC" <<'PY' | tee -a "$SUMMARY"
import sys, json, time, threading, urllib.request
port, served, maxtok, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
concs = [int(x) for x in sys.argv[5].split()]
# distinct prompts so concurrent streams don't share prefix-cache KV (realistic batching)
TOPICS = ["France","Japan","Brazil","Egypt","Canada","India","Norway","Kenya"]
def stream_one(topic, res, i):
    body=json.dumps({"model":served,"stream":True,"max_tokens":maxtok,"temperature":0,
        "ignore_eos":True,"stream_options":{"include_usage":True},
        "messages":[{"role":"user","content":f"Write a detailed multi-paragraph essay about the history, geography, and culture of {topic}."}]}).encode()
    req=urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.time(); tf=tl=None; n=0; ut=0
    try:
        with urllib.request.urlopen(req,timeout=900) as r:
            for raw in r:
                line=raw.decode("utf-8","ignore").strip()
                if not line.startswith("data:"): continue
                d=line[5:].strip()
                if d=="[DONE]": break
                try: j=json.loads(d)
                except Exception: continue
                u=j.get("usage")
                if u and u.get("completion_tokens"): ut=int(u["completion_tokens"])
                c=j.get("choices") or []
                delta=c[0]["delta"].get("content") if c else None
                if delta:
                    now=time.time()
                    if tf is None: tf=now
                    tl=now; n+=1
        res[i]=(t0,tf,tl,(ut or n))
    except Exception as e:
        res[i]=(t0,None,None,0)
for C in concs:
    res=[None]*C; threads=[]
    wall0=time.time()
    for i in range(C):
        th=threading.Thread(target=stream_one,args=(TOPICS[i%len(TOPICS)],res,i)); threads.append(th); th.start()
    for th in threads: th.join()
    wall1=time.time()
    ok=[r for r in res if r and r[3]>1 and r[1] and r[2]]
    tot=sum(r[3] for r in ok)
    # STEADY-STATE metric (the standard): per-stream inter-token decode rate
    # (toks-1)/(last-first), EXCLUDING TTFT. Aggregate = SUM of per-stream rates.
    # NEVER use tokens/wall — wall folds in the TTFT/capture bubble and manufactures
    # phantom cliffs at low concurrency (the Qwen "17.26 C=4" + the C=2 ttft spikes
    # were exactly this artifact; 2026-06-10).
    ps=[(r[3]-1)/(r[2]-r[1]) for r in ok if r[2]>r[1]]
    ps_mean=sum(ps)/len(ps) if ps else float("nan")
    agg_steady=sum(ps) if ps else float("nan")        # true aggregate decode tput
    ttft=[ (r[1]-r[0]) for r in ok if r[1]]
    ttft_mean=sum(ttft)/len(ttft) if ttft else float("nan")
    wall_agg=tot/(wall1-wall0) if wall1>wall0 else float("nan")  # TTFT-inclusive, DIAGNOSTIC ONLY
    print(f"{label}: C={C}  agg_steady={agg_steady:6.2f} tok/s | per_stream={ps_mean:5.2f} tok/s | ttft={ttft_mean:4.2f}s | (wall_agg={wall_agg:5.2f} incl-ttft, diag) | ok={len(ok)}/{C}")
PY
    # cudagraph-capture evidence in the serve log (the hypothesis)
    grep -iE "cudagraph|capture|eager|piecewise" "$slog" 2>/dev/null | grep -iE "warn|skip|fall|disabl|unsupport|not support" | head -4 | sed 's/^/        cg: /' | tee -a "$SUMMARY"
    note "  stopping $cname..."; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
}

main() {
    : > "$SUMMARY"
    echo "Coalesced concurrency sweep [vLLM 0.21 sm_70] model=$(basename "$MODEL") TP=$TP unroll=$UNROLL conc=[$CONC] — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }
    local row label coalesced mmax
    for row in "${CONFIGS[@]}"; do
        IFS='|' read -r label coalesced mmax <<<"$row"
        [[ -n "$ONLY" && "$label" != *"$ONLY"* ]] && continue
        run_config "$label" "$coalesced" "$mmax" || true
        echo "" | tee -a "$SUMMARY"
    done
    note "==== SWEEP SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
    echo "  READ: cliff if m8 C=4 aggregate < m8 C=1; smoking-gun if m8 < m1 at C>=2 (fast kernel loses => eager/capture break)." | tee -a "$SUMMARY"
}
main "$@"
