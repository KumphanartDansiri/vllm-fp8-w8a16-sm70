#!/usr/bin/env bash
# Coalesced FP8 GEMV — e2e go/no-go A/B on stock vLLM 0.21 sm_70
# (image vllm-v100:vllm021-cu126), package mounted at /work (PYTHONPATH=/work/src),
# kernel JIT-compiled in-container. Answers the one open question: does the coalesced
# M=1 decode kernel's microbench win (A.3 25.3 sec/req @17.8% DRAM -> coalesced 2.4
# @29.4%, ~1.0-1.2x cuBLAS) SURVIVE end-to-end on a real model?
#
# THREE ARMS, same model/TP/everything-else, decode tok/s + coherence each:
#   a3    : VLLM_V100_FP8_COALESCED_GEMV=0  + CT_FP8_RESIDENT=1  -> old A.3 GEMV (slow baseline)
#   coal  : VLLM_V100_FP8_COALESCED_GEMV=1  + CT_FP8_RESIDENT=1  -> the new kernel (the test)
#   fp16  : (coalesced off)                 + CT_FP8_RESIDENT=0  -> dequant->FP16 cuBLAS (the CEILING)
# THESIS (for the 16GB / low-GPU-count GitHub audience): coal should close most of the
# a3->fp16 gap WHILE keeping FP8 weight memory (half of fp16). "coal ~= fp16 at half the
# VRAM" = the win. The fp16 arm is the speed ceiling; the a3 arm is what we're beating.
#
# DEFAULT TARGET = Gemma-4-31B dense channel-FP8: PURE DENSE so EVERY Linear is
# coalesced-eligible (M==1, block_w=128, block_h=1, K%128==0) -> maximum e2e signal.
# Validated on 0.21 this session (a3-resident=6.5, fp16=29 tok/s at TP=4). TP=2 here =
# the low-GPU-count audience config (fits 15.5GB/GPU FP8-resident, minimal all-reduce).
# Codex's gate is M==1 ONLY, so single-stream decode (ns implicitly 1 for the timed gen)
# is the right first test; batched M<=8 is future work and will fall back to A.3.
#
# Usage:  ./tools/coalesced_gemv_e2e_ab_vllm021.sh                    # all 3 arms, Gemma-31B, TP=2
#         ONLY=coal ./tools/coalesced_gemv_e2e_ab_vllm021.sh          # just the new-kernel arm
#         MODEL=/mnt/models/... TP=4 ./tools/coalesced_gemv_e2e_ab_vllm021.sh
# Env: IMAGE PORT HEALTH_TIMEOUT MAXTOK GPUMEM MAXLEN MODE TP MODEL ONLY

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
PORT="${PORT:-8021}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
MAXTOK="${MAXTOK:-200}"
GPUMEM="${GPUMEM:-0.90}"
MAXLEN="${MAXLEN:-4096}"
MODE="${MODE:-cudagraph}"
TP="${TP:-4}"   # TP=4 is the proven config; Codex found TP=2 stalls vLLM init/profiling
                # for the Qwen3.6 hybrid on this stack. Gemma-31B at TP=4 = 7.75GB/GPU
                # FP8-resident and matches the session baselines (a3=6.5, fp16=29 tok/s).
MODEL="${MODEL:-/mnt/models/RedHatAI/gemma-4-31B-it-FP8-Dynamic}"
ONLY="${ONLY:-}"
CACHE_TAG="${CACHE_TAG:-021}"
SERVED="coalab"

if [[ "$MODE" == "cudagraph" ]]; then
    EXEC_OPTS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
else
    EXEC_OPTS=(--enforce-eager)
fi

# arm-label | COALESCED | CT_FP8_RESIDENT
ARMS=(
  "a3|0|1"
  "coal|1|1"
  "fp16|0|0"
)

OUT=/tmp/v100_coalesced_ab
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note() { echo "[coal-ab] $*"; }

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

run_arm() {
    local label="$1" coalesced="$2" resident="$3" gpus cname slog sfile
    gpus=$(gpu_list_for_tp "$TP")
    cname="coalab_${label}"; slog="$OUT/${label}_serve.log"; sfile="$OUT/${label}_sample.txt"

    [[ -f "$MODEL/config.json" ]] || { echo "$label: SKIP (missing $MODEL)" | tee -a "$SUMMARY"; return; }
    clean_box_guard "$gpus" || { echo "$label: SKIP (GPUs $gpus busy)" | tee -a "$SUMMARY"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; return 1; }

    note "=== arm=$label  COALESCED=$coalesced CT_FP8_RESIDENT=$resident  (TP=$TP gpus=$gpus mode=$MODE) ==="
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
        -e VLLM_V100_FP8_COALESCED_UNROLL="${VLLM_V100_FP8_COALESCED_UNROLL:-2}" \
        -e VLLM_V100_FP8_COALESCED_M_UNROLL="${VLLM_V100_FP8_COALESCED_M_UNROLL:-${VLLM_V100_FP8_COALESCED_UNROLL:-2}}" \
        -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX="${VLLM_V100_FP8_COALESCED_GEMV_M_MAX:-1}" \
        -e VLLM_V100_FP8_PREFIX_VARIANT_PROFILE="${VLLM_V100_FP8_PREFIX_VARIANT_PROFILE:-0}" \
        -e VLLM_V100_FP8_PREFIX_VARIANT_FILTER="${VLLM_V100_FP8_PREFIX_VARIANT_FILTER:-shared_expert}" \
        -e VLLM_V100_CT_FP8_RESIDENT="$resident" \
        -e VLLM_V100_CT_FP8_RESIDENT_SELFCHECK=1 \
        -e VLLM_V100_CT_MOE_W13_RESIDENT=1 -e VLLM_V100_CT_MOE_W13_FREE_FP16=1 \
        -e VLLM_V100_CT_MOE_W2_GROUPED=1 \
        -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        "$IMAGE" \
        python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
            --tensor-parallel-size "$TP" --dtype float16 "${EXEC_OPTS[@]}" \
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
        grep -nE "Error|Traceback|no kernel image|out of memory|assert" "$slog" | head -8
        docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; return 1
    fi

    # warmup (JIT + cudagraph capture), then a timed single-stream decode
    curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(python3 -c "import json;print(json.dumps({'model':'$SERVED','messages':[{'role':'user','content':'Say hi.'}],'max_tokens':16,'temperature':0}))")" >/dev/null 2>&1 || true

    local verdict; verdict=$(python3 - "$PORT" "$SERVED" "$MAXTOK" "$sfile" <<'PY'
import sys, json, re, time, urllib.request
port, served, maxtok, sfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
body = json.dumps({"model": served, "stream": True, "max_tokens": maxtok, "temperature": 0,
    "stream_options": {"include_usage": True},
    "messages": [{"role": "user", "content": "Write a detailed multi-paragraph essay about the history, geography, and culture of France."}]}).encode()
req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
t0=time.time(); tf=tl=None; n=0; ch=[]; ut=0
try:
    with urllib.request.urlopen(req, timeout=900) as r:
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
                tl=now; n+=1; ch.append(delta)
    s="".join(ch).strip(); open(sfile,"w").write(s); w=s.split()
    rep=(max((w.count(x) for x in set(w)),default=0)/len(w)) if w else 1.0
    ttft=(tf-t0) if tf else float("nan")
    dt=(tl-tf) if (tf and tl and n>1) else float("nan")
    mt=ut if ut else n
    dtps=((mt-1)/dt) if (dt and dt>0) else float("nan")
    ok=bool(s) and n>=20 and rep<0.35
    print(("OK" if ok else "BAD")+f"\t{ut}\t{rep:.2f}\t{ttft:.2f}\t{dtps:.2f}\t"+re.sub(r'\s+',' ',s)[:120])
except Exception as e:
    print(f"BAD\t0\t1.0\tnan\tnan\tstream-error: {e}")
PY
)
    local tag ut rep ttft dtps snip
    tag=$(printf '%s' "$verdict"|cut -f1); ut=$(printf '%s' "$verdict"|cut -f2)
    rep=$(printf '%s' "$verdict"|cut -f3); ttft=$(printf '%s' "$verdict"|cut -f4)
    dtps=$(printf '%s' "$verdict"|cut -f5); snip=$(printf '%s' "$verdict"|cut -f6-)
    # PROVE which kernel ran: variant counts banner (Coalesced GEMV vs A.3) from rank-0.
    local kern; kern=$(grep -oE "kernel variant counts[^\"]*" "$slog" | tail -1)
    local coal_hits; coal_hits=$(grep -c "Coalesced GEMV" "$slog" 2>/dev/null | head -1)
    echo "$label: $tag tok=$ut rep=$rep ttft=${ttft}s decode=${dtps} tok/s | coalesced-banner-hits=${coal_hits:-0} ${kern:+| $kern}" | tee -a "$SUMMARY"
    echo "        \"$snip\"" | tee -a "$SUMMARY"
    note "  stopping $cname..."; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
}

main() {
    : > "$SUMMARY"
    echo "Coalesced FP8 GEMV e2e A/B [vLLM 0.21 sm_70] model=$(basename "$MODEL") TP=$TP mode=$MODE — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }
    local row label coalesced resident
    for row in "${ARMS[@]}"; do
        IFS='|' read -r label coalesced resident <<<"$row"
        [[ -n "$ONLY" && "$label" != *"$ONLY"* ]] && continue
        run_arm "$label" "$coalesced" "$resident" || true
        echo "" | tee -a "$SUMMARY"
    done
    note "==== COALESCED A/B SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
    echo "  READ: coal decode/s should be >> a3 and ~= fp16, with coalesced-banner-hits>0 ONLY in the coal arm." | tee -a "$SUMMARY"
}
main "$@"
