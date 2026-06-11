#!/usr/bin/env bash
# Coalesced FP8 GEMV on Qwen3.5-122B-A10B-FP8 (BLOCK-FP8) — e2e decode A/B on stock
# vLLM 0.21 sm_70 (image vllm-v100:vllm021-cu126), package mounted at /work
# (PYTHONPATH=/work/src), kernel JIT-compiled in-container. TP=8 flagship.
#
# Unlike the CT models (GLM-Air, Gemma-4), 122B is Qwen BLOCK-FP8 (quant_method=fp8,
# weight_block_size=[128,128]) -> it routes through Fp8LinearMethod + the grouped
# `_our_moe_apply_grouped` decode path, NOT the compressed_tensors_v100 hook. So the
# CT env vars (VLLM_V100_CT_*) do NOTHING here; the levers are:
#   - attn / dense Linears  : VLLM_V100_FP8_COALESCED_GEMV   (via _v100_fp8_gemm, M==1)
#   - routed MoE w13 decode : VLLM_V100_FP8_MOE_W13_COALESCED (the grouped coalesced GEMV)
#
# THREE ARMS, same model/TP/cudagraph, streaming steady-state per-stream decode each:
#   base : GEMV=0  MOE_W13_COALESCED=0  -> all A.3 (the documented 34.6 tok/s baseline)
#   attn : GEMV=1  MOE_W13_COALESCED=0  -> coalesced attn/dense Linears only
#   full : GEMV=1  MOE_W13_COALESCED=1  -> + coalesced routed MoE w13 (the full win)
# This mirrors the GLM-Air journey (30.7 -> 45.4 attn -> 56.6 +MoE w13). 122B is the
# can't-fit-FP16 flagship: FP8-resident is the ONLY path that loads on 8xV100-32GB,
# so making it fast IS the headline benefit.
#
# Metric = streaming inter-token steady-state decode tok/s ((tokens-1)/(t_last-t_first)),
# NEVER tokens/wall (folds in TTFT/capture bubble -> phantom cliffs). Single-stream.
#
# Usage:  ./tools/coalesced_122b_ab_vllm021.sh                 # all 3 arms, TP=8
#         ONLY=full ./tools/coalesced_122b_ab_vllm021.sh       # just the full-coalesced arm
#         UNROLL=2 ./tools/coalesced_122b_ab_vllm021.sh        # override K-unroll (default 4)
# Env: IMAGE PORT HEALTH_TIMEOUT MAXTOK GPUMEM MAXLEN MODE TP MODEL ONLY UNROLL NS
#
# Shared-box note: TP=8 grabs all GPUs. The clean-box guard aborts if any GPU is busy
# (son's training). Results are file-captured to /tmp/v100_122b_coalesced_ab/SUMMARY.txt.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
PORT="${PORT:-8021}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3600}"
MAXTOK="${MAXTOK:-200}"
GPUMEM="${GPUMEM:-0.92}"
MAXLEN="${MAXLEN:-4096}"
MODE="${MODE:-cudagraph}"
TP="${TP:-8}"
MODEL="${MODEL:-/mnt/models/Qwen/Qwen3.5-122B-A10B-FP8}"
ONLY="${ONLY:-}"
UNROLL="${UNROLL:-4}"
NS="${NS:-8}"
CACHE_TAG="${CACHE_TAG:-021}"
SERVED="coal122b"

if [[ "$MODE" == "cudagraph" ]]; then
    EXEC_OPTS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
else
    EXEC_OPTS=(--enforce-eager)
fi

# arm-label | COALESCED_GEMV | MOE_W13_COALESCED
ARMS=(
  "base|0|0"
  "attn|1|0"
  "full|1|1"
)

OUT=/tmp/v100_122b_coalesced_ab
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note() { echo "[122b-coal-ab] $*"; }

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
    local label="$1" coalesced="$2" moe_w13="$3" gpus cname slog sfile
    gpus=$(gpu_list_for_tp "$TP")
    cname="coal122b_${label}"; slog="$OUT/${label}_serve.log"; sfile="$OUT/${label}_sample.txt"

    [[ -f "$MODEL/config.json" ]] || { echo "$label: SKIP (missing $MODEL)" | tee -a "$SUMMARY"; return; }
    clean_box_guard "$gpus" || { echo "$label: SKIP (GPUs $gpus busy)" | tee -a "$SUMMARY"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; return 1; }

    note "=== arm=$label  COALESCED_GEMV=$coalesced MOE_W13_COALESCED=$moe_w13  (TP=$TP gpus=$gpus mode=$MODE unroll=$UNROLL) ==="
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
        -e VLLM_V100_FP8_COALESCED_M_UNROLL="$UNROLL" \
        -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX="${VLLM_V100_FP8_COALESCED_GEMV_M_MAX:-8}" \
        -e VLLM_V100_FP8_MOE_W13_COALESCED="$moe_w13" \
        -e VLLM_V100_FP8_MOE_FALLBACK=1 \
        -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1 \
        -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS="${VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS:-128}" \
        -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1 \
        -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        "$IMAGE" \
        python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
            --tensor-parallel-size "$TP" --dtype float16 "${EXEC_OPTS[@]}" \
            --max-model-len "$MAXLEN" --max-num-seqs "$NS" \
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
    # banner proof (capture is best-effort on the 0.21 spawn harness; the controlled
    # flag-only delta base->attn->full is the real proof, banners are corroboration).
    local gemv_hits moe_hits
    gemv_hits=$(grep -c "Coalesced GEMV" "$slog" 2>/dev/null | head -1)
    moe_hits=$(grep -c "decode w13 = grouped COALESCED" "$slog" 2>/dev/null | head -1)
    echo "$label: $tag tok=$ut rep=$rep ttft=${ttft}s decode=${dtps} tok/s | gemv-banner=${gemv_hits:-0} moe-w13-coal-banner=${moe_hits:-0}" | tee -a "$SUMMARY"
    echo "        \"$snip\"" | tee -a "$SUMMARY"
    note "  stopping $cname..."; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
}

main() {
    : > "$SUMMARY"
    echo "Coalesced FP8 122B A/B [vLLM 0.21 sm_70] model=$(basename "$MODEL") TP=$TP mode=$MODE unroll=$UNROLL — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }
    local row label coalesced moe_w13
    for row in "${ARMS[@]}"; do
        IFS='|' read -r label coalesced moe_w13 <<<"$row"
        [[ -n "$ONLY" && "$label" != *"$ONLY"* ]] && continue
        run_arm "$label" "$coalesced" "$moe_w13" || true
        echo "" | tee -a "$SUMMARY"
    done
    note "==== 122B COALESCED A/B SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
    echo "  READ: base ~= documented 34.6; attn > base (coalesced Linears); full >= attn (coalesced MoE w13)." | tee -a "$SUMMARY"
    echo "        Expect modest gains vs GLM-Air's 1.84x — 122B is TP=8 (all-reduce a bigger fixed fraction) AND" | tee -a "$SUMMARY"
    echo "        only 10B active / smaller MoE-intermediate (1024) so the w13 GEMV is a smaller slice of the budget." | tee -a "$SUMMARY"
}
main "$@"
