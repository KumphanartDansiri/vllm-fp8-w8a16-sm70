#!/usr/bin/env bash
# Fit-aware TP × concurrency sweep for the v100-llm-2026 model pages (Ch6).
# For one model+precision, serve at each TP in a list × each concurrency in a list, capture
# steady-state decode (1-user) or per-user+aggregate (N-user). FIT-AWARE: a TP that OOMs / never
# goes healthy is recorded "infeasible" and skipped — the sweep continues. Results land durable
# under results/tp_sweep_<key>_<stamp>/ for build_matrix_from_results.py to extract later.
#
# Usage:
#   MODEL_KEY=q27b PREC=fp8 TPS="1 2 4 8" USERS="1 8" ./tools/tp_concurrency_sweep.sh
#   MODEL_KEY=q27b PREC=fp16 TPS="2 4 8" ...
# Env: MODEL_KEY PREC TPS USERS IMAGE MAXLEN GENTOK NRUN GPUMEM HEALTH_TIMEOUT OUT
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

MODEL_KEY="${MODEL_KEY:-q27b}"; PREC="${PREC:-fp8}"
case "${MODEL_KEY}:${PREC}" in
  q27b:fp16)  MODEL=/mnt/models/Qwen/Qwen3.6-27B ;;
  q27b:fp8)   MODEL=/mnt/models/Qwen/Qwen3.6-27B-FP8 ;;
  q35b:fp16)  MODEL=/mnt/models/Qwen/Qwen3.6-35B-A3B ;;
  q35b:fp8)   MODEL=/mnt/models/Qwen/Qwen3.6-35B-A3B-FP8 ;;
  q27b:int4)  MODEL=/mnt/models/Qwen/Qwen3.5-27B-GPTQ-Int4 ;;       # 3.5 GPTQ engine-matched re-run (0.21)
  q35b:int4)  MODEL=/mnt/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 ;;   # 3.5 GPTQ engine-matched re-run (0.21)
  q122b:fp8)  MODEL=/mnt/models/Qwen/Qwen3.5-122B-A10B-FP8 ;;
  q122b:int4) MODEL=/mnt/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 ;;   # 3.5 122B GPTQ dual-engine consistency run
  g31b:fp16)  MODEL=/mnt/models/google/gemma-4-31B-it ;;
  g31b:fp8)   MODEL=/mnt/models/RedHatAI/gemma-4-31B-it-FP8-Dynamic ;;
  g26b:fp16)  MODEL=/mnt/models/google/gemma-4-26B-A4B-it ;;
  g26b:fp8)   MODEL=/mnt/models/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic ;;
  glm:fp8)    MODEL=/mnt/models/zai-org/GLM-4.5-Air-FP8 ;;
  *) echo "unknown ${MODEL_KEY}:${PREC}"; exit 1 ;;
esac
MODEL="${MODEL_OVERRIDE:-$MODEL}"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"; CACHE_TAG="${CACHE_TAG:-021cu126}"
TPS="${TPS:-1 2 4 8}"; USERS="${USERS:-1 8}"
MAXLEN="${MAXLEN:-4096}"; GENTOK="${GENTOK:-512}"; NRUN="${NRUN:-2}"
GPUMEM="${GPUMEM:-0.90}"; HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"; PORT="${PORT:-8040}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="${OUT:-results/tp_sweep_${MODEL_KEY}_${PREC}_${STAMP}}"
SERVED="tpsweep"; mkdir -p "$OUT"; SUMMARY="$OUT/SUMMARY.txt"; : > "$SUMMARY"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note(){ echo "[tpsweep] $*" | tee -a "$SUMMARY"; }

gpus_for(){ local n="$1" o="" i; for ((i=0;i<n;i++)); do o+="${o:+,}$i"; done; echo "$o"; }
clean_guard(){ local g="$1" any=0 i used; IFS=',' read -ra a <<<"$g"
  for i in "${a[@]}"; do used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null|awk '{print $1+0}')
    [[ "${used:-9999}" -gt 2000 ]] && any=1; done; [[ "$any" -eq 0 ]]; }

note "model=$MODEL prec=$PREC TPS='$TPS' USERS='$USERS' maxlen=$MAXLEN gentok=$GENTOK — $(date -u +%FT%TZ)"
[[ -f "$MODEL/config.json" ]] || { note "MISSING MODEL $MODEL"; exit 1; }

if [[ "$PREC" == fp8 ]]; then
  QUANT=(--quantization fp8)
  FENV=(-e VLLM_V100_FP8_COALESCED_GEMV=1 -e VLLM_V100_FP8_COALESCED_UNROLL=4
        -e VLLM_V100_FP8_COALESCED_M_UNROLL=4 -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8
        -e VLLM_V100_FP8_MOE_W13_COALESCED=1 -e VLLM_V100_FP8_MOE_FALLBACK=1
        -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1 -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=512
        -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1 -e VLLM_V100_CT_FP8_RESIDENT=1
        -e VLLM_V100_CT_MOE_W13_RESIDENT=1 -e VLLM_V100_CT_MOE_W2_GROUPED=1
        -e VLLM_V100_CT_MOE_W13_COALESCED=1)
elif [[ "$PREC" == int4 ]]; then
  QUANT=(--quantization gptq); FENV=(-e VLLM_V100_FP8_COALESCED_GEMV=0)   # stock GPTQ on 0.21 (plugin inert) — engine-matched to FP16/FP8
else
  QUANT=(); FENV=(-e VLLM_V100_FP8_COALESCED_GEMV=0)   # MoE BLOCK_K patch is default-ON
fi

measure(){ # $1=port $2=nusers -> prints "per_user_mean<TAB>aggregate"
  python3 - "$1" "$SERVED" "$GENTOK" "$2" <<'PY'
import json,sys,threading,time,urllib.request
port,served,tok,nu=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
asp=["history","geography","economy","culture","cuisine","politics","science","art"]
res=[None]*nu
def one(u):
    b=json.dumps({"model":served,"stream":True,"max_tokens":tok,"temperature":0,"ignore_eos":True,
      "stream_options":{"include_usage":True},
      "messages":[{"role":"user","content":f"Write a long detailed essay about the {asp[u%len(asp)]} of France."}]}).encode()
    rq=urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",data=b,headers={"Content-Type":"application/json"})
    tf=tl=None;n=0;ut=0
    with urllib.request.urlopen(rq,timeout=2400) as r:
        for raw in r:
            L=raw.decode("utf-8","ignore").strip()
            if not L.startswith("data:"):continue
            d=L[5:].strip()
            if d=="[DONE]":break
            try:j=json.loads(d)
            except:continue
            uu=j.get("usage")
            if uu and uu.get("completion_tokens"):ut=int(uu["completion_tokens"])
            c=j.get("choices") or [];dl=c[0]["delta"].get("content") if c else None
            if dl:
                t=time.time()
                if tf is None:tf=t
                tl=t;n+=1
    mt=ut or n
    res[u]=((mt-1)/(tl-tf)) if tf and tl and tl>tf and mt>1 else float("nan")
ts=[threading.Thread(target=one,args=(u,)) for u in range(nu)]
t0=time.time()
for t in ts:t.start()
for t in ts:t.join()
ok=[x for x in res if x==x]
print(f"{(sum(ok)/len(ok)):.2f}\t{sum(ok):.2f}" if ok else "nan\t nan")
PY
}

for tp in $TPS; do
  gpus=$(gpus_for "$tp"); cname="tpsweep_${MODEL_KEY}_${PREC}_tp${tp}"; slog="$OUT/tp${tp}_serve.log"
  if ! clean_guard "$gpus"; then note "TP$tp: SKIP (GPUs $gpus busy)"; continue; fi
  note "=== TP$tp (gpus=$gpus) loading ==="
  docker rm -f "$cname" >/dev/null 2>&1 || true
  docker run --rm -i --name "$cname" --gpus "\"device=$gpus\"" \
    -v /mnt/models:/mnt/models:ro -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
    -p ${PORT}:${PORT} --shm-size=16g \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_ATTENTION_BACKEND=TRITON_ATTN "${FENV[@]}" \
    "$IMAGE" \
    python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
      --tensor-parallel-size "$tp" --dtype float16 "${QUANT[@]}" \
      --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --max-model-len "$MAXLEN" --max-num-seqs 8 --skip-mm-profiling \
      --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill --host 0.0.0.0 --port "$PORT" \
    </dev/null >"$slog" 2>&1 &
  lpid=$!; healthy=0; waited=0
  while (( waited < HEALTH_TIMEOUT )); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
    kill -0 "$lpid" 2>/dev/null || break
    sleep 10; waited=$((waited+10))
  done
  if [[ "$healthy" != 1 ]]; then
    if grep -qiE "out of memory|OOM|CUDA out of memory|No available memory" "$slog"; then
      note "TP$tp: INFEASIBLE (OOM — does not fit at TP$tp)"
    else
      note "TP$tp: INFEASIBLE/FAIL (never healthy; see $slog)"
      grep -nE "Error|Traceback" "$slog" | head -3 | tee -a "$SUMMARY"
    fi
    docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; continue
  fi
  # warm (JIT + cudagraph capture)
  curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" >/dev/null 2>&1 || true
  for u in $USERS; do
    best_pu=""; best_ag=""
    for r in $(seq 1 "$NRUN"); do
      v=$(measure "$PORT" "$u"); pu=$(cut -f1 <<<"$v"); ag=$(cut -f2 <<<"$v")
      best_pu="$pu"; best_ag="$ag"
      note "TP$tp users=$u run$r: per_user=${pu} tok/s aggregate=${ag} tok/s"
    done
  done
  docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; sleep 5
done
note "DONE -> $OUT"
