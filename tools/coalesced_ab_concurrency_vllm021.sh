#!/usr/bin/env bash
# Coalesced FP8 ON/OFF × concurrency A/B for the FORCED-FP8 flagships (TP=8).
# Stock vLLM 0.21 sm_70 (image vllm-v100:vllm021-cu126), package mounted at /work
# (PYTHONPATH=/work/src), kernel JIT-compiled in-container.
#
# WHAT THIS ANSWERS (the 2026-06-20 "measure, don't build" step): with the
# branchless converter promoted canonically, does the coalesced decode path lift
# the large-MoE flagships across concurrency C1..C8 — and where (if anywhere) does
# the M=8/C8 dense gap still bite at TP=8 where all-reduce tempers everything?
# These two models CAN'T fit FP16 on 8×V100-32GB, so there is no FP16 arm: FP8 is
# the only way to serve them at all. The A/B is coalesced-ON vs coalesced-OFF
# (same FP8 weights, same residency, same everything else) → isolates the kernel.
#
# TWO MODELS, picked by MODEL_KEY (run it twice — 122B first, then GLM-Air):
#   q122b : Qwen3.5-122B-A10B-FP8  — Qwen BLOCK-FP8 (quant_method=fp8).
#           Routes attn through _v100_fp8_gemm and MoE w13 through
#           _our_moe_apply_grouped. A/B toggle = VLLM_V100_FP8_COALESCED_GEMV +
#           VLLM_V100_FP8_MOE_W13_COALESCED. (CT_* env is IRRELEVANT here.)
#   glm   : GLM-4.5-Air-FP8 — compressed-tensors FP8. Routes MoE w13 through the
#           CT path. A/B toggle = VLLM_V100_FP8_COALESCED_GEMV +
#           VLLM_V100_CT_MOE_W13_COALESCED. (no --quantization; auto from config)
# Baseline (coal_off, cudagraph): 122B ~34.6 tok/s @C1; GLM-Air ~30.7 → coal_on
# expectation ~56.6 from the GLM session (1.84×), TP8-all-reduce-tempered.
#
# Per arm: clean-box guard → serve → health-wait → warm → COHERENCE probe (1 stream,
# repetition ratio + sample text; guards "fast but garbage") → concurrency sweep
# (per-user + aggregate decode tok/s) → coalesced-ENGAGEMENT proof from the server
# log (banner must appear ONLY in coal_on) → teardown. Results land durable under
# results/coal_ab_<key>_<stamp>/ with a final comparison table in SUMMARY.txt.
#
# Usage:
#   MODEL_KEY=q122b bash tools/coalesced_ab_concurrency_vllm021.sh
#   MODEL_KEY=glm   bash tools/coalesced_ab_concurrency_vllm021.sh
#   ONLY=coal_on MODEL_KEY=q122b ... bash ...      # just one arm
#   USERS="1 2 4 8 16" GENTOK=512 NRUN=3 MODEL_KEY=q122b bash ...
# Env: MODEL_KEY ONLY USERS GENTOK NRUN TP IMAGE MAXLEN GPUMEM HEALTH_TIMEOUT PORT
#      MODEL_OVERRIDE CACHE_TAG
#
# Shared GPU box: this is a HEAVY TP=8 serve (loads the whole model on GPUs 0-7,
# twice — once per arm). Run on a clean box; the clean-box guard refuses if any of
# GPUs 0-7 has >2 GB used. Launch it yourself and collect the SUMMARY.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

MODEL_KEY="${MODEL_KEY:-q122b}"
TP="${TP:-8}"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
CACHE_TAG="${CACHE_TAG:-021cu126}"
USERS="${USERS:-1 2 4 8}"
GENTOK="${GENTOK:-256}"
NRUN="${NRUN:-2}"
MAXLEN="${MAXLEN:-4096}"
GPUMEM="${GPUMEM:-0.88}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
PORT="${PORT:-8044}"
ONLY="${ONLY:-}"
SERVED="coalab"

# --- model registry + per-model routing (the critical correctness bit) -----------
case "$MODEL_KEY" in
  q122b)
    MODEL=/mnt/models/Qwen/Qwen3.5-122B-A10B-FP8
    QUANT=(--quantization fp8)                 # Qwen BLOCK-FP8 needs the explicit flag
    W13_ON="VLLM_V100_FP8_MOE_W13_COALESCED=1"  # toggled per arm
    W13_OFF="VLLM_V100_FP8_MOE_W13_COALESCED=0"
    BASELINE="~34.6 tok/s @C1 (cudagraph)" ;;
  glm)
    MODEL=/mnt/models/zai-org/GLM-4.5-Air-FP8
    QUANT=()                                    # compressed-tensors: auto from config
    W13_ON="VLLM_V100_CT_MOE_W13_COALESCED=1"
    W13_OFF="VLLM_V100_CT_MOE_W13_COALESCED=0"
    BASELINE="~30.7 tok/s @C1 -> GLM-session coal_on ~56.6 (1.84x), TP8-tempered" ;;
  *) echo "unknown MODEL_KEY='$MODEL_KEY' (use q122b | glm)"; exit 1 ;;
esac
MODEL="${MODEL_OVERRIDE:-$MODEL}"

# max-num-seqs must cover the widest concurrency we drive, else extra streams queue
# (serialize) and the per-user number is wrong. Pick max(USERS, 8).
MAXSEQS=8; for u in $USERS; do (( u > MAXSEQS )) && MAXSEQS="$u"; done

STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="results/coal_ab_${MODEL_KEY}_${STAMP}"
mkdir -p "$OUT"; SUMMARY="$OUT/SUMMARY.txt"; : > "$SUMMARY"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note(){ echo "[coal-ab] $*" | tee -a "$SUMMARY"; }

# Base FP8 env shared by BOTH arms (residency + MoE plumbing). Only the COALESCED
# flags differ between arms; everything else is held constant so the A/B is clean.
BASE_FP8_ENV=(
  -e VLLM_V100_CT_FP8_RESIDENT=1
  -e VLLM_V100_CT_MOE_W13_RESIDENT=1
  -e VLLM_V100_CT_MOE_W13_FREE_FP16=1   # frees transient FP16 w13 -> ~8.3GB/GPU; WITHOUT
                                        # this GLM-Air keeps FP16-resident experts -> OOM@TP8
  -e VLLM_V100_CT_MOE_W2_GROUPED=1
  -e VLLM_V100_FP8_MOE_FALLBACK=1
  -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1
  -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=512
  -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1
  -e VLLM_V100_FP8_COALESCED_UNROLL=4
  -e VLLM_V100_FP8_COALESCED_M_UNROLL=4
  -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8
  -e FP8_WMMA_COUNTER_LOG_EVERY=200   # make the attn variant-counts banner print in a short bench
)

# arm-label | COALESCED_GEMV | <model-specific W13 flag string>
ARMS=(
  "coal_off|0|$W13_OFF"
  "coal_on|1|$W13_ON"
)

gpus_for(){ local n="$1" o="" i; for ((i=0;i<n;i++)); do o+="${o:+,}$i"; done; echo "$o"; }
clean_guard(){ local g="$1" any=0 i used; IFS=',' read -ra a <<<"$g"
  for i in "${a[@]}"; do
    used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null|awk '{print $1+0}')
    [[ "${used:-9999}" -gt 2000 ]] && any=1
  done; [[ "$any" -eq 0 ]]; }

# Single-stream coherence probe: returns "OK|BAD<TAB>ctok<TAB>rep<TAB>ttft<TAB>tok/s<TAB>snippet"
coherence_probe(){
  python3 - "$1" "$SERVED" <<'PY'
import sys,json,re,time,urllib.request
port,served=sys.argv[1],sys.argv[2]
body=json.dumps({"model":served,"stream":True,"max_tokens":160,"temperature":0,"ignore_eos":True,
  "stream_options":{"include_usage":True},
  "messages":[{"role":"user","content":"Write a detailed multi-paragraph essay about the history, geography, and culture of France."}]}).encode()
rq=urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
t0=time.time();tf=tl=None;n=0;ch=[];ut=0
try:
  with urllib.request.urlopen(rq,timeout=900) as r:
    for raw in r:
      L=raw.decode("utf-8","ignore").strip()
      if not L.startswith("data:"):continue
      d=L[5:].strip()
      if d=="[DONE]":break
      try:j=json.loads(d)
      except:continue
      u=j.get("usage")
      if u and u.get("completion_tokens"):ut=int(u["completion_tokens"])
      c=j.get("choices") or [];dl=c[0]["delta"].get("content") if c else None
      if dl:
        t=time.time()
        if tf is None:tf=t
        tl=t;n+=1;ch.append(dl)
  s="".join(ch).strip();w=s.split()
  rep=(max((w.count(x) for x in set(w)),default=0)/len(w)) if w else 1.0
  ttft=(tf-t0) if tf else float("nan")
  dt=(tl-tf) if (tf and tl and n>1) else float("nan")
  mt=ut or n
  tps=((mt-1)/dt) if (dt and dt>0) else float("nan")
  ok=bool(s) and n>=20 and rep<0.35
  print(("OK" if ok else "BAD")+f"\t{mt}\t{rep:.2f}\t{ttft:.2f}\t{tps:.2f}\t"+re.sub(r'\s+',' ',s)[:120])
except Exception as e:
  print(f"BAD\t0\t1.0\tnan\tnan\tprobe-error: {e}")
PY
}

# Concurrency measure: fire $2 simultaneous streams -> "per_user_mean<TAB>aggregate"
measure(){
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
  try:
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
  except Exception:
    res[u]=float("nan")
ts=[threading.Thread(target=one,args=(u,)) for u in range(nu)]
for t in ts:t.start()
for t in ts:t.join()
ok=[x for x in res if x==x]
print(f"{(sum(ok)/len(ok)):.2f}\t{sum(ok):.2f}" if ok else "nan\tnan")
PY
}

run_arm(){
  local label="$1" coalesced="$2" w13kv="$3" gpus cname slog
  gpus=$(gpus_for "$TP"); cname="coalab_${MODEL_KEY}_${label}"; slog="$OUT/${label}_serve.log"
  [[ -f "$MODEL/config.json" ]] || { note "$label: SKIP (missing $MODEL)"; return; }
  clean_guard "$gpus" || { note "$label: SKIP (GPUs $gpus busy — clean the box first)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$SUMMARY"; return; }

  note "=== arm=$label  COALESCED_GEMV=$coalesced  $w13kv  (model=$MODEL_KEY TP=$TP gpus=$gpus) ==="
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
    -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    "${BASE_FP8_ENV[@]}" \
    -e VLLM_V100_FP8_COALESCED_GEMV="$coalesced" \
    -e "$w13kv" \
    "$IMAGE" \
    python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
      --tensor-parallel-size "$TP" --dtype float16 ${QUANT[@]+"${QUANT[@]}"} \
      --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --skip-mm-profiling \
      --disable-custom-all-reduce \
      --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill \
      --host 0.0.0.0 --port "$PORT" \
    </dev/null >"$slog" 2>&1 &
  local lpid=$! healthy=0 waited=0
  while (( waited < HEALTH_TIMEOUT )); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
    kill -0 "$lpid" 2>/dev/null || { note "  server exited before healthy"; break; }
    sleep 10; waited=$((waited+10)); (( waited % 120 == 0 )) && note "  ...loading $label (${waited}s)"
  done
  if [[ "$healthy" != 1 ]]; then
    if grep -qiE "out of memory|CUDA out of memory|No available memory" "$slog"; then
      note "$label: INFEASIBLE (OOM at TP=$TP) — see $slog"
    else
      note "$label: FAIL (never healthy) — see $slog"
      grep -nE "Error|Traceback|no kernel image|assert" "$slog" | head -6 | tee -a "$SUMMARY"
    fi
    docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; return
  fi

  # warm: JIT + cudagraph capture
  curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" >/dev/null 2>&1 || true

  # coherence first — a fast-but-garbage arm must be caught before we trust its tok/s
  local cv tag rep snip
  cv=$(coherence_probe "$PORT"); tag=$(cut -f1 <<<"$cv"); rep=$(cut -f3 <<<"$cv"); snip=$(cut -f6- <<<"$cv")
  note "  coherence: $tag rep=$rep  \"$snip\""

  # concurrency sweep
  local u r v pu ag
  for u in $USERS; do
    pu=""; ag=""
    for r in $(seq 1 "$NRUN"); do
      v=$(measure "$PORT" "$u"); pu=$(cut -f1 <<<"$v"); ag=$(cut -f2 <<<"$v")
      note "  C$u run$r: per_user=${pu} tok/s  aggregate=${ag} tok/s"
    done
    echo "$label C$u $pu $ag" >> "$OUT/_grid.txt"   # last run wins (steady-state)
  done

  # PROVE which kernel actually ran. The w13 marker "grouped COALESCED" is printed
  # once on first engagement by BOTH the 122B path ([V100-FP8-MOE-GROUPED]) and the
  # GLM CT path ([serve_fp8_v100 ct-moe-w13]) — unambiguous, only when engaged. The
  # attn split shows in the variant-counts banner (verbatim, so the user sees the
  # "Coalesced GEMV" count vs A.3/WMMA rather than a presence-only grep).
  local w13_hits banner
  w13_hits=$(grep -c "grouped COALESCED" "$slog" 2>/dev/null | head -1)
  banner=$(grep "kernel variant counts" "$slog" 2>/dev/null | tail -1)
  note "  engagement(w13): grouped-COALESCED-hits=${w13_hits:-0}  (expect >0 in coal_on, 0 in coal_off)"
  note "  engagement(attn): ${banner:-<no variant-counts banner printed>}"
  note "  stopping $cname"; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; sleep 5
}

main(){
  note "Coalesced FP8 ON/OFF x concurrency A/B [vLLM 0.21 sm_70]"
  note "model=$MODEL_KEY ($(basename "$MODEL")) TP=$TP USERS='$USERS' gentok=$GENTOK nrun=$NRUN maxseqs=$MAXSEQS"
  note "baseline: $BASELINE"
  note "stamp=$STAMP out=$OUT"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }
  local row label coalesced w13kv
  for row in "${ARMS[@]}"; do
    IFS='|' read -r label coalesced w13kv <<<"$row"
    [[ -n "$ONLY" && "$label" != "$ONLY" ]] && continue
    run_arm "$label" "$coalesced" "$w13kv"
    echo "" | tee -a "$SUMMARY"
  done

  # final comparison table (per_user + aggregate, off vs on, speedup)
  if [[ -f "$OUT/_grid.txt" ]]; then
    note "==== COMPARISON ($MODEL_KEY TP=$TP) — per_user tok/s | aggregate tok/s ===="
    note "$(printf '%-4s %18s %18s %10s' 'C' 'coal_off(per/agg)' 'coal_on(per/agg)' 'per-speedup')"
    for u in $USERS; do
      local offp offa onp ona spd
      offp=$(awk -v u="C$u" '$1=="coal_off"&&$2==u{print $3}' "$OUT/_grid.txt" | tail -1)
      offa=$(awk -v u="C$u" '$1=="coal_off"&&$2==u{print $4}' "$OUT/_grid.txt" | tail -1)
      onp=$(awk  -v u="C$u" '$1=="coal_on" &&$2==u{print $3}' "$OUT/_grid.txt" | tail -1)
      ona=$(awk  -v u="C$u" '$1=="coal_on" &&$2==u{print $4}' "$OUT/_grid.txt" | tail -1)
      spd=$(awk -v a="${onp:-nan}" -v b="${offp:-nan}" 'BEGIN{if(b+0>0&&a==a)printf "%.2fx",a/b;else print "-"}')
      note "$(printf '%-4s %18s %18s %10s' "C$u" "${offp:--}/${offa:--}" "${onp:--}/${ona:--}" "$spd")"
    done
  fi
  note "DONE. READ: coal_on per_user should beat coal_off at low C and stay >= at C8;"
  note "  aggregate should rise with C; coalesced-banner-hits>0 ONLY in coal_on. Full logs in $OUT/."
}
main "$@"
