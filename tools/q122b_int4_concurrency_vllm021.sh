#!/usr/bin/env bash
# 122B-A10B-GPTQ-Int4 concurrency sweep on vLLM 0.21 — the ENGINE-MATCHED control
# for the FP8-coalesced A/B (tools/coalesced_ab_concurrency_vllm021.sh).
#
# WHY: the prior Int4 sweep ran on vLLM 0.18 while the FP8 sweep ran on 0.21, and
# 0.21 is ~9-15% slower generally (Ch1) and ~40% slower on the 122B specifically
# (vllm021 regression note). That confound could explain or invert the Int4 lead.
# This run puts Int4 on the SAME 0.21 stack we ship FP8 on, with the SAME serve
# flags and the SAME (byte-identical) concurrency client, so the ONLY difference
# vs results/coal_ab_q122b_* is the quantization format. Stock vLLM GPTQ (no plugin
# env) — on V100 gptq_marlin is sm80+ so it uses gptq_gemm (coherent, verified).
#
# Single config (Int4 has no coalesced toggle — it's stock vLLM). C1/C2/C4/C8,
# per-user + aggregate decode tok/s, coherence gate. Durable -> results/q122b_int4_*.
#
# Usage:  bash tools/q122b_int4_concurrency_vllm021.sh
# Env: USERS GENTOK NRUN TP IMAGE MAXLEN GPUMEM HEALTH_TIMEOUT PORT MODEL CACHE_TAG

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

MODEL="${MODEL:-/mnt/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4}"
TP="${TP:-8}"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
CACHE_TAG="${CACHE_TAG:-021cu126}"
USERS="${USERS:-1 2 4 8}"
GENTOK="${GENTOK:-256}"
NRUN="${NRUN:-2}"
MAXLEN="${MAXLEN:-4096}"
GPUMEM="${GPUMEM:-0.88}"            # match the FP8 sweep (0.88), not the prior 0.85
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
PORT="${PORT:-8046}"
SERVED="q122bint4"

MAXSEQS=8; for u in $USERS; do (( u > MAXSEQS )) && MAXSEQS="$u"; done
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="results/q122b_int4_v021_${STAMP}"
mkdir -p "$OUT"; SUMMARY="$OUT/SUMMARY.txt"; : > "$SUMMARY"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note(){ echo "[int4-v021] $*" | tee -a "$SUMMARY"; }

gpus_for(){ local n="$1" o="" i; for ((i=0;i<n;i++)); do o+="${o:+,}$i"; done; echo "$o"; }
clean_guard(){ local g="$1" any=0 i used; IFS=',' read -ra a <<<"$g"
  for i in "${a[@]}"; do
    used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null|awk '{print $1+0}')
    [[ "${used:-9999}" -gt 2000 ]] && any=1
  done; [[ "$any" -eq 0 ]]; }

# --- coherence probe + concurrency client: byte-identical to coalesced_ab -----
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

main(){
  note "122B-A10B-GPTQ-Int4 concurrency sweep [ENGINE-MATCHED to FP8: vLLM 0.21]"
  note "model=$(basename "$MODEL") TP=$TP USERS='$USERS' gentok=$GENTOK nrun=$NRUN maxseqs=$MAXSEQS gpumem=$GPUMEM image=$IMAGE"
  note "stamp=$STAMP out=$OUT"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }
  [[ -f "$MODEL/config.json" ]] || { note "MISSING MODEL $MODEL"; exit 1; }
  local gpus cname slog; gpus=$(gpus_for "$TP"); cname="q122bint4_v021"; slog="$OUT/serve.log"
  clean_guard "$gpus" || { note "SKIP (GPUs $gpus busy — clean the box first)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$SUMMARY"; exit 1; }

  note "=== loading (TP=$TP gpus=$gpus, stock GPTQ, --skip-mm-profiling) ==="
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
    "$IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SERVED" \
      --tensor-parallel-size "$TP" --dtype float16 --quantization gptq \
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
    sleep 10; waited=$((waited+10)); (( waited % 120 == 0 )) && note "  ...loading (${waited}s)"
  done
  if [[ "$healthy" != 1 ]]; then
    if grep -qiE "out of memory|CUDA out of memory|No available memory" "$slog"; then note "INFEASIBLE (OOM at TP=$TP) — $slog"
    else note "FAIL (never healthy) — $slog"; grep -nE "Error|Traceback|no kernel image|assert" "$slog" | head -6 | tee -a "$SUMMARY"; fi
    docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; exit 1
  fi
  curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" >/dev/null 2>&1 || true

  local cv tag rep snip; cv=$(coherence_probe "$PORT"); tag=$(cut -f1 <<<"$cv"); rep=$(cut -f3 <<<"$cv"); snip=$(cut -f6- <<<"$cv")
  note "coherence: $tag rep=$rep  \"$snip\""

  echo "users,run,per_user_tok_s,aggregate_tok_s" > "$OUT/results.csv"
  local u r v pu ag
  for u in $USERS; do
    pu=""; ag=""
    for r in $(seq 1 "$NRUN"); do
      v=$(measure "$PORT" "$u"); pu=$(cut -f1 <<<"$v"); ag=$(cut -f2 <<<"$v")
      note "C$u run$r: per_user=${pu} tok/s  aggregate=${ag} tok/s"
      echo "$u,$r,$pu,$ag" >> "$OUT/results.csv"
    done
  done
  note "stopping $cname"; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true

  note "==== 122B Int4 (vLLM 0.21, matched) vs FP8-coalesced (0.21) ===="
  note "$(printf '%-4s %14s %18s' 'C' 'Int4 per/agg' 'FP8coal per/agg(ref)')"
  # FP8-coalesced 0.21 reference (results/coal_ab_q122b_*, coal_on)
  declare -A FP8=( [1]="57.02/57.02" [2]="46.33/92.66" [4]="40.86/163.44" [8]="31.12/248.96" )
  for u in $USERS; do
    local ip ia; ip=$(awk -F, -v u="$u" '$1==u{p=$3;a=$4} END{print p}' "$OUT/results.csv")
    ia=$(awk -F, -v u="$u" '$1==u{a=$4} END{print a}' "$OUT/results.csv")
    note "$(printf '%-4s %14s %18s' "C$u" "${ip:--}/${ia:--}" "${FP8[$u]:-?}")"
  done
  note "DONE -> $OUT. Both on vLLM 0.21 now: the format delta is engine-controlled."
}
main "$@"
