#!/usr/bin/env bash
# Qwen3.5 TRIAD compatibility + perf sweep on vLLM 0.19 (cu126) — the engine-match
# mirror of the 0.21 driver, so 0.19-vs-0.21 is a clean engine-only comparison.
#
# WHY THIS EXISTS (2026-06-26): the published Qwen3.5 triad (27B dense + 35B-A3B MoE,
# FP16/FP8/Int4) is measured on vLLM 0.21+cu126. To "fully test both families" we need
# the SAME triad on vLLM 0.19+cu126 — completing the 0.19-vs-0.21 engine picture and
# proving both models load + generate + serve on the older freeze-high engine.
#
# This is a near-verbatim clone of tools/tp_concurrency_sweep.sh (the 0.21 driver the
# in-flight GPTQ-021 re-run uses). EVERYTHING is held identical to that driver — same
# direct `docker run`, same measure() client + prompts, same per-precision FENV, same
# params (maxlen 4096, gen 512, ns=8, GPUMEM 0.85, NRUN 2, cudagraph mode=0+FULL_DECODE_ONLY,
# coalesced-ON for FP8) — so the ONLY difference from the 0.21 numbers is IMAGE (engine).
# Changed from that driver: IMAGE (0.19/cu126), CACHE_TAG, GPU base (4-7 not 0-3), PORT
# (8050 not 8040), container prefix (triad019), and an outer model×prec loop so it's one
# self-contained launch.
#
# RUNS ON GPUs 4,5,6,7 ONLY (GPU_BASE=4) so it coexists with the GPTQ-021 re-run on 0-3.
#   - TP4 -> GPUs 4,5,6,7 ; TP2 -> GPUs 4,5. Distinct port (8050) + container names + cache
#     dir from the GPTQ run, so no collision. Each load snapshots GPUs 0-3 and records
#     whether the GPTQ-021 run was still active (honest contention note for the fold).
#   - CAVEAT: while GPTQ-021 is still going (~next 30-40 min) the two servers share CPU /
#     PCIe, so the FIRST few cells run under mild contention; NRUN=2 catches gross drift and
#     the bulk of this sweep runs after GPTQ finishes. To wait for a clean box instead, set
#     WAIT_FOR=$HOME/gptq021_rerun.done (polls until that marker exists, then starts).
#
# CHUNKED PREFILL: matches the 0.21 driver (--no-enable-chunked-prefill) for an apples-to-
# apples comparison. That flag is a known crash-causer ONLY at the 122B-hybrid@28k+cudagraph
# config (NOT here: q27b/q35b @ 4096). Flip CHUNKED=1 to leave chunked prefill ON.
#
# Usage (launch and walk away):
#   tmux new-session -d -s triad019 'cd /home/kumphanartd/vllm-fp8-w8a16-sm70 && \
#     ./tools/triad_compat_vllm019.sh 2>&1 | tee ~/triad019.log; touch ~/triad019.done'
#
# Env: MODELS PRECS TPS USERS GPU_BASE IMAGE CACHE_TAG MAXLEN GENTOK NRUN GPUMEM PORT
#      HEALTH_TIMEOUT WAIT_FOR CHUNKED OUT
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

MODELS="${MODELS:-q27b q35b}"          # both Qwen3.5 families
PRECS="${PRECS:-fp16 fp8 int4}"        # full triad
TPS="${TPS:-4 2}"
USERS="${USERS:-1 2 4 8}"
GPU_BASE="${GPU_BASE:-4}"              # use the 4 GPUs the GPTQ-021 run isn't on
IMAGE="${IMAGE:-vllm-v100-py312:vllm019-cu126}"; CACHE_TAG="${CACHE_TAG:-019cu126}"
MAXLEN="${MAXLEN:-4096}"; GENTOK="${GENTOK:-512}"; NRUN="${NRUN:-2}"
GPUMEM="${GPUMEM:-0.85}"; HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"; PORT="${PORT:-8050}"
CHUNKED="${CHUNKED:-0}"                # 0 = match 0.21 driver (--no-enable-chunked-prefill)
WAIT_FOR="${WAIT_FOR:-}"              # if set, poll until this marker file exists, then start
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="${OUT:-results/triad019_${STAMP}}"
SERVED="triad019"; mkdir -p "$OUT"; SUMMARY="$OUT/SUMMARY.txt"; : > "$SUMMARY"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note(){ echo "[triad019] $*" | tee -a "$SUMMARY"; }

# Qwen3.5 triad model paths (3.5, NOT 3.6 — these are the published-triad checkpoints).
model_path(){
  case "$1:$2" in
    q27b:fp16) echo /mnt/models/Qwen/Qwen3.5-27B ;;
    q27b:fp8)  echo /mnt/models/Qwen/Qwen3.5-27B-FP8 ;;
    q27b:int4) echo /mnt/models/Qwen/Qwen3.5-27B-GPTQ-Int4 ;;
    q35b:fp16) echo /mnt/models/Qwen/Qwen3.5-35B-A3B ;;
    q35b:fp8)  echo /mnt/models/Qwen/Qwen3.5-35B-A3B-FP8 ;;
    q35b:int4) echo /mnt/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 ;;
    q122b:fp8) echo /mnt/models/Qwen/Qwen3.5-122B-A10B-FP8 ;;        # 122B dual-engine consistency (TP8; set GPU_BASE=0)
    q122b:int4) echo /mnt/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 ;;
    *) echo "" ;;
  esac
}

gpus_for(){ local n="$1" o="" i; for ((i=0;i<n;i++)); do o+="${o:+,}$((GPU_BASE+i))"; done; echo "$o"; }
clean_guard(){ local g="$1" any=0 i used; IFS=',' read -ra a <<<"$g"
  for i in "${a[@]}"; do used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null|awk '{print $1+0}')
    [[ "${used:-9999}" -gt 2000 ]] && any=1; done; [[ "$any" -eq 0 ]]; }
# Honest contention record: are GPUs 0-3 (the GPTQ-021 run) still busy right now?
gptq_active(){ local i used; for i in 0 1 2 3; do
    used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null|awk '{print $1+0}')
    [[ "${used:-0}" -gt 2000 ]] && { echo yes; return; }; done; echo no; }

if [[ -n "$WAIT_FOR" ]]; then
  note "WAIT_FOR set — polling for $WAIT_FOR before starting (clean-box mode)..."
  while [[ ! -f "$WAIT_FOR" ]]; do sleep 30; done
  note "marker $WAIT_FOR present — starting."
fi

note "TRIAD-on-0.19 sweep START $(date -u +%FT%TZ)"
note "image=$IMAGE models='$MODELS' precs='$PRECS' TPS='$TPS' users='$USERS' gpus=$(gpus_for 4) port=$PORT"
note "maxlen=$MAXLEN gentok=$GENTOK nrun=$NRUN gpumem=$GPUMEM chunked_prefill=$([[ $CHUNKED == 1 ]] && echo ON || echo OFF)"

# Identical concurrency client to tp_concurrency_sweep.sh -> directly comparable numbers.
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

# Per-precision env — IDENTICAL to tp_concurrency_sweep.sh so the FP8/Int4/FP16 paths are
# configured the same as on 0.21 (CT_* vars are inert for Qwen block-FP8 but kept for byte
# parity with the 0.21 run). Sets the FENV array.
set_fenv(){ local p="$1"
  if [[ "$p" == fp8 ]]; then
    FENV=(-e VLLM_V100_FP8_COALESCED_GEMV=1 -e VLLM_V100_FP8_COALESCED_UNROLL=4
          -e VLLM_V100_FP8_COALESCED_M_UNROLL=4 -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8
          -e VLLM_V100_FP8_MOE_W13_COALESCED=1 -e VLLM_V100_FP8_MOE_FALLBACK=1
          -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1 -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=512
          -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1 -e VLLM_V100_CT_FP8_RESIDENT=1
          -e VLLM_V100_CT_MOE_W13_RESIDENT=1 -e VLLM_V100_CT_MOE_W2_GROUPED=1
          -e VLLM_V100_CT_MOE_W13_COALESCED=1)
    QUANT=(--quantization fp8)
  elif [[ "$p" == int4 ]]; then
    QUANT=(--quantization gptq); FENV=(-e VLLM_V100_FP8_COALESCED_GEMV=0)   # stock GPTQ (plugin inert)
  else
    QUANT=(); FENV=(-e VLLM_V100_FP8_COALESCED_GEMV=0)   # FP16 — Volta MoE BLOCK_K patch is default-ON
  fi
}

CHUNK_FLAG=(--no-enable-chunked-prefill)
[[ "$CHUNKED" == 1 ]] && CHUNK_FLAG=()

for model in $MODELS; do
 for prec in $PRECS; do
  MODEL="$(model_path "$model" "$prec")"
  [[ -n "$MODEL" && -f "$MODEL/config.json" ]] || { note "### $model:$prec — MISSING MODEL ($MODEL) — skip"; continue; }
  set_fenv "$prec"
  note "######## $model $prec  ($MODEL) ########"
  for tp in $TPS; do
    gpus=$(gpus_for "$tp"); cname="triad019_${model}_${prec}_tp${tp}"; slog="$OUT/${model}_${prec}_tp${tp}_serve.log"
    if ! clean_guard "$gpus"; then note "$model $prec TP$tp: SKIP (GPUs $gpus busy)"; continue; fi
    note "=== $model $prec TP$tp (gpus=$gpus) loading === [gptq021 active on 0-3: $(gptq_active)]"
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
        --gpu-memory-utilization "$GPUMEM" "${CHUNK_FLAG[@]}" --host 0.0.0.0 --port "$PORT" \
      </dev/null >"$slog" 2>&1 &
    lpid=$!; healthy=0; waited=0
    while (( waited < HEALTH_TIMEOUT )); do
      curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
      kill -0 "$lpid" 2>/dev/null || break
      sleep 10; waited=$((waited+10))
    done
    if [[ "$healthy" != 1 ]]; then
      if grep -qiE "out of memory|OOM|CUDA out of memory|No available memory" "$slog"; then
        note "$model $prec TP$tp: INFEASIBLE (OOM — does not fit at TP$tp)"
      else
        note "$model $prec TP$tp: INFEASIBLE/FAIL (never healthy; see $slog)"
        grep -nE "Error|Traceback" "$slog" | head -3 | tee -a "$SUMMARY"
      fi
      docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; continue
    fi
    # warm (JIT + cudagraph capture)
    curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" >/dev/null 2>&1 || true
    for u in $USERS; do
      for r in $(seq 1 "$NRUN"); do
        v=$(measure "$PORT" "$u"); pu=$(cut -f1 <<<"$v"); ag=$(cut -f2 <<<"$v")
        note "$model $prec TP$tp users=$u run$r: per_user=${pu} tok/s aggregate=${ag} tok/s"
      done
    done
    docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; sleep 5
  done
 done
done
note "DONE -> $OUT  ($(date -u +%FT%TZ))"
