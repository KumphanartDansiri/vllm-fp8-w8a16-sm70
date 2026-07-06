#!/usr/bin/env bash
# Stage F — TP<=4 serving validation for the wired TurboMind SM70 FP8 engine.
#
# Path A (Codex): disposable JIT + PERSISTED torch_extensions cache — prove the wired
# backend behaves under real TP workers + vLLM serving BEFORE baking the engine into an image.
#
# Recipe:
#   1. PREWARM once: build the W8A16 kernel + the vendored TurboMind engine into a
#      host-mounted TORCH_EXTENSIONS_DIR, so the N TP workers REUSE the same .so
#      (no N-way JIT race on first forward).
#   2. Serve each model with VLLM_V100_FP8_ENGINE_JIT=1 + VLLM_V100_FP8_BACKEND=auto,
#      --enforce-eager (eager first; cudagraph is a later step), chunked-prefill LEFT ON
#      (disabling it is a proven V100 crash-causer), --skip-mm-profiling (VL towers).
#   3. Health -> one streamed generation -> assert coherent output AND that TurboMind
#      actually ENGAGED (banner present), not a silent fallback to ours.
#
# Models: Qwen3.5-27B-FP8 (dense block-FP8) then Qwen3.5-35B-A3B-FP8 (block-FP8 MoE).
# TP<=4 only (TP8 breaks MoE block-128 on I/tp=64 -> deferred). Default TP=2.
#
# Usage (from repo root or anywhere):  bash tools/turbomind_ab/tp_serve_validate.sh
#   env knobs: TP=4  GENTOK=128  MAXLEN=4096  PORT=8021  GPUMEM=0.90  ONLY=dense27b|moe35b
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # tools/turbomind_ab -> repo root

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
TP="${TP:-2}"
GPUS="${GPUS:-$(seq -s, 0 $((TP-1)))}"       # first TP GPUs
PORT="${PORT:-8021}"
MAXLEN="${MAXLEN:-4096}"
NS="${NS:-8}"
GENTOK="${GENTOK:-128}"
GPUMEM="${GPUMEM:-0.90}"
ONLY="${ONLY:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-results/tp_serve_validate_${STAMP}}"
TEXT_CACHE="$HOME/.cache/vllm-v100-021cu126-torchext"
mkdir -p "$OUT" "$TEXT_CACHE" \
  "$HOME/.cache/vllm-v100-021cu126-triton" \
  "$HOME/.cache/vllm-v100-021cu126-torch" \
  "$HOME/.cache/vllm-v100-021cu126-inductor"
SUMMARY="$OUT/SUMMARY.txt"; : > "$SUMMARY"
note(){ echo "[tp-serve] $*" | tee -a "$SUMMARY"; }

# ── clean-box guard: the target GPUs must be idle (shared box) ────────────────────
busy=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | wc -l)
if [[ "$busy" -gt 0 ]]; then
  note "REFUSING: $busy compute process(es) already on the box — free the GPUs first."
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null | tee -a "$SUMMARY"
  exit 2
fi
note "box clean; IMAGE=$IMAGE TP=$TP GPUS=$GPUS MODE=${MODE:-eager} BACKEND=${BACKEND:-auto} gentok=$GENTOK maxlen=$MAXLEN"

COMMON_MOUNTS=(
  -v /mnt/models:/mnt/models:ro -v "$PWD":/work -w /work -e PYTHONPATH=/work/src
  -v "$TEXT_CACHE:/root/.cache/torch_extensions"
  -v "$HOME/.cache/vllm-v100-021cu126-triton:/root/.triton"
  -v "$HOME/.cache/vllm-v100-021cu126-torch:/root/.cache/torch"
  -v "$HOME/.cache/vllm-v100-021cu126-inductor:/tmp/torchinductor_root"
)
# FREE_RAW=1 (default): drop the raw FP8 weight after prepare packs it, so a turbomind
# layer's footprint is packed-only (≈ the ours-FP8-resident size). Required for the MoE
# model to fit (keeping raw+packed ~doubles expert memory -> OOM). Validated safe: smoke
# cos=1.0000 on the packed path + dense TP serve. Set FREE_RAW=0 to keep raw (debug).
# ENGINE_JIT=1 (default): dev — JIT-build the engine into the persisted cache.
# ENGINE_JIT=0: use a baked image whose VLLM_V100_FP8_ENGINE_SO points at a prebuilt .so
# (production packaging; no runtime compile). ensure_engine() load_library()'s it.
TM_ENV=(
  -e VLLM_V100_FP8_ENGINE_JIT="${ENGINE_JIT:-1}" -e VLLM_V100_FP8_BACKEND="${BACKEND:-auto}"
  -e VLLM_V100_FP8_TM_FREE_RAW="${FREE_RAW:-1}"
  -e VLLM_V100_FP8_COALESCED_GEMV=1 -e VLLM_V100_FP8_COALESCED_UNROLL=4
  -e VLLM_V100_FP8_COALESCED_M_UNROLL=4 -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8
)

# MODE=eager (default): --enforce-eager. MODE=cudagraph: mode-0 + FULL_DECODE_ONLY capture
# (the V100-validated decode config; TRITON_ATTN is set above). Cudagraph gives the real
# decode throughput; eager is correctness/logistics only.
if [[ "${MODE:-eager}" == "cudagraph" ]]; then
  MODE_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
else
  MODE_ARGS=(--enforce-eager)
fi

# ── 1) PREWARM: build both extensions into the persisted cache (once) ─────────────
note "prewarm: building W8A16 kernel + TurboMind engine into $TEXT_CACHE ..."
PWLOG="$OUT/prewarm.log"
if docker run --rm --gpus "\"device=$GPUS\"" "${COMMON_MOUNTS[@]}" "${TM_ENV[@]}" "$IMAGE" \
    python3 -c "import fp8_w8a16_sm70.vllm_serve as _v; import fp8_w8a16_sm70.turbomind_fp8_backend as tb; ok=tb.ops_available(need_moe=True); print('PREWARM ops_available(moe)=%s' % ok); import sys; sys.exit(0 if ok else 3)" \
    >"$PWLOG" 2>&1; then
  note "prewarm OK: $(grep -o 'PREWARM ops_available(moe)=True' "$PWLOG" | head -1)"
else
  note "PREWARM FAILED (see $PWLOG):"; tail -5 "$PWLOG" | tee -a "$SUMMARY"; exit 3
fi

# ── 2+3) serve + generate + assert, per model ────────────────────────────────────
declare -a MODELS=(
  "dense27b|/mnt/models/Qwen/Qwen3.5-27B-FP8|q27b|TurboMind DENSE engaged"
  "moe35b|/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8|q35b|TurboMind MoE engaged"
)

overall_fail=0
for row in "${MODELS[@]}"; do
  IFS='|' read -r TAG MODEL SERVED NEED_BANNER <<<"$row"
  [[ -n "$ONLY" && "$ONLY" != "$TAG" ]] && continue
  [[ ! -d "$MODEL" ]] && { note "SKIP $TAG: model dir missing ($MODEL)"; continue; }

  CNAME="tpval_${TAG}_tp${TP}"
  SLOG="$OUT/serve_${TAG}.log"; SAMPLE="$OUT/sample_${TAG}.txt"
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  note "=== $TAG: serving $MODEL (TP=$TP, ${MODE:-eager}, backend=${BACKEND:-auto}) ==="

  docker run --rm -i --name "$CNAME" --gpus "\"device=$GPUS\"" \
    "${COMMON_MOUNTS[@]}" -p ${PORT}:${PORT} --shm-size=16g \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    "${TM_ENV[@]}" "$IMAGE" \
    python3 -m fp8_w8a16_sm70.vllm_serve \
      --model "$MODEL" --served-model-name "$SERVED" \
      --tensor-parallel-size "$TP" --dtype float16 --quantization fp8 \
      "${MODE_ARGS[@]}" \
      --max-model-len "$MAXLEN" --max-num-seqs "$NS" --skip-mm-profiling \
      --gpu-memory-utilization "$GPUMEM" \
      --host 0.0.0.0 --port "$PORT" \
    </dev/null >"$SLOG" 2>&1 &
  LPID=$!

  healthy=0; waited=0
  while (( waited < 2400 )); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
    kill -0 "$LPID" 2>/dev/null || { note "$TAG: server exited before healthy"; break; }
    sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "$TAG ...loading (${waited}s)"
  done
  if [[ "$healthy" != 1 ]]; then
    note "$TAG FAIL: never healthy"
    grep -nE "Error|Traceback|out of memory|RuntimeError|assert" "$SLOG" | head -20 | tee -a "$SUMMARY"
    docker stop "$CNAME" >/dev/null 2>&1 || true; wait "$LPID" 2>/dev/null || true
    overall_fail=1; continue
  fi
  note "$TAG healthy after ${waited}s"

  # one streamed generation; capture tok/s + repetition ratio + sha
  v=$(python3 - "$PORT" "$SERVED" "$GENTOK" "$SAMPLE" <<'PY'
import hashlib, json, sys, time, urllib.request
port, served, tok, sfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
prompt = ("Write a detailed, multi-section essay on the history, geography, economy, "
          "and culture of France. Use clear subsections with headings.")
body=json.dumps({"model":served,"stream":True,"max_tokens":tok,"temperature":0,"ignore_eos":True,
 "stream_options":{"include_usage":True},"messages":[{"role":"user","content":prompt}]}).encode()
req=urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body,
                           headers={"Content-Type":"application/json"})
t0=time.time(); tf=tl=None; chunks=0; ch=[]; usage=0
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line=raw.decode("utf-8","ignore").strip()
            if not line.startswith("data:"): continue
            data=line[5:].strip()
            if data=="[DONE]": break
            try: j=json.loads(data)
            except Exception: continue
            u=j.get("usage")
            if u and u.get("completion_tokens"): usage=int(u["completion_tokens"])
            c=j.get("choices") or []
            d=c[0]["delta"].get("content") if c else None
            if d:
                now=time.time(); tf=tf or now; tl=now; chunks+=1; ch.append(d)
except Exception as e:
    print(f"0\t0\t0\tERR:{type(e).__name__}"); sys.exit(0)
s="".join(ch); open(sfile,"w").write(s)
tok=usage or chunks
decode=(tok-1)/(tl-tf) if tf and tl and tl>tf and tok>1 else float("nan")
words=s.split()
rep=(max((words.count(w) for w in set(words)), default=0)/len(words)) if words else 1.0
print(f"{tok}\t{decode:.2f}\t{rep:.3f}\t{hashlib.sha256(s.encode()).hexdigest()[:12]}")
PY
)
  gtok=$(printf '%s' "$v" | cut -f1); gdec=$(printf '%s' "$v" | cut -f2)
  grep_rep=$(printf '%s' "$v" | cut -f3); gsha=$(printf '%s' "$v" | cut -f4)
  note "$TAG gen: tokens=$gtok decode=${gdec} tok/s rep=$grep_rep sha=$gsha"
  note "$TAG sample: $(head -c 160 "$SAMPLE" 2>/dev/null | tr '\n' ' ')"

  echo "--- $TAG backend decisions ---" | tee -a "$SUMMARY"
  grep -E "TurboMind FP8 engine:|TurboMind DENSE engaged|TurboMind MoE engaged" "$SLOG" | sort -u | tee -a "$SUMMARY"
  fb=$(grep -cE "backend=ours|fallback:" "$SLOG" 2>/dev/null || echo 0)

  # assertions: turbomind engaged for the expected branch + coherent, non-degenerate output
  engaged=0; grep -qE "$NEED_BANNER" "$SLOG" && engaged=1
  coherent=0
  awk -v r="$grep_rep" -v t="$gtok" 'BEGIN{exit !(r+0<0.5 && t+0>20)}' && coherent=1
  if [[ "$engaged" == 1 && "$coherent" == 1 ]]; then
    note "$TAG PASS (turbomind engaged; coherent output; ours-fallback lines=$fb)"
  else
    note "$TAG FAIL (engaged=$engaged coherent=$coherent rep=$grep_rep tok=$gtok) — see $SLOG"
    overall_fail=1
  fi

  docker stop "$CNAME" >/dev/null 2>&1 || true; wait "$LPID" 2>/dev/null || true
done

note "=========================================================="
if [[ "$overall_fail" == 0 ]]; then
  note "TP<=$TP SERVE VALIDATION: PASS  ->  next: bake engine into image + rerun WITHOUT JIT"
else
  note "TP<=$TP SERVE VALIDATION: FAIL  ->  inspect serve logs in $OUT"
fi
exit "$overall_fail"
