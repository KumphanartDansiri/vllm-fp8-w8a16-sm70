#!/usr/bin/env bash
# Performance Experiment v2 — ONE CELL (model x precision x engine).
# See docs/PERF_EXPERIMENT_V2.md. Runs, in one serve (cudagraph, skip-mm for VL):
#   A. correctness battery (5 tests @ C1, gated -> quality_status pass/suspect/fail)
#   B. TTFT short(~2k) + long(~24k) input; long also FA-on for FA-eligible models
#   C. decode C1/C2/C4/C8 via Test-4 long-form (ignore_eos, 256 tok, topic-rotated, median of NRUN)
# Durable -> results/perf_v2_<key>_<prec>_<engine>_<stamp>/. CSV carries quality_status.
#
# Usage:
#   MODEL_KEY=q27b PREC=fp8 ENGINE=021 bash tools/perf_v2_cell.sh
# Env: MODEL_KEY PREC ENGINE TP USERS GENTOK NRUN TTFT_REPS FA_ARM PORT GPUMEM HEALTH_TIMEOUT

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

MODEL_KEY="${MODEL_KEY:-q27b}"; PREC="${PREC:-fp8}"; ENGINE="${ENGINE:-021}"
USERS="${USERS:-1 2 4 8}"; GENTOK="${GENTOK:-256}"; NRUN="${NRUN:-5}"; TTFT_REPS="${TTFT_REPS:-3}"
GPUMEM="${GPUMEM:-0.88}"; HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"; PORT="${PORT:-8050}"
FA_DIR="${FA_DIR:-/home/kumphanartd/flash-attention-v100}"
TTFT_ONLY="${TTFT_ONLY:-0}"   # 1 = reconcile run: re-measure TTFT only (decode already valid)
SERVED="perfv2"

# ---- registry: path | VL(skip-mm) | min_tp | tf5 | fa_eligible -------------------
case "$MODEL_KEY" in
  q27b)  MPATH_fp8=/mnt/models/Qwen/Qwen3.6-27B-FP8;            MPATH_fp16=/mnt/models/Qwen/Qwen3.6-27B;            VL=0; MINTP=2; TF5=0; FAEL=1 ;;
  q35b)  MPATH_fp8=/mnt/models/Qwen/Qwen3.6-35B-A3B-FP8;        MPATH_fp16=/mnt/models/Qwen/Qwen3.6-35B-A3B;        VL=0; MINTP=4; TF5=0; FAEL=1 ;;
  q122b) MPATH_fp8=/mnt/models/Qwen/Qwen3.5-122B-A10B-FP8;      MPATH_int4=/mnt/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4; VL=1; MINTP=8; TF5=0; FAEL=1 ;;
  glm)   MPATH_fp8=/mnt/models/zai-org/GLM-4.5-Air-FP8;         VL=0; MINTP=8; TF5=0; FAEL=1 ;;
  g31b)  MPATH_fp8=/mnt/models/RedHatAI/gemma-4-31B-it-FP8-Dynamic;     MPATH_fp16=/mnt/models/google/gemma-4-31B-it;     VL=1; MINTP=4; TF5=1; FAEL=0 ;;
  g26b)  MPATH_fp8=/mnt/models/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic; MPATH_fp16=/mnt/models/google/gemma-4-26B-A4B-it; VL=1; MINTP=4; TF5=1; FAEL=0 ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY"; exit 1 ;;
esac
TP="${TP:-$MINTP}"
FA_ARM="${FA_ARM:-$FAEL}"   # default: on for FA-eligible, and only used on TTFT-long

# ---- precision -> model path + quant flags ---------------------------------------
case "$PREC" in
  fp8)  MODEL="${MPATH_fp8:-}";  QUANT=() ;;
  fp16) MODEL="${MPATH_fp16:-}"; QUANT=() ;;
  int4) MODEL="${MPATH_int4:-}"; QUANT=(--quantization gptq) ;;
  *) echo "unknown PREC=$PREC"; exit 1 ;;
esac
[[ -n "$MODEL" ]] || { echo "$MODEL_KEY:$PREC has no model path (infeasible combo)"; exit 2; }
# Per-PRECISION min-TP (a precision rule, not a model special-case): FP16 is 2 bytes/
# param, so dense ~30-50B models need >= TP4 to fit on 32GB cards (q27b-FP16 ~54GB OOMs
# at the fp8 min of TP2). FP8/Int4 keep their per-model MINTP.
[[ "$PREC" == fp16 && "${TP}" -lt 4 ]] && TP=4
# Qwen block-FP8 needs explicit --quantization fp8; GLM/Gemma are compressed-tensors (auto)
if [[ "$PREC" == fp8 && ( "$MODEL_KEY" == q27b || "$MODEL_KEY" == q35b || "$MODEL_KEY" == q122b ) ]]; then
  QUANT=(--quantization fp8)
fi

# ---- engine -> image -------------------------------------------------------------
if [[ "$ENGINE" == 019 ]]; then
  IMAGE="${IMAGE:-vllm-v100-py312:vllm019-cu126}"; CACHE_TAG="${CACHE_TAG:-019cu126}"
  [[ "$TF5" == 1 ]] && { IMAGE="vllm-v100-py312:vllm019-tf5"; CACHE_TAG="019tf5"; }
  EXTRA_SERVE=()
elif [[ "$ENGINE" == 021 ]]; then
  IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"; CACHE_TAG="${CACHE_TAG:-021cu126}"
  # Gemma-4 on 0.21 (transformers 5.x IS in this image): max_num_batched_tokens must be
  # >= max_model_len when chunked-prefill is off (vLLM pydantic check), AND >=2496 for the
  # vision path. We serve --max-model-len 32768, so use 32768 (the 2496 value false-SKIP'd
  # all Gemma-021 cells with "MNBT < max_model_len").
  EXTRA_SERVE=(); [[ "$TF5" == 1 ]] && EXTRA_SERVE=(--max-num-batched-tokens 32768)
else echo "unknown ENGINE=$ENGINE (use 019|021)"; exit 1; fi

# ---- FP8 plugin env (coalesced + vectorized dequant ship by default) -------------
FP8_ENV=()
if [[ "$PREC" == fp8 ]]; then
  FP8_ENV=(
    -e VLLM_V100_CT_FP8_RESIDENT=1 -e VLLM_V100_CT_MOE_W13_RESIDENT=1
    -e VLLM_V100_CT_MOE_W13_FREE_FP16=1 -e VLLM_V100_CT_MOE_W2_GROUPED=1
    -e VLLM_V100_CT_MOE_W13_COALESCED=1
    -e VLLM_V100_FP8_MOE_FALLBACK=1 -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1
    -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=512 -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1
    -e VLLM_V100_FP8_MOE_W13_COALESCED=1
    -e VLLM_V100_FP8_COALESCED_GEMV=1 -e VLLM_V100_FP8_COALESCED_UNROLL=4
    -e VLLM_V100_FP8_COALESCED_M_UNROLL=4 -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8
  )
fi
SERVE_MOD="python3 -m fp8_w8a16_sm70.vllm_serve"
[[ "$PREC" == int4 ]] && SERVE_MOD="python3 -m vllm.entrypoints.openai.api_server"

MAXSEQS=8; for u in $USERS; do (( u > MAXSEQS )) && MAXSEQS="$u"; done
[[ "$VL" == 1 ]] && SKIPMM=(--skip-mm-profiling) || SKIPMM=()
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="results/perf_v2_${MODEL_KEY}_${PREC}_${ENGINE}_${STAMP}"
mkdir -p "$OUT/tests"; SUMMARY="$OUT/SUMMARY.txt"; CSV="$OUT/rows.csv"; : > "$SUMMARY"
echo "model,prec,engine,tp,metric,c,value,unit,quality_status,exactness" > "$CSV"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note(){ echo "[perfv2] $*" | tee -a "$SUMMARY"; }
gpus_for(){ local n="$1" o="" i; for ((i=0;i<n;i++)); do o+="${o:+,}$i"; done; echo "$o"; }
clean_guard(){ local g="$1" any=0 i used; IFS=',' read -ra a <<<"$g"
  for i in "${a[@]}"; do used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null|awk '{print $1+0}'); [[ "${used:-9999}" -gt 2000 ]] && any=1; done
  [[ "$any" -eq 0 ]]; }

serve_up(){ # $1=extra docker env array name (FA), writes global LPID/CNAME/SLOG; returns 0 if healthy
  local faflag="${1:-0}" gpus cname slog; gpus=$(gpus_for "$TP")
  cname="perfv2_${MODEL_KEY}_${PREC}_${ENGINE}_fa${faflag}"; slog="$OUT/serve_fa${faflag}.log"
  CNAME="$cname"; SLOG="$slog"
  local FAENV=(); local BLOCK=(); local PYLIB=(); local PYPATH=/work/src
  if [[ "$faflag" == 1 ]]; then
    FAENV=(-e VLLM_V100_FLASH_ATTN=1); BLOCK=(--block-size 256)
    # Stage ONLY the compiled .so (root-owned build dir -> alpine copy). The ai-bond
    # python shim must NOT be importable (it half-satisfies vLLM's flash-attn probe
    # and crashes rotary init); expose just flash_attn_v100_cuda.*.so on PYTHONPATH.
    mkdir -p "$OUT/pylib"
    docker run --rm -v "$FA_DIR":/fasrc:ro -v "$PROJECT_ROOT/$OUT":/out alpine sh -c \
      "cp /fasrc/build/lib.linux-x86_64-cpython-312/flash_attn_v100_cuda.*.so /out/pylib/ 2>/dev/null && chown -R $(id -u):$(id -g) /out/pylib" >/dev/null 2>&1
    if [[ -z "$(ls "$OUT/pylib"/flash_attn_v100_cuda.*.so 2>/dev/null)" ]]; then
      note "FA .so stage FAILED ($FA_DIR build missing) — skipping FA arm"; return 1
    fi
    PYLIB=(-v "$PROJECT_ROOT/$OUT/pylib":/falib:ro); PYPATH=/work/src:/falib
  fi
  clean_guard "$gpus" || { note "SKIP serve (GPUs $gpus busy)"; return 1; }
  docker rm -f "$cname" >/dev/null 2>&1 || true
  docker run --rm -i --name "$cname" --gpus "\"device=$gpus\"" \
    -v /mnt/models:/mnt/models:ro -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH="$PYPATH" \
    "${PYLIB[@]}" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
    -p ${PORT}:${PORT} --shm-size=16g \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_ATTENTION_BACKEND=TRITON_ATTN "${FP8_ENV[@]}" "${FAENV[@]}" \
    "$IMAGE" $SERVE_MOD --model "$MODEL" --served-model-name "$SERVED" \
      --tensor-parallel-size "$TP" --dtype float16 ${QUANT[@]+"${QUANT[@]}"} \
      --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --max-model-len 32768 --max-num-seqs "$MAXSEQS" "${SKIPMM[@]}" "${EXTRA_SERVE[@]}" "${BLOCK[@]}" \
      --disable-custom-all-reduce --gpu-memory-utilization "$GPUMEM" \
      --no-enable-chunked-prefill --no-enable-prefix-caching \
      --host 0.0.0.0 --port "$PORT" </dev/null >"$slog" 2>&1 &
  LPID=$!; local waited=0
  while (( waited < HEALTH_TIMEOUT )); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && return 0
    kill -0 "$LPID" 2>/dev/null || { note "serve exited before healthy (fa=$faflag) — $slog"; return 1; }
    sleep 10; waited=$((waited+10)); (( waited % 120 == 0 )) && note "  ...loading fa=$faflag (${waited}s)"
  done
  grep -qiE "out of memory|No available memory" "$slog" && note "INFEASIBLE (OOM at TP=$TP)" || note "FAIL (never healthy fa=$faflag)"
  return 1
}
serve_down(){ docker stop "$CNAME" >/dev/null 2>&1 || true; wait "$LPID" 2>/dev/null || true; sleep 4; }
warm(){ curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" >/dev/null 2>&1 || true; }

main(){
  note "CELL model=$MODEL_KEY prec=$PREC engine=$ENGINE TP=$TP image=$IMAGE VL=$VL tf5=$TF5 fa_eligible=$FAEL"
  note "model=$MODEL  out=$OUT  stamp=$STAMP"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE MISSING"; exit 1; }
  [[ -f "$MODEL/config.json" ]] || { note "MODEL MISSING $MODEL"; exit 2; }

  serve_up 0 || { note "cell aborted (serve)"; exit 3; }
  warm
  # TTFT_ONLY=1 -> reconcile run: re-measure TTFT only (decode is prefix-caching-
  # invariant and already valid). Otherwise the full battery + decode + TTFT pass.
  local MAINPHASE=main; [[ "$TTFT_ONLY" == 1 ]] && MAINPHASE=ttftonly
  python3 tools/perf_v2_client.py --port "$PORT" --served "$SERVED" --out "$OUT" \
    --users "$USERS" --gentok "$GENTOK" --nrun "$NRUN" --ttft-reps "$TTFT_REPS" \
    --model "$MODEL_KEY" --prec "$PREC" --engine "$ENGINE" --tp "$TP" --csv "$CSV" \
    --phase "$MAINPHASE" 2>&1 | tee -a "$SUMMARY"
  serve_down

  # B-ttft FA-on (long only): FA-eligible MHA/GQA models, ENGINE=021 only (the FA .so is
  # built against the 0.21 libtorch ABI -> ImportError 'undefined symbol c10::cuda::
  # CUDAStream::query' on 0.19), and our-plugin only (int4 runs stock vLLM, no FA hook).
  if [[ "$FA_ARM" == 1 && "$FAEL" == 1 && "$ENGINE" == 021 && "$PREC" != int4 ]]; then
    note "=== FA-on arm (TTFT-long) ==="
    if serve_up 1; then
      warm
      python3 tools/perf_v2_client.py --port "$PORT" --served "$SERVED" --out "$OUT" \
        --model "$MODEL_KEY" --prec "$PREC" --engine "$ENGINE" --tp "$TP" --csv "$CSV" \
        --ttft-reps "$TTFT_REPS" --phase ttft_fa 2>&1 | tee -a "$SUMMARY"
      serve_down
    else note "FA-on arm skipped (serve failed)"; fi
  fi
  note "DONE cell -> $OUT (rows.csv carries quality_status)"
}
main "$@"
