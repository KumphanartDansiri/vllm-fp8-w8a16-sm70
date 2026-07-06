#!/usr/bin/env bash
# TurboMind-vs-ours SERVING EXACTNESS gate (CORRECTNESS, not performance).
#
# Serves the SAME model twice — BACKEND=ours then BACKEND=auto(=turbomind where eligible) —
# EAGER, greedy (temp=0), one job at a time, same TP. Captures per-token + logprobs for a fixed
# prompt set and diffs them: token-agreement, first-divergence, and the reference model's Δlogp
# at that divergence (logit-tie => benign numerical noise; large gap => systematic => inspect).
#
# Coherent text is smoke; AGREEMENT with the trusted ours path is correctness (1catai lesson).
#
# Usage (from repo root, in tmux):
#   MODEL=/mnt/models/Qwen/Qwen3.5-27B-FP8 SERVED=q27b TP=2 bash tools/turbomind_ab/exactness_ab.sh
#   env: TP, PORT(8041), MAXTOK(200), GPUS(first TP), IMAGE(baked fp8engine)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126-fp8engine}"
MODEL="${MODEL:-/mnt/models/Qwen/Qwen3.5-27B-FP8}"
SERVED="${SERVED:-q27b}"
TP="${TP:-2}"
GPUS="${GPUS:-$(seq -s, 0 $((TP-1)))}"
PORT="${PORT:-8041}"
MAXLEN="${MAXLEN:-4096}"
MAXTOK="${MAXTOK:-200}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-results/exactness_${SERVED}_tp${TP}_${STAMP}}"
TEXT_CACHE="$HOME/.cache/vllm-v100-021cu126-torchext"
mkdir -p "$OUT" "$TEXT_CACHE" "$HOME/.cache/vllm-v100-021cu126-triton" \
  "$HOME/.cache/vllm-v100-021cu126-torch" "$HOME/.cache/vllm-v100-021cu126-inductor"
SUM="$OUT/SUMMARY.txt"; : > "$SUM"
note(){ echo "[exact] $*" | tee -a "$SUM"; }

busy=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | wc -l)
if [[ "${SKIP_GUARD:-0}" != 1 && "$busy" -gt 0 ]]; then
  note "REFUSING: $busy compute process(es) on the box — this correctness run needs an isolated GPU."
  exit 2
fi
note "MODEL=$MODEL SERVED=$SERVED TP=$TP GPUS=$GPUS PORT=$PORT MAXTOK=$MAXTOK IMAGE=$IMAGE (eager, temp=0)"

serve_and_capture() {   # $1 = backend (ours|auto), $2 = capture label
  local BE="$1" LABEL="$2"
  local CNAME="exact_${SERVED}_${LABEL}" SLOG="$OUT/serve_${LABEL}.log"
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  note "serving BACKEND=$BE ($LABEL) ..."
  docker run --rm -i --name "$CNAME" --gpus "\"device=$GPUS\"" \
    -v /mnt/models:/mnt/models:ro -v "$PWD":/work -w /work -e PYTHONPATH=/work/src \
    -v "$TEXT_CACHE:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-021cu126-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-021cu126-torch:/root/.cache/torch" \
    -v "$HOME/.cache/vllm-v100-021cu126-inductor:/tmp/torchinductor_root" \
    -p ${PORT}:${PORT} --shm-size=16g \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    -e VLLM_V100_FP8_ENGINE_JIT=0 -e VLLM_V100_FP8_BACKEND="$BE" -e VLLM_V100_FP8_TM_FREE_RAW=1 \
    -e VLLM_V100_FP8_COALESCED_GEMV=1 -e VLLM_V100_FP8_COALESCED_UNROLL=4 \
    -e VLLM_V100_FP8_COALESCED_M_UNROLL=4 -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8 \
    "$IMAGE" \
    python3 -m fp8_w8a16_sm70.vllm_serve \
      --model "$MODEL" --served-model-name "$SERVED" \
      --tensor-parallel-size "$TP" --dtype float16 --quantization fp8 --enforce-eager \
      --max-model-len "$MAXLEN" --max-num-seqs 4 --skip-mm-profiling \
      --gpu-memory-utilization 0.90 --host 0.0.0.0 --port "$PORT" \
    </dev/null >"$SLOG" 2>&1 &
  local LPID=$! waited=0
  while (( waited < 2400 )); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
    kill -0 "$LPID" 2>/dev/null || { note "$LABEL: server exited before healthy"; \
      grep -nE "Error|Traceback|out of memory|RuntimeError" "$SLOG" | head -12 | tee -a "$SUM"; return 1; }
    sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "$LABEL ...loading (${waited}s)"
  done
  note "$LABEL healthy (${waited}s); capturing $MAXTOK-tok greedy generations + logprobs"
  python3 tools/turbomind_ab/exactness_capture.py --port "$PORT" --served "$SERVED" \
    --out "$OUT/cap_${LABEL}.json" --max-tokens "$MAXTOK" 2>&1 | tee -a "$SUM"
  local rc=${PIPESTATUS[0]}
  docker stop "$CNAME" >/dev/null 2>&1 || true; wait "$LPID" 2>/dev/null || true
  return "$rc"
}

serve_and_capture ours ours || { note "ours capture FAILED"; exit 1; }
serve_and_capture auto turbomind || { note "turbomind capture FAILED"; exit 1; }

note "=========================== EXACTNESS COMPARE ==========================="
python3 tools/turbomind_ab/exactness_compare.py "$OUT/cap_ours.json" "$OUT/cap_turbomind.json" \
  | tee -a "$SUM"
note "artifacts in $OUT"
