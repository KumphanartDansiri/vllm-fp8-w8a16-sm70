#!/usr/bin/env bash
# One-shot: profile where FP8 MoE decode time actually goes (route vs GEMV vs
# scatter vs dispatch), to test the hypothesis that the GEMV is mature and the
# headroom is in route/scatter/data-movement (Codex 2026-06-13). Uses the
# built-in per-section CUDA-event profiler (VLLM_V100_FP8_MOE_PROFILE).
# Serves q35b-FP8 TP4, warms, decodes past the profiler warmup, captures the
# [V100-FP8-MOE-PROFILE] bucket lines.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
CACHE_TAG=021cu126
MODEL="${MODEL:-/mnt/models/Qwen/Qwen3.6-35B-A3B-FP8}"
TP="${TP:-4}"; GPUS="${GPUS:-0,1,2,3}"; PORT="${PORT:-8027}"
NUSERS="${NUSERS:-1}"     # concurrent decode streams -> drives the decode bucket M
GENTOK="${GENTOK:-600}"   # > PROFILE_WARMUP_CALLS(200) so buckets accumulate
OUT="${OUT:-results/moe_fp8_profile_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"; SLOG="$OUT/serve.log"; SERVED=moeprof
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
docker rm -f moeprof >/dev/null 2>&1 || true
echo "[prof] serving $MODEL TP=$TP (profile on); log=$SLOG"
docker run --rm -i --name moeprof --gpus "\"device=$GPUS\"" \
  -v /mnt/models:/mnt/models:ro -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
  -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
  -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
  -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
  -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
  -p ${PORT}:${PORT} --shm-size=16g \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  -e VLLM_V100_FP8_COALESCED_GEMV=1 -e VLLM_V100_FP8_COALESCED_UNROLL=4 \
  -e VLLM_V100_FP8_COALESCED_M_UNROLL=4 -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8 \
  -e VLLM_V100_FP8_MOE_W13_COALESCED=1 -e VLLM_V100_FP8_MOE_FALLBACK=1 \
  -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1 -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=128 \
  -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1 \
  -e VLLM_V100_FP8_MOE_PROFILE=1 -e VLLM_V100_FP8_MOE_PROFILE_WARMUP_CALLS=100 \
  -e VLLM_V100_FP8_MOE_PROFILE_EVERY=64 \
  "$IMAGE" \
  python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
    --tensor-parallel-size "$TP" --dtype float16 --quantization fp8 \
    --compilation-config '{"mode":0,"cudagraph_mode":"NONE"}' --enforce-eager \
    --max-model-len 4096 --max-num-seqs 8 --skip-mm-profiling \
    --gpu-memory-utilization 0.90 --no-enable-chunked-prefill \
    --host 0.0.0.0 --port "$PORT" </dev/null >"$SLOG" 2>&1 &
LPID=$!
healthy=0; waited=0
while (( waited < 2400 )); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
  kill -0 "$LPID" 2>/dev/null || { echo "[prof] server died"; break; }
  sleep 10; waited=$((waited+10)); (( waited % 120 == 0 )) && echo "[prof] loading ${waited}s"
done
[[ "$healthy" != 1 ]] && { echo "[prof] FAIL never healthy"; grep -nE "Error|Traceback" "$SLOG" | head; docker stop moeprof >/dev/null 2>&1; exit 1; }
echo "[prof] healthy; decoding $GENTOK tokens x NUSERS=$NUSERS concurrent (decode M->$NUSERS) to fill profiler"
python3 - "$PORT" "$SERVED" "$GENTOK" "$NUSERS" <<'PY' || true
import json, sys, threading, urllib.request
port, served, tok, nusers = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
aspects=["history","geography","economy","culture","cuisine","politics","science","art"]
def one(u):
    body=json.dumps({"model":served,"stream":False,"max_tokens":tok,"temperature":0,"ignore_eos":True,
      "messages":[{"role":"user","content":f"Write a long detailed essay about the {aspects[u%len(aspects)]} of France."}]}).encode()
    req=urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    try: urllib.request.urlopen(req,timeout=2400).read()
    except Exception as e: print("req err",e)
ts=[threading.Thread(target=one,args=(u,)) for u in range(nusers)]
for t in ts: t.start()
for t in ts: t.join()
PY
sleep 3
docker stop moeprof >/dev/null 2>&1 || true; wait "$LPID" 2>/dev/null || true
echo "[prof] === PROFILE buckets (decode, M=1) ==="
grep "V100-FP8-MOE-PROFILE" "$SLOG" | tail -20 | tee "$OUT/profile_buckets.txt"
echo "[prof] done -> $OUT"
