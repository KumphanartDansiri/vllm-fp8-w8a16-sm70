#!/usr/bin/env bash
# FP8 SERVING PERFORMANCE: TurboMind vs our FP8 dequant, SAME checkpoint + SAME hardware.
# Correctness is already locked (docs/FP8_ENGINE_STAGE_F_LOADER_WIRING.md); this measures ONLY
# the buyer-facing "what do I get?" numbers. GPTQ/AWQ are a separate later phase — not compared here.
#
# ONE config per invocation (env knobs); run configs SEQUENTIALLY (one job at a time, isolated cache).
#   MODEL=/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8 SERVED=q35b TP_SIZE=4 BACKEND=turbomind MODE=cudagraph \
#     CONC="1 2 4 8" bash tools/turbomind_ab/fp8_turbomind_vs_ours_perf.sh
# Appends rows to $CSV: model,tp,backend,mode,conc,peruser_toks,agg_toks,ttft_s,residentMiB,peakMiB
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126-fp8engine}"
MODEL="${MODEL:-/mnt/models/Qwen/Qwen3.5-35B-A3B-FP8}"
SERVED="${SERVED:-q35b}"
TP_SIZE="${TP_SIZE:-2}"
GPUS="${GPUS:-$(seq -s, 0 $((TP_SIZE-1)))}"
BACKEND="${BACKEND:-turbomind}"        # ours | turbomind (=auto engages turbomind where eligible)
MODE="${MODE:-cudagraph}"              # eager | cudagraph (cudagraph = the meaningful serving path)
CONC="${CONC:-1 2 4 8}"
GENTOK="${GENTOK:-256}"
MAXLEN="${MAXLEN:-4096}"
PORT="${PORT:-8051}"
OUT="${OUT:-results/fp8_perf}"
CSV="${CSV:-$OUT/perf.csv}"
TEXT_CACHE="$HOME/.cache/vllm-v100-021cu126-torchext"
mkdir -p "$OUT" "$TEXT_CACHE" "$HOME/.cache/vllm-v100-021cu126-triton" \
  "$HOME/.cache/vllm-v100-021cu126-torch" "$HOME/.cache/vllm-v100-021cu126-inductor"
[[ -f "$CSV" ]] || echo "model,tp,backend,mode,conc,peruser_toks,agg_toks,ttft_s,residentMiB,peakMiB" > "$CSV"
note(){ echo "[perf] $*"; }
# BACKEND=turbomind is passed to the plugin as auto so an odd ineligible shard falls back to ours
# instead of hard-raising; engagement is asserted from the logs below.
PLUGIN_BACKEND="$BACKEND"; [[ "$BACKEND" == turbomind ]] && PLUGIN_BACKEND=auto

busy=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | wc -l)
if [[ "${SKIP_GUARD:-0}" != 1 && "$busy" -gt 0 ]]; then
  note "REFUSING: $busy compute proc(s) on the box — perf needs an isolated GPU."; exit 2; fi

if [[ "$MODE" == cudagraph ]]; then
  MODE_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
else
  MODE_ARGS=(--enforce-eager)
fi

CNAME="perf_${SERVED}_tp${TP_SIZE}_${BACKEND}_${MODE}"
SLOG="$OUT/serve_${CNAME}.log"
docker rm -f "$CNAME" >/dev/null 2>&1 || true
note "serve $SERVED TP=$TP_SIZE BACKEND=$BACKEND($PLUGIN_BACKEND) MODE=$MODE GPUS=$GPUS gentok=$GENTOK"
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
  -e VLLM_V100_FP8_ENGINE_JIT=0 -e VLLM_V100_FP8_BACKEND="$PLUGIN_BACKEND" -e VLLM_V100_FP8_TM_FREE_RAW=1 \
  -e VLLM_V100_FP8_COALESCED_GEMV=1 -e VLLM_V100_FP8_COALESCED_UNROLL=4 \
  -e VLLM_V100_FP8_COALESCED_M_UNROLL=4 -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8 \
  "$IMAGE" \
  python3 -m fp8_w8a16_sm70.vllm_serve \
    --model "$MODEL" --served-model-name "$SERVED" \
    --tensor-parallel-size "$TP_SIZE" --dtype float16 --quantization fp8 "${MODE_ARGS[@]}" \
    --max-model-len "$MAXLEN" --max-num-seqs 16 --skip-mm-profiling \
    --gpu-memory-utilization 0.90 --host 0.0.0.0 --port "$PORT" \
  </dev/null >"$SLOG" 2>&1 &
LPID=$!

healthy=0; waited=0
while (( waited < 2400 )); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
  kill -0 "$LPID" 2>/dev/null || { note "$CNAME: server exited before healthy"; \
    grep -nE "Error|Traceback|out of memory|RuntimeError" "$SLOG" | head -15; break; }
  sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "...loading (${waited}s)"
done
[[ "$healthy" == 1 ]] || { note "$CNAME FAILED to become healthy"; docker stop "$CNAME" >/dev/null 2>&1; exit 1; }

# engagement confirmation (buyer trust: prove which path actually ran)
eng=$(grep -cE "TurboMind (DENSE|MoE) engaged" "$SLOG" 2>/dev/null || echo 0)
if [[ "$BACKEND" == turbomind ]]; then
  (( eng > 0 )) && note "engagement OK: TurboMind engaged ($eng banners)" \
    || note "WARN: BACKEND=turbomind but NO TurboMind-engaged banner — measuring ours?!"
else
  (( eng == 0 )) && note "engagement OK: ours (no TurboMind banners)" \
    || note "WARN: BACKEND=ours but $eng TurboMind-engaged banners present"
fi

resident=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null | sort -n | tail -1)
note "healthy (${waited}s); resident=${resident}MiB; concurrency sweep [$CONC]"
PKF="$OUT/.peak_${CNAME}"; echo "${resident:-0}" > "$PKF"
( while :; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null | sort -n | tail -1); \
    cur=$(cat "$PKF" 2>/dev/null||echo 0); [[ -n "$m" && "$m" -gt "$cur" ]] && echo "$m" > "$PKF"; sleep 1; done ) &
SAMPLER=$!

for C in $CONC; do
  row=$(python3 tools/turbomind_ab/perf_bench_client.py --port "$PORT" --served "$SERVED" \
        --conc "$C" --max-tokens "$GENTOK")
  note "$row"
  pu=$(sed -n 's/.*peruser_toks=\([0-9.na]*\).*/\1/p' <<<"$row")
  ag=$(sed -n 's/.*agg_toks=\([0-9.na]*\).*/\1/p' <<<"$row")
  tt=$(sed -n 's/.*ttft_s=\([0-9.na]*\).*/\1/p' <<<"$row")
  peak=$(cat "$PKF" 2>/dev/null || echo "$resident")
  echo "$SERVED,$TP_SIZE,$BACKEND,$MODE,$C,$pu,$ag,$tt,$resident,$peak" >> "$CSV"
done

kill "$SAMPLER" 2>/dev/null || true
docker stop "$CNAME" >/dev/null 2>&1 || true; wait "$LPID" 2>/dev/null || true
note "done -> $CSV"
