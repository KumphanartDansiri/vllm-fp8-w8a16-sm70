#!/usr/bin/env bash
# FA-V100 prefill e2e TTFT A/B — GLM-4.5-Air-FP8, TP=8, vLLM 0.21 (vllm021-cu126).
#
# THE proof point for the FlashAttention-V100 integration (docs/FA_V100_AUDIT.md):
# microbench says Triton unified_attention = 945 ms/layer @26k (x46 = 43.5s = the
# documented ~42s TTFT residual) vs ai-bond 110 ms/layer (x46 = 5.1s). Projection:
# TTFT@26k 60s -> ~22s. This script measures it END TO END.
#
# TWO ARMS, byte-identical config except ONE env flag:
#   triton : VLLM_V100_FLASH_ATTN=0  -> current promoted config (baseline)
#   fa     : VLLM_V100_FLASH_ATTN=1  -> prefill routed to flash_attn_v100
# Both arms: --block-size 256 (ai-bond paged constraint; SAME in baseline to isolate
# the kernel effect), prefix caching OFF (every trial = true prefill), cudagraph
# FULL_DECODE_ONLY (decode path untouched by the patch BY CONSTRUCTION — verify
# decode tok/s is unchanged across arms).
#
# Prereqs (already done this session): ai-bond built for torch2.11+cu126 with the
# 3 working-tree patches (BLOCK_N_128=128 straddle fix, setup.py pin 12.6, tanhf);
# extension importable from /fa/build/lib.linux-x86_64-cpython-312 (no install).
#
# Usage:  ./tools/fa_prefill_ttft_ab_vllm021.sh            # both arms
#         ONLY=fa ./tools/fa_prefill_ttft_ab_vllm021.sh    # one arm
# Env: IMAGE PORT HEALTH_TIMEOUT GPUMEM MAXLEN PROMPT_TOKENS TRIALS MAXTOK ONLY
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
FA_DIR="${FA_DIR:-/home/kumphanartd/flash-attention-v100}"
# NOTE: only the compiled .so is exposed to the serving container. ai-bond's
# `flash_attn` python shim must NOT be importable by vLLM — its presence makes
# vLLM's optional flash-attn probe half-succeed (`import flash_attn` OK,
# `flash_attn.ops` missing) and crashes GLM4 rotary init (common.py:138).
PORT="${PORT:-8023}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3600}"
GPUMEM="${GPUMEM:-0.90}"
MAXLEN="${MAXLEN:-28672}"
PROMPT_TOKENS="${PROMPT_TOKENS:-26000}"
TRIALS="${TRIALS:-2}"
MAXTOK="${MAXTOK:-64}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
SKIP_MM="${SKIP_MM:-0}"   # 1 = --skip-mm-profiling (ch1 standard for VL-capable archs)
MNBT="${MNBT:-}"          # --max-num-batched-tokens (chunked-prefill granularity; default vllm 2048)
MODEL="${MODEL:-/mnt/models/zai-org/GLM-4.5-Air-FP8}"
MODE="${MODE:-cudagraph}"   # eager: for models whose decode-graph capture fails (122B hybrid)
TP="${TP:-8}"
if [[ "$MODE" == "cudagraph" ]]; then
    EXEC_OPTS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
else
    EXEC_OPTS=(--enforce-eager)
fi
ONLY="${ONLY:-}"
CACHE_TAG="${CACHE_TAG:-021}"
SERVED="faab"

OUT=/tmp/v100_fa_ttft_ab
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
: > "$SUMMARY"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note() { echo "[fa-ab] $*"; }

clean_box_guard() {
    local any=0 i used pids
    for i in 0 1 2 3 4 5 6 7; do
        used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
        pids=$(nvidia-smi --id="$i" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        [[ "${used:-9999}" -gt 2000 || "${pids:-1}" -gt 0 ]] && any=1
    done
    [[ "$any" -eq 0 ]]
}

run_arm() {
    local label="$1" faflag="$2" cname slog
    cname="faab_${label}"; slog="$OUT/${label}_serve.log"
    note "=== arm=$label VLLM_V100_FLASH_ATTN=$faflag (TP=$TP block=256 maxlen=$MAXLEN prompt=$PROMPT_TOKENS) ==="
    docker rm -f "$cname" >/dev/null 2>&1 || true
    docker run --rm -i --name "$cname" --gpus "\"device=$(seq -s, 0 $((TP-1)))\"" \
        -v /mnt/models:/mnt/models:ro \
        -v "$PROJECT_ROOT":/work -w /work \
        -v "$OUT/pylib":/falib:ro \
        -e PYTHONPATH=/work/src:/falib \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
        -p ${PORT}:${PORT} --shm-size=16g \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
        -e VLLM_V100_FLASH_ATTN="$faflag" \
        -e VLLM_V100_CT_FP8_RESIDENT=1 -e VLLM_V100_CT_FP8_RESIDENT_SELFCHECK=1 \
        -e VLLM_V100_CT_MOE_W13_RESIDENT=1 -e VLLM_V100_CT_MOE_W13_FREE_FP16=1 \
        -e VLLM_V100_CT_MOE_W2_GROUPED=1 -e VLLM_V100_CT_MOE_W13_COALESCED=1 \
        -e VLLM_V100_CT_MOE_PREFILL_TILED=1 -e VLLM_V100_CT_MOE_PREFILL_FUSED=1 \
        -e VLLM_V100_CT_CHANNEL_WMMA=1 \
        -e VLLM_V100_FP8_COALESCED_GEMV=1 -e VLLM_V100_FP8_COALESCED_UNROLL=4 \
        -e VLLM_V100_FP8_COALESCED_M_UNROLL=4 -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8 \
        -e VLLM_V100_FP8_MOE_W13_COALESCED=1 -e VLLM_V100_FP8_MOE_FALLBACK=1 \
        -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1 -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=128 \
        -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1 \
        -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        "$IMAGE" \
        python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
            --tensor-parallel-size "$TP" --dtype float16 "${EXEC_OPTS[@]}" \
            --max-model-len "$MAXLEN" --max-num-seqs 8 --block-size "$BLOCK_SIZE" \
            $([ "$SKIP_MM" = "1" ] && echo --skip-mm-profiling) \
            $([ -n "$MNBT" ] && echo --max-num-batched-tokens "$MNBT") \
            --gpu-memory-utilization "$GPUMEM" \
            \
            --host 0.0.0.0 --port "$PORT" \
        </dev/null >"$slog" 2>&1 &
    local lpid=$!

    local healthy=0 waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$lpid" 2>/dev/null || { note "  server exited before healthy"; break; }
        sleep 10; waited=$((waited+10)); (( waited % 120 == 0 )) && note "  ...loading $label (${waited}s)"
    done
    if [[ "$healthy" != 1 ]]; then
        echo "$label: FAIL (never healthy) — $slog" | tee -a "$SUMMARY"
        grep -nE "Error|Traceback|no kernel image|out of memory|assert" "$slog" | head -8
        docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; return 1
    fi
    note "  healthy after ${waited}s; warmup..."

    # short warmup (JIT, cudagraph capture)
    curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":16,\"temperature\":0}" >/dev/null 2>&1 || true

    # TTFT trials: long-prompt streaming, varied filler each trial (prefix-cache-proof
    # even though caching is off). Reports ttft, decode tok/s, prompt tokens, snippet.
    : > "$OUT/${label}_trials.txt"   # fresh per run — reruns must not mix evidence
    local t
    for t in $(seq 1 "$TRIALS"); do
        python3 - "$PORT" "$SERVED" "$PROMPT_TOKENS" "$MAXTOK" "$t" <<'PY' | tee -a "$OUT/${label}_trials.txt"
import json, sys, time, urllib.request, re
port, served, ptok, maxtok, trial = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
# ~0.75 tok/word -> words; vary seed word per trial to defeat any caching
seedw = ["alpha","bravo","charlie","delta","echo"][trial % 5]
words = int(ptok / 1.31)
para = ("The %s expedition crossed the silent valley while measuring glacier retreat, "
        "soil moisture, and the migration of mountain birds. " % seedw)
body = (para * (words // len(para.split()) + 1))
body = " ".join(body.split()[:words])
prompt = body + "\n\nIn one short sentence, what was being measured?"
req = {"model": served, "messages": [{"role": "user", "content": prompt}],
       "max_tokens": maxtok, "temperature": 0, "stream": True,
       "stream_options": {"include_usage": True}}
data = json.dumps(req).encode()
t0 = time.time(); tf = None; chunks = []; usage = None
r = urllib.request.urlopen(urllib.request.Request(
    f"http://localhost:{port}/v1/chat/completions", data=data,
    headers={"Content-Type": "application/json"}), timeout=1800)
for raw in r:
    line = raw.decode("utf-8", "ignore").strip()
    if not line.startswith("data:"): continue
    payload = line[5:].strip()
    if payload == "[DONE]": break
    try: j = json.loads(payload)
    except Exception: continue
    if j.get("usage"): usage = j["usage"]
    ch = j.get("choices") or []
    if ch and (ch[0].get("delta") or {}).get("content"):
        if tf is None: tf = time.time()
        chunks.append(ch[0]["delta"]["content"])
tend = time.time()
text = "".join(chunks)
ttft = (tf - t0) if tf else float("nan")
n = len(chunks)
dtps = (n - 1) / (tend - tf) if tf and n > 1 else float("nan")
ptk = (usage or {}).get("prompt_tokens", "?")
ok = bool(re.search(r"glacier|moisture|bird|measur", text, re.I)) and len(text) > 10
print(f"trial{trial}\t{'OK' if ok else 'SUSPECT'}\tprompt_tokens={ptk}\tttft={ttft:.2f}s\t"
      f"decode={dtps:.2f} tok/s\t{re.sub(chr(92)+'s+',' ',text)[:100]}")
PY
    done

    # banners: did the fa arm actually route?
    local routed; routed=$(grep -c "prefill -> flash_attn_v100" "$slog" 2>/dev/null || true)
    local fellback; fellback=$(grep -m3 "fallback to Triton" "$slog" 2>/dev/null | tr '\n' ';' || true)
    {
      echo "$label: $(tail -n +1 "$OUT/${label}_trials.txt" | tr '\n' ' | ')"
      echo "$label: fa-route-banner=$routed ${fellback:+fallbacks: $fellback}"
    } | tee -a "$SUMMARY"

    docker stop "$cname" >/dev/null 2>&1 || true
    wait "$lpid" 2>/dev/null || true
    sleep 5
}

note "results -> $OUT (SUMMARY.txt)"
clean_box_guard || { note "ABORT: box not clean (need all 8 GPUs)"; exit 3; }
[[ -f "$FA_DIR/build/lib.linux-x86_64-cpython-312/flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so" ]] \
  || { note "ABORT: ai-bond cu126 build missing in $FA_DIR/build"; exit 4; }
# stage ONLY the compiled extension (root-owned build dir -> copy via docker)
docker run --rm -v "$FA_DIR":/fasrc:ro -v "$OUT":/out alpine sh -c \
  "mkdir -p /out/pylib && cp /fasrc/build/lib.linux-x86_64-cpython-312/flash_attn_v100_cuda.*.so /out/pylib/ && chown -R $(id -u):$(id -g) /out/pylib" \
  || { note "ABORT: failed to stage flash_attn_v100_cuda.so"; exit 4; }
note "staged $(ls "$OUT/pylib" 2>/dev/null | tr '\n' ' ')"

for arm in "triton|0" "fa|1"; do
    IFS='|' read -r label faflag <<<"$arm"
    [[ -n "$ONLY" && "$ONLY" != "$label" ]] && continue
    run_arm "$label" "$faflag"
done
note "=== SUMMARY ==="
cat "$SUMMARY"