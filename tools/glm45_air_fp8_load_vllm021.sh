#!/usr/bin/env bash
# GLM-4.5-Air-FP8 LOAD probe on the STOCK vLLM 0.21 source build
# (image vllm-v100:vllm021-cu126), via the fp8_w8a16_sm70 monkey-patches mounted
# at /work (PYTHONPATH=/work/src), kernel JIT-compiled in-container and cached.
#
# WHAT THIS PROBES (and what it does NOT)
#   GLM-4.5-Air-FP8 is a compressed-tensors W8A8-FP8 checkpoint (per-channel FP8
#   weights + dynamic per-token FP8 activations), arch Glm4MoeForCausalLM, 46
#   layers / 128 routed experts / 1 shared / first_k_dense_replace=1.
#   On sm_70:
#     - Linear  W8A8 -> falls back to CompressedTensorsW8A16Fp8, which our
#       compressed_tensors_v100.py lowers to min_cap=70 and DEQUANTS FP8->FP16.
#     - MoE     CompressedTensorsW8A8Fp8MoEMethod -> our CT-MoE hook dequants
#       experts FP8->FP16 and runs the unquantized FP16 fused-MoE.
#   Both paths are FP16-RESIDENT: ~112 GB FP8 on disk -> ~210 GB FP16 at load.
#   On 8x V100-32GB (256 GB) that is ~26-27 GB/GPU of weights ALONE at TP=8,
#   leaving very little for KV cache. So this is fundamentally a LOAD/OOM probe:
#   "does the FP16-resident path even fit and generate one coherent reply?" It is
#   NOT a perf measurement (FP16-resident has no FP8 memory/speed win).
#
# Levers tuned for the memory squeeze: TP=8, tiny --max-model-len, high
# --gpu-memory-utilization, ns=8. Bump MAXLEN down / GPUMEM up if it OOMs; that
# only changes whether it fits, not whether the patch works.
#
# MODE=cudagraph is the DEFAULT (the validated best path: mode=0 + FULL_DECODE_ONLY
# + TRITON_ATTN, ~30.7 tok/s decode). Set MODE=eager to disable the graph (for
# profiling, or A/B). DEPTH=<N> prefills ~N tokens to measure decode-at-depth.
#
# Usage:  ./tools/glm45_air_fp8_load_vllm021.sh
# Env:    IMAGE PORT HEALTH_TIMEOUT MAXTOK GPUMEM MAXLEN NS MODEL MODE DEPTH ATTN_BACKEND

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
PORT="${PORT:-8021}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3600}"   # FP16-resident dequant of 112 GB is slow
MAXTOK="${MAXTOK:-200}"
# DEPTH (Phase 3 decode-at-depth): if >0, prefill a filler prompt of ~DEPTH tokens
# so the measured decode reflects attention over a DEEP KV (the real 8k/32k decode
# number), not a shallow short-prompt decode. 0 = short prompt (default).
DEPTH="${DEPTH:-0}"
GPUMEM="${GPUMEM:-0.95}"                     # push it: weights eat ~27 GB/GPU
MAXLEN="${MAXLEN:-2048}"                     # keep KV small so it can fit at all
NS="${NS:-8}"                                # ns=8 (historical ns=1 hybrid bug lever)
# Execution MODE knob (Phase 3): cudagraph (DEFAULT) | eager.
#   cudagraph = mode=0 (NO Dynamo/torch.compile — our pybind kernels are
#               `torch._dynamo.disable`'d, which is a FATAL graph break under
#               vLLM 0.21's default fullgraph compile) + FULL_DECODE_ONLY cudagraph
#               (TRITON_ATTN supports it: AttentionCGSupport.ALWAYS). This is the
#               proven mode=0+FULL_DECODE_ONLY envelope. Pins TRITON_ATTN so the
#               cudagraph-capable backend is chosen deterministically (not the FA
#               fallback). THE VALIDATED BEST PATH — measured GLM-Air mixed FP8:
#               0.37 loop → 2.68 eager → 30.7 cudagraph (and FP16-fused cudagraph is
#               only 4.81, so mixed-FP8+cudagraph wins speed AND concurrency).
#   eager     = --enforce-eager (escape hatch: no graph). USE THIS FOR PROFILING
#               (CUDA-event timers / nsys are unreliable inside a replayed graph),
#               or to A/B against the graph, or if a future op breaks capture.
MODE="${MODE:-cudagraph}"
# Pin TRITON_ATTN always: it's the AttentionCGSupport.ALWAYS backend vLLM already
# auto-selects on V100 (FA2 needs cc>=8), so pinning is behavior-neutral for eager
# and removes ambiguity for cudagraph (GPT/Codex hardening).
ATTN_BACKEND="${ATTN_BACKEND:-TRITON_ATTN}"
# Array-based opts (robust to spaces/quoting; the JSON-no-spaces trick was fragile).
if [[ "$MODE" == "cudagraph" ]]; then
    EXEC_OPTS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
else
    EXEC_OPTS=(--enforce-eager)
fi
MODEL="${MODEL:-/mnt/models/zai-org/GLM-4.5-Air-FP8}"
SERVED="glm45air"
TP=8

OUT=/tmp/v100_glm45air_021
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
SLOG="$OUT/serve.log"
RFILE="$OUT/response.json"
SFILE="$OUT/sample.txt"
CACHE_TAG="${CACHE_TAG:-021}"   # must match the 0.21 (torch 2.11+cu126) ABI cache
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note() { echo "[glm45air] $*"; }

clean_box_guard() {
    local apps used
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1} END{print s+0}')
    [[ "$apps" -eq 0 && "$used" -le 2000 ]]
}

: > "$SUMMARY"
echo "GLM-4.5-Air-FP8 load probe [vLLM 0.21 sm_70, FP16-resident CT] — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"

docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "FAIL: missing $MODEL" | tee -a "$SUMMARY"; exit 1; }
clean_box_guard || { echo "SKIP: box busy (other CUDA apps or >2GB used) — free the GPUs first" | tee -a "$SUMMARY"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 1; }

cname="glm45air_load"
note "=== $MODEL (TP=$TP, mode=$MODE, attn=$ATTN_BACKEND, max-len=$MAXLEN, gpu-mem=$GPUMEM, CT_FP8_RESIDENT=${VLLM_V100_CT_FP8_RESIDENT:-0}) ==="
if [[ "${VLLM_V100_CT_FP8_RESIDENT:-0}" == "1" ]]; then
    note "    Phase 1: channel Linears FP8-resident (experts still FP16). Expect VRAM below the ~31.9 GB/GPU FP16 baseline."
else
    note "    FP16-resident baseline: ~210 GB across 8 GPUs (~26-27 GB/GPU weights). Expect a slow load or OOM."
fi
docker rm -f "$cname" >/dev/null 2>&1 || true
docker run --rm -i --name "$cname" --gpus '"device=0,1,2,3,4,5,6,7"' \
    -v /mnt/models:/mnt/models:ro \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
    -p ${PORT}:${PORT} --shm-size=16g \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
    -e VLLM_V100_CT_FP8_RESIDENT="${VLLM_V100_CT_FP8_RESIDENT:-0}" \
    -e VLLM_V100_CT_FP8_RESIDENT_SELFCHECK="${VLLM_V100_CT_FP8_RESIDENT_SELFCHECK:-1}" \
    -e VLLM_V100_CT_FP8_RESIDENT_EXCLUDE="${VLLM_V100_CT_FP8_RESIDENT_EXCLUDE:-}" \
    -e VLLM_V100_CT_FP8_MOE_SELFCHECK="${VLLM_V100_CT_FP8_MOE_SELFCHECK:-1}" \
    -e VLLM_V100_CT_MOE_W13_RESIDENT="${VLLM_V100_CT_MOE_W13_RESIDENT:-0}" \
    -e VLLM_V100_CT_MOE_W13_RESIDENT_MAXLAYERS="${VLLM_V100_CT_MOE_W13_RESIDENT_MAXLAYERS:-0}" \
    -e VLLM_V100_CT_MOE_W13_FREE_FP16="${VLLM_V100_CT_MOE_W13_FREE_FP16:-0}" \
    -e VLLM_V100_CT_MOE_W2_GROUPED="${VLLM_V100_CT_MOE_W2_GROUPED:-1}" \
    -e VLLM_V100_CT_MOE_W2_K_SPLIT="${VLLM_V100_CT_MOE_W2_K_SPLIT:-1}" \
    -e VLLM_V100_CT_MOE_W2_CHUNK="${VLLM_V100_CT_MOE_W2_CHUNK:-60000}" \
    -e VLLM_V100_CT_MOE_PREFILL_TILED="${VLLM_V100_CT_MOE_PREFILL_TILED:-1}" \
    -e VLLM_V100_CT_MOE_PREFILL_FUSED="${VLLM_V100_CT_MOE_PREFILL_FUSED:-1}" \
    -e VLLM_V100_CT_CHANNEL_WMMA="${VLLM_V100_CT_CHANNEL_WMMA:-1}" \
    -e VLLM_V100_CT_PROFILE="${VLLM_V100_CT_PROFILE:-0}" \
    -e VLLM_ATTENTION_BACKEND="$ATTN_BACKEND" \
    -e VLLM_V100_FP8_COALESCED_GEMV="${VLLM_V100_FP8_COALESCED_GEMV:-1}" \
    -e VLLM_V100_FP8_COALESCED_UNROLL="${VLLM_V100_FP8_COALESCED_UNROLL:-4}" \
    -e VLLM_V100_FP8_COALESCED_M_UNROLL="${VLLM_V100_FP8_COALESCED_M_UNROLL:-4}" \
    -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX="${VLLM_V100_FP8_COALESCED_GEMV_M_MAX:-8}" \
    -e VLLM_V100_CT_MOE_W13_COALESCED="${VLLM_V100_CT_MOE_W13_COALESCED:-1}" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "$IMAGE" \
    python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
        --tensor-parallel-size "$TP" --dtype float16 "${EXEC_OPTS[@]}" \
        --max-model-len "$MAXLEN" --max-num-seqs "$NS" \
        --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill \
        --host 0.0.0.0 --port "$PORT" ${EXTRA:-} \
    </dev/null >"$SLOG" 2>&1 &
lpid=$!

healthy=0 waited=0
while (( waited < HEALTH_TIMEOUT )); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
    kill -0 "$lpid" 2>/dev/null || { note "  server process exited before healthy"; break; }
    sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "  ...loading (${waited}s)"
done

# Did our CT patches engage, and how many Linears went FP8-resident vs fell back?
# (rank-0 ct-layer lines carry the real tally.) Note: `grep -c | head -1` avoids
# the `grep -c || echo 0` double-"0" that breaks later arithmetic.
cnt() { local n; n=$(grep -c "$1" "$SLOG" 2>/dev/null | head -1); echo "${n:-0}"; }
ct_res=$(cnt "ct-layer] resident")
ct_fb=$(cnt "ct-layer] fallback")
ct_moe=$(cnt "ct-moe.*dequant experts FP8->FP16")
oom=$(grep -ciE "out of memory|CUDA out of memory|OutOfMemoryError" "$SLOG" 2>/dev/null | head -1); oom=${oom:-0}
echo "CT Linear: resident=$ct_res fallback=$ct_fb (rank0) | CT-MoE banner=$ct_moe | OOM hits=$oom" | tee -a "$SUMMARY"
# fallback reasons (distinct) — shows WHY any Linear stayed FP16 (TP-K/N alignment etc.)
grep -oE "ct-layer] fallback .*why=[^ ]*" "$SLOG" 2>/dev/null | grep -oE "why=[^ )]*" | sort | uniq -c | sed 's/^/        /' | tee -a "$SUMMARY"
# VRAM headline: model-weights footprint + KV headroom (the Phase-1 payoff).
grep -hoE "model weights take [0-9.]+ ?GiB|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$SLOG" 2>/dev/null | sed 's/^/        /' | tee -a "$SUMMARY"
# self-check: any resident layer whose kernel output diverged from dequant-FP16
# on real weights — pinpoints an unsafe Linear family (qkv/o/gate_up/...).
sc_bad=$(cnt "ct-selfcheck] BAD")
echo "self-check BAD layers=$sc_bad" | tee -a "$SUMMARY"
# Phase 2a: grouped-MoE-kernel real-weight self-check on w13 experts (diagnostic;
# execution still FP16). The last ct-moe-selfcheck line carries the ok/bad tally.
moe_sc=$(grep "ct-moe-selfcheck] w13" "$SLOG" 2>/dev/null | tail -1)
[[ -n "$moe_sc" ]] && echo "MoE w13 self-check: ${moe_sc#*ct-moe-selfcheck] }" | tee -a "$SUMMARY"
# Phase 2d: real-weight self-check of the FP16 grouped w2 kernel (diagnostic).
moe_sc2=$(grep "ct-moe-selfcheck] w2" "$SLOG" 2>/dev/null | tail -1)
[[ -n "$moe_sc2" ]] && echo "MoE w2 self-check: ${moe_sc2#*ct-moe-selfcheck] }" | tee -a "$SUMMARY"
# Phase 2b: mixed CT-MoE (w13 FP8 routed + runner-combined shared). Positive
# runtime marker = the one-shot ENGAGED banner; ERR = a layer disabled+fell back.
w13_eng=$(cnt "ct-moe-w13] mixed path ENGAGED")
w13_err=$(cnt "ct-moe-w13] mixed ERR")
w13_freed=$(cnt "ct-moe-w13] FREED FP16 w13")
if (( w13_eng + w13_err + w13_freed > 0 )); then
    echo "MoE w13-mixed: ENGAGED=$w13_eng (>=1 => mixed ran) ERR->disable=$w13_err FREED-FP16=$w13_freed" | tee -a "$SUMMARY"
    grep -E "ct-moe-w13] (mixed path ENGAGED|mixed ERR|FREED FP16)" "$SLOG" 2>/dev/null | head -4 | sed 's/^/        /' | tee -a "$SUMMARY"
fi
if (( sc_bad > 0 )); then
    echo "  unsafe families (prefix sample):" | tee -a "$SUMMARY"
    grep "ct-selfcheck] BAD" "$SLOG" 2>/dev/null | grep -oE "prefix=[^ ]*" \
        | sed -E 's/\.[0-9]+\./.N./g' | sort | uniq -c | sed 's/^/        /' | tee -a "$SUMMARY"
fi

if [[ "$healthy" != 1 ]]; then
    if (( oom > 0 )); then
        echo "RESULT: OOM — patch path is correct but FP16-resident does not fit at TP=8/max-len=$MAXLEN." | tee -a "$SUMMARY"
        echo "        (Retry with lower MAXLEN / higher GPUMEM to confirm, or pursue FP8-resident MoE.)" | tee -a "$SUMMARY"
    else
        echo "RESULT: FAIL (never healthy in ${HEALTH_TIMEOUT}s, no OOM) — see $SLOG" | tee -a "$SUMMARY"
    fi
    note "first error lines:"; grep -nE "Error|OutOfMemory|NotImplementedError|Traceback|raise|assert|No FP8 MoE backend" "$SLOG" | head -12
    tail -n 25 "$SLOG"
    docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
    exit 1
fi

note "  healthy after ${waited}s. Warmup request (dodge first-call Triton JIT spikes)..."
# A throwaway generation warms vLLM's own Triton kernels (_compute_slot_mapping,
# kernel_unified_attention) that JIT-compile on first use and otherwise depress
# the measured decode rate. WARMUP=0 to skip.
if [[ "${WARMUP:-1}" == "1" ]]; then
    warmbody=$(python3 -c "import json;print(json.dumps({'model':'$SERVED','messages':[{'role':'user','content':'Say hello in one sentence.'}],'max_tokens':24,'temperature':0}))")
    curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' -d "$warmbody" >/dev/null 2>&1 || true
fi
note "  Streaming one generation (coherence + decode tok/s)..."
# Stream so we separate prefill (TTFT) from steady-state decode tok/s. The decode
# rate = (tokens after the first) / (time between first and last token), which is
# the number Phase 2d targets (the per-expert w2 loop pinned it at ~1 tok/s).
verdict=$(python3 - "$PORT" "$SERVED" "$MAXTOK" "$SFILE" "$DEPTH" <<'PY'
import sys, json, re, time, urllib.request
port, served, maxtok, sfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
depth = int(sys.argv[5]) if len(sys.argv) > 5 else 0
prompt = "Write a detailed multi-paragraph essay about the history, geography, and culture of France."
if depth > 0:
    # Build a filler prompt of ~depth tokens so decode runs over a DEEP KV. Target a
    # char budget that UNDERSHOOTS depth (~3.2 chars/tok vs real ~4) to stay under
    # max-model-len; numbered lines avoid pathological repeated-token compression.
    # The TRUE depth is the reported prompt_tokens, not this estimate.
    target_chars = int(depth * 3.2)
    parts = []; clen = 0; i = 0
    while clen < target_chars:
        ln = f"Reference note {i}: France has a long and varied history across many regions and eras. "
        parts.append(ln); clen += len(ln); i += 1
    prompt = ("Here is reference text:\n" + "".join(parts) +
              "\nNow write a detailed multi-paragraph essay about the history, "
              "geography, and culture of France.")
body = json.dumps({"model": served, "stream": True, "max_tokens": maxtok, "temperature": 0,
    "stream_options": {"include_usage": True},
    "messages": [{"role": "user", "content": prompt}]}).encode()
req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                            data=body, headers={"Content-Type": "application/json"})
t0 = time.time(); t_first = t_last = None; n = 0; chunks = []; usage_tok = 0; prompt_tok = 0
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except Exception:
                continue
            # gold-standard token counts from the trailing usage chunk (choices=[])
            u = d.get("usage")
            if u and u.get("completion_tokens"):
                usage_tok = int(u["completion_tokens"])
                prompt_tok = int(u.get("prompt_tokens") or 0)
            ch = d.get("choices") or []
            delta = ch[0]["delta"].get("content") if ch else None
            if delta:
                now = time.time()
                if t_first is None:
                    t_first = now
                t_last = now; n += 1; chunks.append(delta)
    s = "".join(chunks).strip(); open(sfile, "w").write(s)
    words = s.split()
    rep = (max((words.count(w) for w in set(words)), default=0)/len(words)) if words else 1.0
    bang = (s.count("!")/len(s)) if s else 1.0
    ttft = (t_first - t0) if t_first else float("nan")
    dt = (t_last - t_first) if (t_first and t_last and n > 1) else float("nan")
    # decode tok/s from API completion_tokens (gold standard) when available, else
    # the SSE chunk count; both reported so any divergence is visible.
    meas_tok = usage_tok if usage_tok else n
    dtps = ((meas_tok - 1)/dt) if (dt and dt > 0) else float("nan")
    ok = bool(s) and n >= 20 and bang < 0.3 and rep < 0.35
    print(("OK" if ok else "BAD") + f"\t{n}\t{rep:.2f}\t{ttft:.2f}\t{dtps:.2f}\t{usage_tok}\t{prompt_tok}\t" + re.sub(r'\s+',' ',s)[:140])
except Exception as e:
    print(f"BAD\t0\t1.00\tnan\tnan\t0\t0\tstream-error: {e}")
PY
)
tag=$(printf '%s' "$verdict" | cut -f1);  ntok=$(printf '%s' "$verdict" | cut -f2)
rep=$(printf '%s' "$verdict" | cut -f3);  ttft=$(printf '%s' "$verdict" | cut -f4)
dtps=$(printf '%s' "$verdict" | cut -f5); utok=$(printf '%s' "$verdict" | cut -f6)
ptok=$(printf '%s' "$verdict" | cut -f7); snip=$(printf '%s' "$verdict" | cut -f8-)
# utok = API completion_tokens (gold standard); ntok = SSE delta chunks; ptok =
# prompt_tokens (the actual decode DEPTH). decode_tok/s is from utok when present.
echo "DECODE: mode=$MODE depth_tok=${ptok} ttft=${ttft}s decode_tok/s=${dtps} (api_tok=${utok} sse_chunks=${ntok})" | tee -a "$SUMMARY"
if [[ "$tag" == "OK" ]]; then
    echo "RESULT: LOADS + GENERATES  (api_tok=${utok}, rep=$rep, ${dtps} tok/s) | \"$snip\"" | tee -a "$SUMMARY"
else
    echo "RESULT: loaded but BAD output (tag=$tag sse_chunks=$ntok rep=$rep) | \"$snip\" — see $SLOG" | tee -a "$SUMMARY"
fi

note "  stopping $cname..."; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
echo; note "==== SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
