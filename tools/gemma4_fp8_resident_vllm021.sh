#!/usr/bin/env bash
# Gemma-4 FP8 *RESIDENT* probe on stock vLLM 0.21 sm_70 (image vllm-v100:vllm021-cu126),
# via the fp8_w8a16_sm70 patches mounted at /work (PYTHONPATH=/work/src), kernel
# JIT-compiled in-container and cached. This is the GLM-4.5-Air resident stack
# (compressed_tensors_v100.py: FP8-resident channel Linears + grouped CT-MoE +
# cudagraph) re-pointed at the Gemma-4 FP8 checkpoints. Launch and walk away;
# per-model results -> /tmp/v100_gemma4_resident/SUMMARY.txt, logs -> *_serve.log.
#
# WHY "resident" vs the existing gemma4_compat_vllm021.sh:
#   gemma4_compat runs STOCK `vllm serve` (compressed-tensors -> dequant->FP16, the
#   FP16-resident path: ~9 tok/s eager, memory-heavy). THIS script runs OUR module
#   so the FP8 weights stay FP8 and are served by the W8A16 CUDA kernel, then layered
#   with grouped-MoE + mode=0/FULL_DECODE_ONLY cudagraph — the GLM-Air recipe that
#   took GLM from 0.37->30.7 tok/s and freed ~8 GB/GPU. Goal: do the same for Gemma-4.
#
# THREE TARGETS (architecture-fit, per the project's arch-fit rule):
#   g26b-a4b-fp8dyn  MoE, active/total = 4B/26B = 0.15 < 0.2  -> our path WINS
#                    (channel scale, NO shared expert; the headline resident target).
#   g31b-fp8dyn      DENSE channel-dynamic. Dense -> our path is memory-win only,
#                    not a speed-win vs FP16; run for correctness/coverage.
#   g31b-fp8blk      DENSE block [128,128] -> exercises the native block W8A16 kernel
#                    directly (block scale is exactly what that kernel wants).
#
# GEMMA-4 SPECIFICS (vs GLM-Air, which is where the runtime risk lives):
#   - Interleaved SLIDING-WINDOW attention (window=1024, 5:1 with full_attention).
#     GLM-Air was standard attention. The open question this script ANSWERS: does
#     V100 TRITON_ATTN + FULL_DECODE_ONLY cudagraph capture sliding-window decode?
#     If cudagraph fails/falls back, set MODE=eager and the FP8-resident weights
#     still apply (you just lose the graph speedup). Watch *_serve.log for capture
#     warnings; the DECODE line reports the mode that actually ran.
#   - MoE has NO shared expert. The CT-MoE mixed path is shared-expert-OPTIONAL
#     (guarded `se is not None`), and Gemma folds per_expert_scale into topk_weights
#     BEFORE apply, so the grouped kernel composes. Validate END-TO-END (coherent
#     text), not apply-level L2 — the resident MoE path has no apply-level self-check.
#   - moe_intermediate_size=704 -> w13 N=1408, w2 K=704 (704%128=64 partial tail):
#     exercises the partial-N w13 hardening + the FP16 grouped-w2 tail path.
#
# Usage:  ./tools/gemma4_fp8_resident_vllm021.sh                 # all three, in order
#         ONLY=g26b-a4b ./tools/gemma4_fp8_resident_vllm021.sh   # just the MoE
#         MODE=eager   ./tools/gemma4_fp8_resident_vllm021.sh    # disable cudagraph
# Env: IMAGE PORT HEALTH_TIMEOUT MAXTOK GPUMEM MAXLEN NS MODE DEPTH TP ONLY
#      ATTN_BACKEND  + all VLLM_V100_CT_* flags (defaults below mirror GLM-Air).

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
PORT="${PORT:-8021}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
MAXTOK="${MAXTOK:-200}"
DEPTH="${DEPTH:-0}"
GPUMEM="${GPUMEM:-0.90}"
MAXLEN="${MAXLEN:-4096}"
NS="${NS:-8}"
TP="${TP:-4}"
MODE="${MODE:-cudagraph}"
ATTN_BACKEND="${ATTN_BACKEND:-TRITON_ATTN}"
ONLY="${ONLY:-}"
CACHE_TAG="${CACHE_TAG:-021}"

# Resident-stack flags (defaults mirror the validated GLM-Air envelope). Each has a
# kill switch; flip to 0 to A/B. Dense models ignore the MoE_* flags (no experts).
F_CT_FP8_RESIDENT="${VLLM_V100_CT_FP8_RESIDENT:-1}"
F_CT_FP8_RESIDENT_SELFCHECK="${VLLM_V100_CT_FP8_RESIDENT_SELFCHECK:-1}"
F_CT_FP8_RESIDENT_EXCLUDE="${VLLM_V100_CT_FP8_RESIDENT_EXCLUDE:-}"
F_CT_FP8_MOE_SELFCHECK="${VLLM_V100_CT_FP8_MOE_SELFCHECK:-1}"
F_CT_MOE_W13_RESIDENT="${VLLM_V100_CT_MOE_W13_RESIDENT:-1}"
F_CT_MOE_W13_MAXLAYERS="${VLLM_V100_CT_MOE_W13_RESIDENT_MAXLAYERS:-0}"
F_CT_MOE_W13_FREE_FP16="${VLLM_V100_CT_MOE_W13_FREE_FP16:-1}"
F_CT_MOE_W2_GROUPED="${VLLM_V100_CT_MOE_W2_GROUPED:-1}"
F_CT_MOE_W2_K_SPLIT="${VLLM_V100_CT_MOE_W2_K_SPLIT:-1}"
F_CT_MOE_W2_CHUNK="${VLLM_V100_CT_MOE_W2_CHUNK:-60000}"
F_CT_MOE_PREFILL_TILED="${VLLM_V100_CT_MOE_PREFILL_TILED:-1}"
F_CT_MOE_PREFILL_FUSED="${VLLM_V100_CT_MOE_PREFILL_FUSED:-1}"
F_CT_CHANNEL_WMMA="${VLLM_V100_CT_CHANNEL_WMMA:-1}"
F_CT_PROFILE="${VLLM_V100_CT_PROFILE:-0}"

if [[ "$MODE" == "cudagraph" ]]; then
    EXEC_OPTS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
else
    EXEC_OPTS=(--enforce-eager)
fi

# label|path|tp   (tp can be overridden globally with TP=)
# ORDER: dense first (isolates the channel-resident + sliding-window/cudagraph risk
# on the simpler path), MoE last (adds grouped-expert complexity on top).
MODELS=(
  "g31b-fp8dyn|/mnt/models/RedHatAI/gemma-4-31B-it-FP8-Dynamic|$TP"
  "g31b-fp8blk|/mnt/models/RedHatAI/gemma-4-31B-it-FP8-block|$TP"
  "g26b-a4b-fp8dyn|/mnt/models/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic|$TP"
)

OUT=/tmp/v100_gemma4_resident
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note() { echo "[g4-resident] $*"; }

# Guard only the GPUs THIS job will use (TP-many, starting at device 0). On the
# shared box, leaving the higher GPUs to the son's training is fine.
gpu_list_for_tp() { local n="$1" i out=""; for ((i=0;i<n;i++)); do out+="${out:+,}$i"; done; echo "$out"; }
clean_box_guard() {
    local gpus="$1" used pids any=0
    IFS=',' read -ra idxs <<<"$gpus"
    for i in "${idxs[@]}"; do
        used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
        pids=$(nvidia-smi --id="$i" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        if [[ "${used:-9999}" -gt 2000 || "${pids:-1}" -gt 0 ]]; then any=1; fi
    done
    [[ "$any" -eq 0 ]]
}

run_one() {
    local label="$1" model="$2" tp="$3" gpus cname slog rfile sfile
    gpus=$(gpu_list_for_tp "$tp")
    cname="g4res_${label}"; slog="$OUT/${label}_serve.log"; sfile="$OUT/${label}_sample.txt"

    if [[ ! -f "$model/config.json" ]]; then
        echo "$label: SKIP (not on disk: $model)" | tee -a "$SUMMARY"; return; fi
    clean_box_guard "$gpus" || { echo "$label: SKIP (GPUs $gpus busy)" | tee -a "$SUMMARY"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; return 1; }

    note "=== $label : $model  (TP=$tp gpus=$gpus mode=$MODE attn=$ATTN_BACKEND maxlen=$MAXLEN) ==="
    note "    resident: CT_FP8=$F_CT_FP8_RESIDENT MOE_W13=$F_CT_MOE_W13_RESIDENT FREE_FP16=$F_CT_MOE_W13_FREE_FP16 W2_GROUPED=$F_CT_MOE_W2_GROUPED"

    docker rm -f "$cname" >/dev/null 2>&1 || true
    docker run --rm -i --name "$cname" --gpus "\"device=$gpus\"" \
        -v /mnt/models:/mnt/models:ro \
        -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
        -p ${PORT}:${PORT} --shm-size=16g \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
        -e VLLM_V100_CT_FP8_RESIDENT="$F_CT_FP8_RESIDENT" \
        -e VLLM_V100_CT_FP8_RESIDENT_SELFCHECK="$F_CT_FP8_RESIDENT_SELFCHECK" \
        -e VLLM_V100_CT_FP8_RESIDENT_EXCLUDE="$F_CT_FP8_RESIDENT_EXCLUDE" \
        -e VLLM_V100_CT_FP8_MOE_SELFCHECK="$F_CT_FP8_MOE_SELFCHECK" \
        -e VLLM_V100_CT_MOE_W13_RESIDENT="$F_CT_MOE_W13_RESIDENT" \
        -e VLLM_V100_CT_MOE_W13_RESIDENT_MAXLAYERS="$F_CT_MOE_W13_MAXLAYERS" \
        -e VLLM_V100_CT_MOE_W13_FREE_FP16="$F_CT_MOE_W13_FREE_FP16" \
        -e VLLM_V100_CT_MOE_W2_GROUPED="$F_CT_MOE_W2_GROUPED" \
        -e VLLM_V100_CT_MOE_W2_K_SPLIT="$F_CT_MOE_W2_K_SPLIT" \
        -e VLLM_V100_CT_MOE_W2_CHUNK="$F_CT_MOE_W2_CHUNK" \
        -e VLLM_V100_CT_MOE_PREFILL_TILED="$F_CT_MOE_PREFILL_TILED" \
        -e VLLM_V100_CT_MOE_PREFILL_FUSED="$F_CT_MOE_PREFILL_FUSED" \
        -e VLLM_V100_CT_CHANNEL_WMMA="$F_CT_CHANNEL_WMMA" \
        -e VLLM_V100_CT_PROFILE="$F_CT_PROFILE" \
        -e VLLM_ATTENTION_BACKEND="$ATTN_BACKEND" \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        "$IMAGE" \
        python3 -m fp8_w8a16_sm70.vllm_serve --model "$model" --served-model-name "$label" \
            --tensor-parallel-size "$tp" --dtype float16 "${EXEC_OPTS[@]}" \
            --max-model-len "$MAXLEN" --max-num-seqs "$NS" \
            --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill \
            --host 0.0.0.0 --port "$PORT" ${EXTRA:-} \
        </dev/null >"$slog" 2>&1 &
    local lpid=$!

    local healthy=0 waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$lpid" 2>/dev/null || { note "  server exited before healthy"; break; }
        sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "  ...loading $label (${waited}s)"
    done

    # Did the resident patches engage? (rank-0 ct-layer lines carry the tally.)
    cnt() { local n; n=$(grep -c "$1" "$slog" 2>/dev/null | head -1); echo "${n:-0}"; }
    local ct_res ct_fb moe_eng moe_freed oom
    ct_res=$(cnt "ct-layer] resident"); ct_fb=$(cnt "ct-layer] fallback")
    moe_eng=$(cnt "ct-moe-w13] mixed path ENGAGED"); moe_freed=$(cnt "ct-moe-w13] FREED FP16 w13")
    oom=$(grep -ciE "out of memory|CUDA out of memory|OutOfMemoryError" "$slog" 2>/dev/null | head -1); oom=${oom:-0}
    echo "$label: CT Linear resident=$ct_res fallback=$ct_fb | MoE-w13 ENGAGED=$moe_eng FREED-FP16=$moe_freed | OOM=$oom" | tee -a "$SUMMARY"
    grep -oE "ct-layer] fallback .*why=[^ ]*" "$slog" 2>/dev/null | grep -oE "why=[^ )]*" | sort | uniq -c | sed 's/^/        /' | tee -a "$SUMMARY"
    grep -hoE "model weights take [0-9.]+ ?GiB|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$slog" 2>/dev/null | sed 's/^/        /' | tee -a "$SUMMARY"
    local sc_bad; sc_bad=$(cnt "ct-selfcheck] BAD"); echo "        self-check BAD layers=$sc_bad" | tee -a "$SUMMARY"
    # sliding-window / cudagraph capture signal (the Gemma-specific risk)
    grep -iE "cudagraph|sliding|capture|FULL_DECODE_ONLY|fall.?back to eager|piecewise" "$slog" 2>/dev/null | grep -iE "warn|error|disabl|unsupport|fall" | head -3 | sed 's/^/        cg: /' | tee -a "$SUMMARY"

    if [[ "$healthy" != 1 ]]; then
        local reason; reason=$(grep -oE "no kernel image|NotImplementedError[^\"]*|not support[^\"]*|capability[^\"]*|Unknown quantization|ValueError[^\"]*|KeyError[^\"]*|AssertionError[^\"]*|out of memory" "$slog" 2>/dev/null | head -1)
        if (( oom > 0 )); then
            echo "$label: FAIL (OOM at TP=$tp/maxlen=$MAXLEN — lower MAXLEN or raise GPUMEM/TP)" | tee -a "$SUMMARY"
        else
            echo "$label: FAIL (never healthy) reason=\"${reason:-see log}\" — $slog" | tee -a "$SUMMARY"
        fi
        note "first error lines:"; grep -nE "Error|OutOfMemory|NotImplementedError|Traceback|raise|assert" "$slog" | head -10
        tail -n 20 "$slog"; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; return 1
    fi
    note "  healthy after ${waited}s. warmup + streaming decode..."
    if [[ "${WARMUP:-1}" == "1" ]]; then
        local warmbody; warmbody=$(python3 -c "import json;print(json.dumps({'model':'$label','messages':[{'role':'user','content':'Say hello in one sentence.'}],'max_tokens':24,'temperature':0}))")
        curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' -d "$warmbody" >/dev/null 2>&1 || true
    fi

    local verdict; verdict=$(python3 - "$PORT" "$label" "$MAXTOK" "$sfile" "$DEPTH" <<'PY'
import sys, json, re, time, urllib.request
port, served, maxtok, sfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
depth = int(sys.argv[5]) if len(sys.argv) > 5 else 0
prompt = "Write a detailed multi-paragraph essay about the history, geography, and culture of France."
if depth > 0:
    target_chars = int(depth * 3.2); parts=[]; clen=0; i=0
    while clen < target_chars:
        ln = f"Reference note {i}: France has a long and varied history across many regions and eras. "
        parts.append(ln); clen += len(ln); i += 1
    prompt = ("Here is reference text:\n" + "".join(parts) +
              "\nNow write a detailed multi-paragraph essay about the history, geography, and culture of France.")
body = json.dumps({"model": served, "stream": True, "max_tokens": maxtok, "temperature": 0,
    "stream_options": {"include_usage": True},
    "messages": [{"role": "user", "content": prompt}]}).encode()
req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                            data=body, headers={"Content-Type": "application/json"})
t0=time.time(); t_first=t_last=None; n=0; chunks=[]; usage_tok=0; prompt_tok=0
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8","ignore").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: d = json.loads(data)
            except Exception: continue
            u = d.get("usage")
            if u and u.get("completion_tokens"):
                usage_tok=int(u["completion_tokens"]); prompt_tok=int(u.get("prompt_tokens") or 0)
            ch = d.get("choices") or []
            delta = ch[0]["delta"].get("content") if ch else None
            if delta:
                now=time.time()
                if t_first is None: t_first=now
                t_last=now; n+=1; chunks.append(delta)
    s = "".join(chunks).strip(); open(sfile,"w").write(s); words=s.split()
    rep = (max((words.count(w) for w in set(words)), default=0)/len(words)) if words else 1.0
    bang = (s.count("!")/len(s)) if s else 1.0
    ttft = (t_first - t0) if t_first else float("nan")
    dt = (t_last - t_first) if (t_first and t_last and n>1) else float("nan")
    meas_tok = usage_tok if usage_tok else n
    dtps = ((meas_tok-1)/dt) if (dt and dt>0) else float("nan")
    ok = bool(s) and n>=20 and bang<0.3 and rep<0.35
    print(("OK" if ok else "BAD") + f"\t{n}\t{rep:.2f}\t{ttft:.2f}\t{dtps:.2f}\t{usage_tok}\t{prompt_tok}\t" + re.sub(r'\s+',' ',s)[:140])
except Exception as e:
    print(f"BAD\t0\t1.00\tnan\tnan\t0\t0\tstream-error: {e}")
PY
)
    local tag ntok rep ttft dtps utok ptok snip
    tag=$(printf '%s' "$verdict"|cut -f1); ntok=$(printf '%s' "$verdict"|cut -f2)
    rep=$(printf '%s' "$verdict"|cut -f3); ttft=$(printf '%s' "$verdict"|cut -f4)
    dtps=$(printf '%s' "$verdict"|cut -f5); utok=$(printf '%s' "$verdict"|cut -f6)
    ptok=$(printf '%s' "$verdict"|cut -f7); snip=$(printf '%s' "$verdict"|cut -f8-)
    echo "$label: DECODE mode=$MODE depth_tok=$ptok ttft=${ttft}s decode_tok/s=$dtps (api_tok=$utok)" | tee -a "$SUMMARY"
    if [[ "$tag" == "OK" ]]; then
        echo "$label: PASS  (api_tok=$utok rep=$rep ${dtps} tok/s) | \"$snip\"" | tee -a "$SUMMARY"
    else
        echo "$label: FAIL output (tag=$tag chunks=$ntok rep=$rep) | \"$snip\" — $slog" | tee -a "$SUMMARY"
    fi
    note "  stopping $cname..."; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
}

main() {
    : > "$SUMMARY"
    echo "Gemma-4 FP8 RESIDENT probe [vLLM 0.21 sm_70, $IMAGE] mode=$MODE TP=$TP — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }
    local row label model tp
    for row in "${MODELS[@]}"; do
        IFS='|' read -r label model tp <<<"$row"
        [[ -n "$ONLY" && "$label" != *"$ONLY"* ]] && continue
        run_one "$label" "$model" "$tp" || true
        echo "" | tee -a "$SUMMARY"
    done
    note "==== GEMMA-4 RESIDENT SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
}
main "$@"
