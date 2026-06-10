#!/usr/bin/env bash
# GLM-4.7-Flash (FP16 MoE) load + coherence + tok/s on the STOCK vLLM 0.21 build
# (image vllm-v100:vllm021-cu126).
#
# WHAT THIS IS
#   GLM-4.7-Flash is arch Glm4MoeLiteForCausalLM (model_type glm4_moe_lite):
#   ~30B UNQUANTIZED FP16/BF16 lite-MoE — 47 layers, 64 routed experts, 4 active,
#   1 shared, first_k_dense_replace=1, MTP head present (num_nextn_predict_layers=1).
#   Because it is UNQUANTIZED, none of the fp8_w8a16_sm70 FP8/compressed-tensors
#   patches engage — it runs the stock vLLM unquantized fused-MoE path (the same
#   one validated for Qwen3.6-35B-A3B FP16). So this is a clean canary for "does
#   the GLM-4.x *Lite* MoE architecture load + generate on the 0.21/sm_70 lane",
#   independent of the FP8 work.
#
#   ~30B FP16 (~62 GB on disk) -> TP=4 is plenty (~15.6 GB/GPU). All 4 GPUs of
#   the first node; no need for TP=8.
#
# Launcher: defaults to the fp8_w8a16_sm70.vllm_serve wrapper for harness
#   consistency (its patches feature-detect to no-ops on unquantized models).
#   Set STOCK=1 to launch plain `vllm serve` and prove it needs ZERO patches.
#
# Usage:  ./tools/glm47_flash_fp16_vllm021.sh
#         MODE=cudagraph ./tools/glm47_flash_fp16_vllm021.sh
#         STOCK=1 ./tools/glm47_flash_fp16_vllm021.sh
# Env:    IMAGE PORT HEALTH_TIMEOUT MAXTOK GPUMEM MAXLEN NS TP MODE STOCK MODEL

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
PORT="${PORT:-8021}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
MAXTOK="${MAXTOK:-200}"
GPUMEM="${GPUMEM:-0.90}"
MAXLEN="${MAXLEN:-4096}"
NS="${NS:-8}"
TP="${TP:-4}"
MODE="${MODE:-eager}"          # eager | cudagraph
STOCK="${STOCK:-0}"            # 1 -> plain `vllm serve` (no wrapper / no patches)
NRUN="${NRUN:-3}"
TIMETOK="${TIMETOK:-128}"
CG_CONFIG="${CG_CONFIG:-{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}}"
MODEL="${MODEL:-/mnt/models/zai-org/GLM-4.7-Flash}"
SERVED="glm47flash"
# KNOWN-BLOCKED on V100 (2026-06-08): GLM-4.7-Flash uses MLA attention
# (q_lora_rank=768/kv_lora_rank=512). vLLM 0.21 selects the MLA *prefill* backend
# in a SEPARATE subsystem (vllm/v1/attention/backends/mla/prefill/) from the
# decode backend. For "Hopper (SM90) and older" — which includes sm_70 — the
# ONLY prefill backend offered is FLASH_ATTN, which rejects sm_70 at kernel call
# ("FlashAttention only supports Ampere GPUs or newer"). The other prefill
# backends (FLASHINFER/TRTLLM_RAGGED/TOKENSPEED_MLA) are all major==10 (Blackwell)
# only; there is NO Triton/naive/SDPA MLA prefill anywhere. So forcing TRITON_MLA
# fixes only DECODE, not prefill — the model loads but crashes (engine 500) on the
# first token. Making it generate = writing+registering a V100 MLA prefill backend
# (its own project). ATTN_BACKEND below sets the *decode* backend only; default
# unset (auto) since no override unblocks prefill on Volta.
ATTN_BACKEND="${ATTN_BACKEND:-}"

OUT=/tmp/v100_glm47flash_021
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY_${MODE}.txt"
SLOG="$OUT/${MODE}_serve.log"
RFILE="$OUT/${MODE}_response.json"
SFILE="$OUT/${MODE}_sample.txt"
CACHE_TAG="${CACHE_TAG:-021}"   # 0.21 = torch 2.11+cu126 ABI; keep separate from 0.19
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note() { echo "[glm47flash] $*"; }

clean_box_guard() {
    local apps used
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{s+=$1} END{print s+0}')
    [[ "$apps" -eq 0 && "$used" -le 2000 ]]
}

# Guard: the model directory must be a COMPLETE download (all index-referenced
# shards present, none suspiciously small). A partial download fails the load
# with a confusing error, so check it up front.
download_complete_guard() {
    python3 - "$MODEL" <<'PY'
import json, os, sys, glob
M = sys.argv[1]
idx = glob.glob(os.path.join(M, "*.index.json"))
sft = glob.glob(os.path.join(M, "*.safetensors"))
if not idx:
    # single-file or no index: accept if at least one safetensors exists
    print("OK" if sft else "BAD: no index and no safetensors"); sys.exit()
d = json.load(open(idx[0]))
shards = sorted(set(d["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(M, s))]
small = [s for s in shards if os.path.exists(os.path.join(M, s))
         and os.path.getsize(os.path.join(M, s)) < 100 * 1024 * 1024]
partial = glob.glob(os.path.join(M, "*.incomplete")) + glob.glob(os.path.join(M, "*.part"))
if missing or small or partial:
    print(f"BAD: missing={len(missing)} small={len(small)} partial={len(partial)} "
          f"(have {len(shards)-len(missing)}/{len(shards)} shards)")
else:
    tot = sum(os.path.getsize(os.path.join(M, s)) for s in shards)
    print(f"OK ({len(shards)} shards, {tot/1e9:.1f} GB)")
PY
}

: > "$SUMMARY"
echo "GLM-4.7-Flash FP16 lite-MoE [vLLM 0.21 sm_70, MODE=$MODE, STOCK=$STOCK] — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"

docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "FAIL: missing $MODEL" | tee -a "$SUMMARY"; exit 1; }
dl=$(download_complete_guard)
echo "download: $dl" | tee -a "$SUMMARY"
[[ "$dl" == OK* ]] || { echo "ABORT: model download incomplete — let it finish first." | tee -a "$SUMMARY"; exit 1; }
clean_box_guard || { echo "SKIP: box busy (other CUDA apps or >2GB used)" | tee -a "$SUMMARY"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 1; }

# arch banner (read locally, no GPU needed)
arch=$(python3 -c "import json;print(json.load(open('$MODEL/config.json')).get('architectures',['?'])[0])" 2>/dev/null)
echo "arch: $arch  TP=$TP  max-len=$MAXLEN  mode=$MODE  attn=$ATTN_BACKEND" | tee -a "$SUMMARY"

gpus="0,1,2,3"; (( TP > 4 )) && gpus="0,1,2,3,4,5,6,7"
case "$MODE" in
    eager)     MARGS=(--enforce-eager) ;;
    cudagraph) MARGS=(--compilation-config "$CG_CONFIG") ;;
    *) note "unknown MODE=$MODE"; exit 1 ;;
esac

# launcher: wrapper (patches no-op on FP16) or pure stock
if [[ "$STOCK" == "1" ]]; then
    LAUNCH=(vllm serve "$MODEL" --served-model-name "$SERVED")
else
    LAUNCH=(python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED")
fi

cname="glm47flash_${MODE}"
note "=== $MODEL ($arch, TP=$TP on $gpus, $MODE, STOCK=$STOCK) ==="
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
    ${ATTN_BACKEND:+-e VLLM_ATTENTION_BACKEND="$ATTN_BACKEND"} \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "$IMAGE" \
    "${LAUNCH[@]}" \
        --tensor-parallel-size "$TP" --dtype float16 "${MARGS[@]}" \
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

oom=$(grep -ciE "out of memory|CUDA out of memory|OutOfMemoryError" "$SLOG" 2>/dev/null)
rarch=$(grep -oE "Resolved architecture: [A-Za-z0-9]+" "$SLOG" 2>/dev/null | tail -1)
echo "resolved: ${rarch:-<none>}  | OOM hits=$oom" | tee -a "$SUMMARY"

if [[ "$healthy" != 1 ]]; then
    echo "RESULT: FAIL (never healthy in ${HEALTH_TIMEOUT}s) — see $SLOG" | tee -a "$SUMMARY"
    note "first error lines:"; grep -nE "Error|OutOfMemory|NotImplementedError|Traceback|raise|assert|Unsupported|not support" "$SLOG" | head -12
    tail -n 25 "$SLOG"
    docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
    exit 1
fi
note "  healthy after ${waited}s. Generating coherence check..."

url="http://localhost:${PORT}/v1/chat/completions"
body=$(python3 -c "import json;print(json.dumps({'model':'$SERVED','messages':[{'role':'user','content':'Write a detailed multi-paragraph essay about the history, geography, and culture of France.'}],'max_tokens':$MAXTOK,'temperature':0}))")
curl -s "$url" -H 'Content-Type: application/json' -d "$body" >"$RFILE" 2>&1
verdict=$(python3 - "$RFILE" "$SFILE" <<'PY'
import json, sys, re
rfile, sfile = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(rfile))
    # Distinguish an engine/server crash (500, error object) from genuinely
    # incoherent generation — a crash is NOT "bad output".
    if isinstance(d, dict) and "choices" not in d and d.get("error"):
        err = d["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        code = err.get("code", "?") if isinstance(err, dict) else "?"
        print(f"CRASH\t0\t1.00\tengine error {code}: " + re.sub(r'\s+', ' ', str(msg))[:160])
        sys.exit()
    text = d["choices"][0]["message"]["content"]
    ntok = d.get("usage", {}).get("completion_tokens", 0); s = text.strip()
    open(sfile, "w").write(text)
    words = s.split()
    rep = (max((words.count(w) for w in set(words)), default=0)/len(words)) if words else 1.0
    bang = (s.count("!")/len(s)) if s else 1.0
    ok = bool(s) and ntok >= 20 and bang < 0.3 and rep < 0.35
    print(("OK" if ok else "BAD") + f"\t{ntok}\t{rep:.2f}\t" + re.sub(r'\s+',' ',s)[:160])
except Exception as e:
    print(f"CRASH\t0\t1.00\tunparseable response: {e}")
PY
)
tag=$(printf '%s' "$verdict" | cut -f1); ntok=$(printf '%s' "$verdict" | cut -f2)
rep=$(printf '%s' "$verdict" | cut -f3); snip=$(printf '%s' "$verdict" | cut -f4-)

# An engine crash (e.g. MLA-prefill FlashAttention on sm_70) kills the engine —
# don't bother timing tok/s, and report the root cause distinctly from "bad output".
if [[ "$tag" == "CRASH" ]]; then
    cause=$(grep -oE "FlashAttention only supports Ampere GPUs or newer|No valid MLA prefill backend[^\"]*|RuntimeError: [^\"]*" "$SLOG" 2>/dev/null | head -1)
    echo "RESULT [$MODE]: LOADS but GENERATION CRASHED (engine 500) — $snip" | tee -a "$SUMMARY"
    [[ -n "$cause" ]] && echo "        root cause: $cause" | tee -a "$SUMMARY"
else
    # tok/s: NRUN timed ignore_eos runs (the box is free here, so a real number)
    tbody=$(python3 -c "import json;print(json.dumps({'model':'$SERVED','messages':[{'role':'user','content':'Continue this story in vivid detail.'}],'max_tokens':$TIMETOK,'temperature':0,'ignore_eos':True}))")
    curl -s "$url" -H 'Content-Type: application/json' -d "$tbody" >/dev/null 2>&1   # warmup / capture
    tot_t=0 tot_tok=0
    for i in $(seq 1 "$NRUN"); do
        s_t=$(date +%s.%N); curl -s "$url" -H 'Content-Type: application/json' -d "$tbody" >"$OUT/.t.json" 2>&1; e_t=$(date +%s.%N)
        ct=$(python3 -c "import json;print(json.load(open('$OUT/.t.json'))['usage']['completion_tokens'])" 2>/dev/null || echo 0)
        tot_tok=$((tot_tok+${ct:-0})); tot_t=$(python3 -c "print($tot_t+($e_t-$s_t))")
    done
    toks=$(python3 -c "print(f'{$tot_tok/$tot_t:.2f}')" 2>/dev/null || echo "n/a")
    if [[ "$tag" == "OK" ]]; then
        echo "RESULT [$MODE]: LOADS + GENERATES  (${ntok} tok, rep=$rep, ~${toks} tok/s) | \"$snip\"" | tee -a "$SUMMARY"
    else
        echo "RESULT [$MODE]: loaded but BAD output (tag=$tag tok=$ntok rep=$rep) | \"$snip\" — see $SLOG" | tee -a "$SUMMARY"
    fi
fi
note "  stopping $cname..."; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
echo; note "==== SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
