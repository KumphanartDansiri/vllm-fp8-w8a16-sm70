#!/usr/bin/env bash
# ======================================================================================
# GLM-4.7-Flash (Glm4MoeLite, MLA attention) on V100 — A/B the MLA-prefill unblock.
# DUAL-ENGINE: runs on BOTH vLLM 0.21 (cu126) and vLLM 0.19 (tf5/cu128).
#
#   arm OFF : VLLM_V100_MLA_PREFILL=0 -> stock vLLM MLA prefill (FLASH_ATTN backend)
#             -> EXPECTED first-token engine 500 "FlashAttention only supports Ampere".
#   arm ON  : VLLM_V100_MLA_PREFILL=1 -> MLA prefill routed to ai-bond flash_attn_v100
#             -> EXPECTED coherent generation. Decode stays TritonMLA either way.
#
# The fa_v100_mla_prefill patch resolves the prefill choke point per engine:
#   0.21 -> vllm.v1.attention.backends.mla.prefill.flash_attn.FlashAttnPrefillBackend
#   0.19 -> vllm.model_executor.layers.attention.mla_attention.MLACommonImpl (base;
#           reached via TritonMLAImpl super()). Same method name + signature on both.
#
# GLM-4.7-Flash MLA dims: qk_head_dim=256 (192+64), v_head_dim=256 -> D=256 ai-bond
# tile, NO padding. ~30B FP16 lite-MoE (64 experts/4 active, 1 shared) -> TP=4 fits.
# It is a <think></think> REASONING model -> use /v1/chat/completions, natural EOS
# (NOT ignore_eos -> that forces 4096 and induces a reasoning-model repetition artifact).
# Chunked prefill LEFT ON (never --no-enable-chunked-prefill, per project rule); a long
# prompt exercises run_prefill_context_chunk -> the LSE merge path.
#
# TTFT: every probe streams (stream=True, include_usage) so we report TTFT (time to first
# token = prefill UX cost) SEPARATELY from steady-state decode tok/s. The long-prompt TTFT
# is the meaningful long-context number (chunked-prefill + LSE merge).
#
# The ai-bond .so is torch-ABI-specific: the prebuilt one targets torch 2.11 (0.21 image)
# and will NOT import under the 0.19 images' torch 2.10 (undefined c10::cuda symbol). This
# tool keeps a PER-ENGINE .so cache and auto-builds (sm_70, no GPU needed) inside the
# selected image when the cached .so does not import. BUILD_SO=force rebuilds; =never aborts.
#
# Usage:  ENGINE=021 ./tools/glm47_flash_mla_v100_ab.sh                 # vLLM 0.21 (validated)
#         ENGINE=019 ./tools/glm47_flash_mla_v100_ab.sh                 # vLLM 0.19 (the new arm)
#         ENGINE=019 MODE=cudagraph ./tools/glm47_flash_mla_v100_ab.sh  # + decode graph
#         ENGINE=019 ARMS=on ./tools/glm47_flash_mla_v100_ab.sh         # only the ON arm
#         ENGINE=019 BUILD_SO=force ./tools/glm47_flash_mla_v100_ab.sh  # force .so rebuild
# Env: ENGINE IMAGE FA_DIR MODEL PORT TP MAXLEN NS MAXTOK LONGREPS MODE ARMS MLA_CG
#      HEALTH_TIMEOUT BUILD_SO JOBS TEMP
# ======================================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

ENGINE="${ENGINE:-021}"
case "$ENGINE" in
    021) DEF_IMAGE="vllm-v100:vllm021-cu126";      DEF_CACHE="021cu126"; ENGINE_LABEL="vLLM 0.21+cu126" ;;
    019) DEF_IMAGE="vllm-v100-py312:vllm019-tf5";  DEF_CACHE="019tf5";   ENGINE_LABEL="vLLM 0.19+tf5(transformers-5.x)" ;;
    *)   echo "ENGINE must be 021 | 019"; exit 1 ;;
esac
IMAGE="${IMAGE:-$DEF_IMAGE}"
CACHE_TAG="${CACHE_TAG:-$DEF_CACHE}"

FA_DIR="${FA_DIR:-/home/kumphanartd/flash-attention-v100}"
MODEL="${MODEL:-/mnt/models/zai-org/GLM-4.7-Flash}"
SERVED="glm47flash"
PORT="${PORT:-8021}"
TP="${TP:-4}"
MAXLEN="${MAXLEN:-8192}"
NS="${NS:-8}"
MAXTOK="${MAXTOK:-256}"
LONGREPS="${LONGREPS:-300}"          # ~10 tok/rep -> ~3000-tok prompt (spans chunked-prefill chunks -> LSE)
MODE="${MODE:-cudagraph}"            # eager | cudagraph  (cudagraph is MANDATORY for usable decode: 6 -> 37 tok/s)
ARMS="${ARMS:-off on}"
TEMP="${TEMP:-0.0}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
BUILD_SO="${BUILD_SO:-auto}"         # auto | force | never
JOBS="${JOBS:-8}"
CG_CONFIG="${CG_CONFIG:-{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="${OUT:-results/glm47_mla_${ENGINE}_${STAMP}}"
mkdir -p "$OUT"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
SO_CACHE="$HOME/.cache/fa-v100-so/$ENGINE"   # per-engine .so (torch-ABI specific)
mkdir -p "$SO_CACHE"
SEED_SO_GLOB="$FA_DIR/build/lib.linux-x86_64-cpython-312/flash_attn_v100_cuda.*.so"

SUMMARY="$OUT/SUMMARY.txt"; : > "$SUMMARY"
note() { echo "[glm47-mla:$ENGINE] $*"; }
log()  { echo "$*" | tee -a "$SUMMARY"; }

clean_box_guard() {
    local any=0 i used pids
    for i in $(seq 0 $((TP-1))); do
        used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
        pids=$(nvidia-smi --id="$i" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        [[ "${used:-9999}" -gt 2000 || "${pids:-1}" -gt 0 ]] && any=1
    done
    [[ "$any" -eq 0 ]]
}

# import-test the cached .so INSIDE the target image (no GPU needed). torch MUST be
# imported FIRST so libtorch symbols are loaded before the extension resolves them —
# otherwise a perfectly good .so fails with "undefined symbol: ...c10...".
so_imports() {
    ls "$SO_CACHE"/flash_attn_v100_cuda*.so >/dev/null 2>&1 || return 1
    docker run --rm --entrypoint python3 \
        -v "$SO_CACHE":/falib:ro -e PYTHONPATH=/falib "$IMAGE" \
        -c "import torch; import flash_attn_v100_cuda as m; assert hasattr(m,'varlen_fwd'); print('OK')" 2>/dev/null | grep -q OK
}

# build the sm_70 .so against THIS image's torch into SO_CACHE.
# setup.py guards on torch.cuda.is_available(), so ONE GPU must be VISIBLE — but the
# compile is nvcc/CPU-bound and allocates no GPU memory (context-init probe only), so it
# is low-contention and runs fine even while the box is busy with training. BUILD_GPU
# selects which device to expose (default 0).
BUILD_GPU="${BUILD_GPU:-device=0}"
build_so() {
    log "  building ai-bond .so (sm_70) against $ENGINE_LABEL torch — nvcc/CPU compile, GPU $BUILD_GPU visible (no compute), ~minutes ..."
    rm -f "$SO_CACHE"/flash_attn_v100_cuda*.so
    docker run --rm --entrypoint bash --gpus "\"$BUILD_GPU\"" \
        -v "$FA_DIR":/fasrc -v "$SO_CACHE":/so "$IMAGE" -c '
            set -e
            cd /fasrc
            TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS='"$JOBS"' python3 setup.py build_ext \
                --build-lib /tmp/fablib --build-temp /tmp/fabtmp -j '"$JOBS"' >/tmp/fabuild.log 2>&1 \
                || { echo "BUILD FAILED — tail:"; tail -25 /tmp/fabuild.log; exit 1; }
            cp /tmp/fablib/flash_attn_v100_cuda*.so /so/
            echo "built: $(ls /so/flash_attn_v100_cuda*.so)"
        ' 2>&1 | tee -a "$SUMMARY"
}

ensure_so() {
    # seed the cache from the prebuilt host .so if empty (matches torch 2.11 = 0.21 image)
    if ! ls "$SO_CACHE"/flash_attn_v100_cuda*.so >/dev/null 2>&1; then
        # shellcheck disable=SC2086
        ls $SEED_SO_GLOB >/dev/null 2>&1 && cp $SEED_SO_GLOB "$SO_CACHE"/ 2>/dev/null || true
    fi
    if [[ "$BUILD_SO" == "force" ]]; then
        build_so || { log "ABORT: forced .so build failed"; exit 1; }
    fi
    if so_imports; then
        log "  ai-bond .so OK (imports under $ENGINE_LABEL)"
        return 0
    fi
    if [[ "$BUILD_SO" == "never" ]]; then
        log "ABORT: cached .so does not import under $IMAGE and BUILD_SO=never."
        log "       Run with BUILD_SO=force (or auto) to compile it for this engine."
        exit 1
    fi
    log "  cached .so does not import under $ENGINE_LABEL (torch-ABI mismatch) -> auto-build"
    build_so || { log "ABORT: .so build failed"; exit 1; }
    so_imports || { log "ABORT: freshly built .so still does not import"; exit 1; }
    log "  ai-bond .so OK (built + imports under $ENGINE_LABEL)"
}

log "GLM-4.7-Flash MLA-prefill A/B [$ENGINE_LABEL sm_70, MODE=$MODE, TP=$TP, maxlen=$MAXLEN] — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "model=$MODEL image=$IMAGE out=$OUT"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "ABORT: image $IMAGE missing"; exit 1; }
[[ -f "$MODEL/config.json" ]] || { log "ABORT: missing $MODEL"; exit 1; }
[[ -f "$FA_DIR/setup.py" ]] || { log "ABORT: flash-attention-v100 source not at $FA_DIR"; exit 1; }

arch=$(python3 -c "import json;print(json.load(open('$MODEL/config.json')).get('architectures',['?'])[0])" 2>/dev/null)
log "arch: $arch"

ensure_so   # per-engine .so ready (build is GPU-free, so do it before the clean-box gate)

# BUILD_ONLY=1: stop after the .so build. Run this in tmux to pre-warm the per-engine .so
# cache detached — survives disconnect/relocation. The compile is nvcc/CPU-bound (one GPU
# visible for torch's availability probe, but no GPU compute/memory), so it is safe to run
# while the box is busy. The later benchmark run then skips the compile.
#   e.g.:  tmux new -s faso019 -d 'ENGINE=019 BUILD_ONLY=1 ./tools/glm47_flash_mla_v100_ab.sh 2>&1 | tee /tmp/faso019.log'
if [[ "${BUILD_ONLY:-0}" == "1" ]]; then
    log "BUILD_ONLY=1 -> per-engine .so ready at $SO_CACHE ; exiting before serve."
    exit 0
fi

clean_box_guard || { log "SKIP: box busy (>2GB used or other CUDA apps on GPU 0..$((TP-1)))"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 1; }

case "$MODE" in
    eager)     MARGS=(--enforce-eager) ;;
    cudagraph) MARGS=(--compilation-config "$CG_CONFIG") ;;
    *) log "ABORT: unknown MODE=$MODE"; exit 1 ;;
esac
gpus="$(seq -s, 0 $((TP-1)))"

# Build a short and a long prompt (long -> spans chunked-prefill chunks -> LSE merge).
SHORT_PF="$OUT/.prompt_short.txt"; LONG_PF="$OUT/.prompt_long.txt"
cat > "$SHORT_PF" <<'EOF'
Explain in three sentences why mixture-of-experts models can be memory-heavy but compute-light at inference time.
EOF
python3 -c "open('$LONG_PF','w').write(('The quick brown fox jumps over the lazy dog. ' * ${LONGREPS}) + ' Summarize the repeated sentence and state how many times it appeared.')"

# Streaming probe -> TTFT (prefill UX) + steady-state decode tok/s, separated.
# prints TSV: TAG  NTOK  TTFT_s  DECODE_TPS  TOTAL_s  REP  SHA
probe() {  # $1=label  $2=prompt_file  $3=maxtok  $4=outfile
    local label="$1" pf="$2" mt="$3" of="$4" v
    v=$(python3 - "$PORT" "$SERVED" "$pf" "$mt" "$of" "$TEMP" <<'PY'
import sys, json, time, urllib.request, hashlib
port, served, pf, maxtok, of = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
temp = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
prompt = open(pf).read()
body = json.dumps({"model": served, "stream": True, "max_tokens": maxtok, "temperature": temp,
    "seed": 1234, "stream_options": {"include_usage": True},
    "messages": [{"role":"user","content": prompt}]}).encode()
req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                            data=body, headers={"Content-Type":"application/json"})
t0=time.time(); tf=tl=None; n=0; ch=[]; ut=0
try:
    with urllib.request.urlopen(req, timeout=2400) as rr:
        for raw in rr:
            line=raw.decode("utf-8","ignore").strip()
            if not line.startswith("data:"): continue
            d=line[5:].strip()
            if d=="[DONE]": break
            try: j=json.loads(d)
            except Exception: continue
            u=j.get("usage")
            if u and u.get("completion_tokens"): ut=int(u["completion_tokens"])
            c=j.get("choices") or []
            delta=c[0]["delta"].get("content") if c else None
            if delta:
                now=time.time()
                if tf is None: tf=now
                tl=now; n+=1; ch.append(delta)
    s="".join(ch); open(of,"w").write(s); w=s.split()
    rep=(max((w.count(x) for x in set(w)),default=0)/len(w)) if w else 1.0
    ttft=(tf-t0) if tf else float("nan")
    dwin=(tl-tf) if (tf and tl and n>1) else float("nan")
    mt2=ut if ut else n
    dtps=((mt2-1)/dwin) if (dwin and dwin>0) else float("nan")
    total=(tl-t0) if tl else float("nan")
    h=hashlib.sha256(s.encode()).hexdigest()[:16]
    ok=bool(s.strip()) and n>=5
    print(("OK" if ok else "BAD")+f"\t{mt2}\t{ttft:.2f}\t{dtps:.2f}\t{total:.2f}\t{rep:.3f}\t{h}")
except Exception as e:
    print(f"ERR\t0\tnan\tnan\tnan\t1.0\t{type(e).__name__}")
PY
)
    local tag tok tt dt tot rp sha
    tag=$(printf '%s' "$v"|cut -f1); tok=$(printf '%s' "$v"|cut -f2); tt=$(printf '%s' "$v"|cut -f3)
    dt=$(printf '%s' "$v"|cut -f4); tot=$(printf '%s' "$v"|cut -f5); rp=$(printf '%s' "$v"|cut -f6); sha=$(printf '%s' "$v"|cut -f7)
    log "    [$label] $tag  ttft=${tt}s  decode=${dt} tok/s  total=${tot}s  ${tok}tok  rep=$rp  sha=$sha"
    [[ -s "$of" ]] && log "    [$label] sample: $(head -c 240 "$of" | tr '\n' ' ')"
}

run_arm() {
    local flag="$1" label cname slog
    [[ "$flag" == "1" ]] && label="ON" || label="OFF"
    cname="glm47mla_${ENGINE}_${label}"; slog="$OUT/${label}_serve.log"
    log ""
    log "================== arm=$label (VLLM_V100_MLA_PREFILL=$flag) =================="
    docker rm -f "$cname" >/dev/null 2>&1 || true
    docker run --rm -i --name "$cname" --gpus "\"device=$gpus\"" \
        -v /mnt/models:/mnt/models:ro \
        -v "$PROJECT_ROOT":/work -w /work \
        -v "$SO_CACHE":/falib:ro \
        -e PYTHONPATH=/work/src:/falib \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
        -p ${PORT}:${PORT} --shm-size=16g \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
        -e VLLM_V100_MLA_PREFILL="$flag" \
        -e VLLM_V100_MLA_DECODE_CUDAGRAPH="${MLA_CG:-0}" \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        "$IMAGE" \
        python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
            --tensor-parallel-size "$TP" --dtype float16 "${MARGS[@]}" \
            --max-model-len "$MAXLEN" --max-num-seqs "$NS" \
            --gpu-memory-utilization 0.90 \
            --host 0.0.0.0 --port "$PORT" \
        </dev/null >"$slog" 2>&1 &
    local lpid=$! healthy=0 waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$lpid" 2>/dev/null || { note "  server exited before healthy (arm=$label)"; break; }
        sleep 5; waited=$((waited+5))
    done
    if (( healthy )); then
        log "  server healthy in ${waited}s"
        if [[ "$flag" == "1" ]]; then
            log "  -- warmup (absorb Triton JIT spikes; timing discarded) --"
            probe "$label/warmup" "$SHORT_PF" 32 "$OUT/${label}_warmup.txt" >/dev/null 2>&1 || true
        fi
        log "  -- short prompt (new_tokens/causal): TTFT + steady-state decode --"
        probe "$label/short" "$SHORT_PF" "$MAXTOK" "$OUT/${label}_short.txt"
        log "  -- long prompt (~${LONGREPS} reps ~3k tok -> chunked context/LSE): long-context TTFT --"
        probe "$label/long"  "$LONG_PF" 64 "$OUT/${label}_long.txt"
        local routed crash
        routed=$(grep -c "MLA prefill -> flash_attn_v100" "$slog" 2>/dev/null || true)
        crash=$(grep -c "only supports Ampere\|FlashAttention only" "$slog" 2>/dev/null || true)
        log "  evidence: MLA-route banner=$routed  ampere-crash=$crash"
    else
        log "  NOT HEALTHY in ${waited}s — tail of serve.log:"
        grep -iE "ampere|flashattention|error|traceback|mla" "$slog" 2>/dev/null | tail -8 | sed 's/^/    /' | tee -a "$SUMMARY"
    fi
    docker rm -f "$cname" >/dev/null 2>&1 || true
    sleep 3
}

for a in $ARMS; do
    [[ "$a" == "on" ]] && run_arm 1 || run_arm 0
done
log ""
log "DONE. Summary: $SUMMARY ; per-arm serve logs: $OUT/{OFF,ON}_serve.log ; outputs: $OUT/{OFF,ON}_{short,long}.txt"
