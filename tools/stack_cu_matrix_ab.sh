#!/usr/bin/env bash
# Stack-decomposition matrix: cu128 vs cu126 vs engine, FP8 vs FP16, on the small
# Qwen3.6 test bed (TP=4). Answers the live 0.21-perf-regression question
# ([[project_vllm021_perf_regression]]): how much of the 0.19->0.21 decode regression is
# the CUDA wheel (cu126, forced on 0.21 to land in the <12.8 CMake branch that keeps sm_70
# unpatched) vs the engine+torch itself — and is the effect FP8-path-specific or general
# (FP16 control).
#
# THREE STACK CELLS (identical FP8 source mounted into each, kernel JIT-compiled per-image,
# ABI-isolated cache tag):
#   c1_019cu128 : vllm 0.19.0  torch 2.10.0+cu128  CUDA 12.8   (vllm-v100-py312:vllm019)
#   c2_019cu126 : vllm 0.19.0  torch 2.10.0+cu126  CUDA 12.6   (vllm-v100-py312:vllm019-cu126)
#   c3_021cu126 : vllm 0.21.0  torch 2.11.0+cu126  CUDA 12.6   (vllm-v100:vllm021-cu126)
# DECOMPOSE: (c1->c2)=cu128-vs-cu126 at fixed 0.19/2.10 ; (c2->c3)=pure 0.19->0.21 at fixed cu126.
#
# FOUR MODEL-PRECISION ROWS (TP=4): 27B & 35B-A3B, FP8 (our kernel) and FP16 (cuBLAS ceiling).
# FP16 runs the SAME entrypoint — the FP8 patches are feature-detected and no-op on quant=none.
#
# STANDARDIZED, PUBLISHABLE PARAMS (the whole point — consistent across every cell):
#   max-model-len 8192 ; generate 4096 tok with ignore_eos (depth-honest, not shallow-inflated) ;
#   temperature 0 ; max-num-seqs 8 ; cudagraph ; streaming steady-state decode (NEVER tokens/wall).
#   FP8 kernel PINNED to A.3 (coalesced OFF) so FP8 work is byte-identical across cells and the
#   ONLY variable is the stack. (COALESCED=1 = a separate production pass.)
#
# CONTAMINATION: single-stream decode is CPU-sensitive; son's aiagent procs can peg cores ->
# ~+-10% noise. Each row logs loadavg. NRUN repeats average it.
#
# Usage (KERNEL-FIRST SMOKE — validate load/compile/coherence on all 3 engines, cheap):
#   ONLY_MODEL=q27b-fp8 MAXTOK=128 NRUN=1 ./tools/stack_cu_matrix_ab.sh
# Usage (FULL standardized batch, ~1h):
#   ./tools/stack_cu_matrix_ab.sh
# Filters: ONLY_MODEL=q35b-fp8  ONLY_CELL=c2   (substring match)
# Env: PORT HEALTH_TIMEOUT MAXTOK GPUMEM MAXLEN MODE NS NRUN COALESCED UNROLL ONLY_MODEL ONLY_CELL

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

PORT="${PORT:-8021}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3000}"
MAXTOK="${MAXTOK:-4096}"        # generation length (ignore_eos) — depth-honest standard
GPUMEM="${GPUMEM:-0.92}"
MAXLEN="${MAXLEN:-8192}"
MODE="${MODE:-cudagraph}"
NS="${NS:-8}"
NRUN="${NRUN:-2}"
COALESCED="${COALESCED:-0}"     # 0 = pure A.3 (clean stack test); 1 = coalesced ON (production pass)
UNROLL="${UNROLL:-4}"
SKIP_MM="${SKIP_MM:-1}"         # DEFAULT 1: pass --skip-mm-profiling (skip vision-encoder dummy at
                                # startup). This is a TEXT-decode benchmark — the max-dummy-image vision
                                # profile is ~98% of cold-start (MEASURED: 853s->19s, ~45x) and pure noise
                                # for our purpose; decode tok/s is UNAFFECTED. Recorded in the header +
                                # per-cell banner so 19s startups are never compared against old 850s ones.
                                # SKIP_MM=0 ONLY to test multimodal capacity/profiling behavior.
MM_OPTS=()
[[ "$SKIP_MM" == "1" ]] && MM_OPTS+=(--skip-mm-profiling)
ONLY_MODEL="${ONLY_MODEL:-}"
ONLY_CELL="${ONLY_CELL:-}"
SERVED="stackab"

if [[ "$MODE" == "cudagraph" ]]; then
    EXEC_OPTS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
else
    EXEC_OPTS=(--enforce-eager)
fi

# row-label | model | tp | precision(fp8|fp16)
MODELS=(
  "q27b-fp8|/mnt/models/Qwen/Qwen3.6-27B-FP8|4|fp8"
  "q27b-fp16|/mnt/models/Qwen/Qwen3.6-27B|4|fp16"
  "q35b-fp8|/mnt/models/Qwen/Qwen3.6-35B-A3B-FP8|4|fp8"
  "q35b-fp16|/mnt/models/Qwen/Qwen3.6-35B-A3B|4|fp16"
)
# cell-label | image | cache-tag
CELLS=(
  "c1_019cu128|vllm-v100-py312:vllm019|019cu128"
  "c2_019cu126|vllm-v100-py312:vllm019-cu126|019cu126"
  "c3_021cu126|vllm-v100:vllm021-cu126|021cu126"
)

OUT=/tmp/v100_stack_cu_matrix
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
note() { echo "[stack-ab] $*"; }

gpu_list_for_tp() { local n="$1" i out=""; for ((i=0;i<n;i++)); do out+="${out:+,}$i"; done; echo "$out"; }

# CPU-clean gate (perf-regression memory: single-stream decode is CPU-sensitive; son's
# aiagent procs peg cores -> ~+-10% noise that drowns the cu126-vs-cu128 deltas). For a
# PUBLISHABLE perf run the box must be CPU-idle too, not just GPU-idle. Signals:
#   - any FOREIGN-user process burning > CPU_FOREIGN_PCT (default 50) %CPU  -> contended
#   - 1-min loadavg above CPU_LOAD_MAX (default 6.0)                        -> contended
# Set CPU_GATE=0 to disable (e.g. the works/doesn't functional smoke, where noise is fine).
CPU_GATE="${CPU_GATE:-1}"
CPU_FOREIGN_PCT="${CPU_FOREIGN_PCT:-50}"
CPU_LOAD_MAX="${CPU_LOAD_MAX:-6.0}"
cpu_clean_guard() {
    [[ "$CPU_GATE" == "1" ]] || return 0
    local me load foreign
    me="$(id -un)"
    load=$(awk '{print $1}' /proc/loadavg)
    foreign=$(ps -eo user:32,%cpu --no-headers 2>/dev/null \
        | awk -v me="$me" -v t="$CPU_FOREIGN_PCT" '$1!=me && ($2+0)>t {n++} END{print n+0}')
    if [[ "$foreign" -gt 0 ]]; then
        note "CPU NOT CLEAN: $foreign foreign-user proc(s) > ${CPU_FOREIGN_PCT}% CPU (aiagent/training?). loadavg=$load"
        ps -eo user:16,pid,%cpu,comm --sort=-%cpu --no-headers 2>/dev/null | awk -v me="$me" '$1!=me' | head -4
        return 1
    fi
    if awk -v l="$load" -v m="$CPU_LOAD_MAX" 'BEGIN{exit !(l>m)}'; then
        note "CPU NOT CLEAN: loadavg(1m)=$load > $CPU_LOAD_MAX"; return 1
    fi
    return 0
}
clean_box_guard() {
    local gpus="$1" used pids any=0; IFS=',' read -ra idxs <<<"$gpus"
    for i in "${idxs[@]}"; do
        used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
        pids=$(nvidia-smi --id="$i" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        [[ "${used:-9999}" -gt 2000 || "${pids:-1}" -gt 0 ]] && any=1
    done
    [[ "$any" -eq 0 ]]
}

run_one() {
    local mlabel="$1" model="$2" tp="$3" prec="$4" clabel="$5" image="$6" ctag="$7"
    local gpus cname slog sfile tag
    gpus=$(gpu_list_for_tp "$tp")
    cname="stackab_${mlabel}_${clabel}"; slog="$OUT/${mlabel}_${clabel}_serve.log"; sfile="$OUT/${mlabel}_${clabel}_sample.txt"

    docker image inspect "$image" >/dev/null 2>&1 || { echo "$mlabel/$clabel: SKIP (image $image missing)" | tee -a "$SUMMARY"; return; }
    [[ -f "$model/config.json" ]] || { echo "$mlabel/$clabel: SKIP (missing $model)" | tee -a "$SUMMARY"; return; }
    clean_box_guard "$gpus" || { echo "$mlabel/$clabel: SKIP (GPUs $gpus busy)" | tee -a "$SUMMARY"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; return 1; }
    cpu_clean_guard || { echo "$mlabel/$clabel: SKIP (CPU contended — defer to idle box; set CPU_GATE=0 to force)" | tee -a "$SUMMARY"; return 1; }

    for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${ctag}-$s"; done
    local cpu_load; cpu_load=$(awk '{print $1}' /proc/loadavg)
    note "=== $mlabel ($prec) @ $clabel  image=$image cache=$ctag  (TP=$tp ns=$NS gen=$MAXTOK coal=$COALESCED skip_mm=$SKIP_MM loadavg=$cpu_load) ==="

    docker rm -f "$cname" >/dev/null 2>&1 || true
    docker run --rm -i --name "$cname" --gpus "\"device=$gpus\"" \
        -v /mnt/models:/mnt/models:ro \
        -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
        -v "$HOME/.cache/vllm-v100-${ctag}-torchext:/root/.cache/torch_extensions" \
        -v "$HOME/.cache/vllm-v100-${ctag}-triton:/root/.triton" \
        -v "$HOME/.cache/vllm-v100-${ctag}-torch:/root/.cache/torch" \
        -v "$HOME/.cache/vllm-v100-${ctag}-inductor:/tmp/torchinductor_root" \
        -p ${PORT}:${PORT} --shm-size=16g \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
        -e VLLM_V100_FP8_COALESCED_GEMV="$COALESCED" \
        -e VLLM_V100_FP8_COALESCED_UNROLL="$UNROLL" \
        -e VLLM_V100_FP8_COALESCED_M_UNROLL="$UNROLL" \
        -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX="$([[ "$COALESCED" == "1" ]] && echo 8 || echo 1)" \
        -e VLLM_V100_FP8_MOE_W13_COALESCED="$COALESCED" \
        -e VLLM_V100_FP8_MOE_FALLBACK=1 \
        -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1 \
        -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=128 \
        -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1 \
        -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        "$image" \
        python3 -m fp8_w8a16_sm70.vllm_serve --model "$model" --served-model-name "$SERVED" \
            --tensor-parallel-size "$tp" --dtype float16 "${EXEC_OPTS[@]}" "${MM_OPTS[@]}" \
            --max-model-len "$MAXLEN" --max-num-seqs "$NS" \
            --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill \
            --host 0.0.0.0 --port "$PORT" \
        </dev/null >"$slog" 2>&1 &
    local lpid=$!

    local healthy=0 waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$lpid" 2>/dev/null || { note "  server exited before healthy"; break; }
        sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "  ...loading $mlabel@$clabel (${waited}s)"
    done
    if [[ "$healthy" != 1 ]]; then
        echo "$mlabel/$clabel: FAIL (never healthy) — $slog" | tee -a "$SUMMARY"
        grep -nE "Error|Traceback|no kernel image|out of memory|assert|ImportError|ModuleNotFound" "$slog" | head -8
        docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; return 1
    fi

    # warmup (JIT + cudagraph capture)
    curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(python3 -c "import json;print(json.dumps({'model':'$SERVED','messages':[{'role':'user','content':'Say hi.'}],'max_tokens':16,'temperature':0}))")" >/dev/null 2>&1 || true

    local runs=() hashes=() r
    for r in $(seq 1 "$NRUN"); do
        local runfile="${sfile%.txt}_run${r}.txt"
        local v; v=$(python3 - "$PORT" "$SERVED" "$MAXTOK" "$runfile" <<'PY'
import sys, json, re, time, urllib.request, hashlib
port, served, maxtok, sfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
body = json.dumps({"model": served, "stream": True, "max_tokens": maxtok, "temperature": 0,
    "ignore_eos": True, "stream_options": {"include_usage": True},
    "messages": [{"role": "user", "content": "Write a detailed multi-paragraph essay about the history, geography, and culture of France, then continue with science, technology, and philosophy."}]}).encode()
req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
t0=time.time(); tf=tl=None; n=0; ch=[]; ut=0
try:
    with urllib.request.urlopen(req, timeout=1800) as rr:
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
    s="".join(ch).strip(); open(sfile,"w").write(s); w=s.split()
    rep=(max((w.count(x) for x in set(w)),default=0)/len(w)) if w else 1.0
    dt=(tl-tf) if (tf and tl and n>1) else float("nan")
    mt=ut if ut else n
    dtps=((mt-1)/dt) if (dt and dt>0) else float("nan")
    ok=bool(s) and n>=20 and rep<0.35
    h=hashlib.sha256(s.encode()).hexdigest()[:16]   # exact-greedy fingerprint for Exact/Stable grading + cross-machine compare
    print(("OK" if ok else "BAD")+f"\t{ut}\t{rep:.2f}\t{dtps:.2f}\t{h}")
except Exception as e:
    print(f"BAD\t0\t1.0\tnan\tnohash")
PY
)
        local t2 dtps hsh; t2=$(printf '%s' "$v"|cut -f1); dtps=$(printf '%s' "$v"|cut -f4); hsh=$(printf '%s' "$v"|cut -f5)
        note "    run $r/$NRUN: $t2 decode=${dtps} tok/s ($(printf '%s' "$v"|cut -f2) tok, rep=$(printf '%s' "$v"|cut -f3), sha=$hsh)"
        # append to the raw manifest (CSV) — full outputs live in *_run*.txt for later presentation/compare
        echo "$mlabel,$clabel,$prec,$r,$t2,$(printf '%s' "$v"|cut -f2),${dtps},$hsh,$runfile" >> "$OUT/manifest.csv"
        [[ "$t2" == "OK" ]] && { runs+=("$dtps"); hashes+=("$hsh"); }
    done

    local mean="nan" tag2="BAD"
    if (( ${#runs[@]} > 0 )); then
        mean=$(python3 -c "a=[float(x) for x in '${runs[*]}'.split()];print(f'{sum(a)/len(a):.2f}')")
        tag2="OK"
    fi
    # run-to-run determinism (Exact-candidate within this config) vs drift (Stable) — same-config only;
    # cross-precision (FP8 vs FP16) is NEVER Exact (different numerics) — judged later from the saved text.
    local det="n/a"
    if (( ${#hashes[@]} >= 2 )); then
        det="det"; local h0="${hashes[0]}" hx
        for hx in "${hashes[@]}"; do [[ "$hx" != "$h0" ]] && det="drift"; done
    fi
    # kernel proof (FP8 rows only): variant banner + coalesced banners
    local kern; kern=$(grep -oE "kernel variant counts after [0-9]+ calls: [^\"]*" "$slog" | tail -1)
    # startup (init engine = profile + kv cache + warmup); with skip_mm=1 this excludes the vision-encoder profile
    local startup; startup=$(grep -oE "init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s" "$slog" | tail -1 | grep -oE "[0-9.]+ s$")
    local snip; snip=$(tr '\n' ' ' < "${sfile%.txt}_run1.txt" 2>/dev/null | sed 's/  */ /g' | cut -c1-90)
    echo "$mlabel/$clabel [$prec]: $tag2 mean-decode=${mean} tok/s (${#runs[@]} runs: ${runs[*]:-none}) | startup=${startup:-?} | run2run=${det} sha=${hashes[0]:-none} ${kern:+| $kern}" | tee -a "$SUMMARY"
    echo "        \"$snip\"" | tee -a "$SUMMARY"
    note "  stopping $cname..."; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
}

main() {
    : > "$SUMMARY"
    echo "Stack cu/engine × FP8/FP16 matrix [TP4 Qwen3.6 test bed, ns=$NS gen=$MAXTOK maxlen=$MAXLEN mode=$MODE coalesced=$COALESCED skip_mm=$SKIP_MM] — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
    [[ "$SKIP_MM" == "1" ]] && echo "  NOTE skip_mm=1: --skip-mm-profiling ON -> startup is ~19s not ~850s (vision-encoder profile skipped); decode tok/s UNAFFECTED. Startup here is NOT comparable to skip_mm=0 runs." >> "$SUMMARY"
    # raw-output manifest: every run's full text saved to *_run*.txt; this CSV indexes them + sha for compare
    [[ -f "$OUT/manifest.csv" ]] || echo "model,cell,prec,run,tag,tokens,decode_tps,sha256_16,outfile" > "$OUT/manifest.csv"
    local mrow clabel image ctag mlabel model tp prec crow
    for mrow in "${MODELS[@]}"; do
        IFS='|' read -r mlabel model tp prec <<<"$mrow"
        [[ -n "$ONLY_MODEL" && "$mlabel" != *"$ONLY_MODEL"* ]] && continue
        for crow in "${CELLS[@]}"; do
            IFS='|' read -r clabel image ctag <<<"$crow"
            [[ -n "$ONLY_CELL" && "$clabel" != *"$ONLY_CELL"* ]] && continue
            run_one "$mlabel" "$model" "$tp" "$prec" "$clabel" "$image" "$ctag" || true
        done
        echo "" | tee -a "$SUMMARY"
    done
    note "==== STACK MATRIX SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
    echo "  DECOMPOSE per model: (c1->c2)=cu128-vs-cu126 ; (c2->c3)=pure 0.19->0.21. FP16 rows = the control + ceiling." | tee -a "$SUMMARY"
}
main "$@"
