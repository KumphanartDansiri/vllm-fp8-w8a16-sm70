#!/usr/bin/env bash
# CHAPTER 1 — Reliability & Baseline benchmark (V100, vLLM 0.21+cu126, single stack).
# Message: "V100 is old but not dead — modern Qwen/Gemma models load, generate coherent text,
# and serve at useful decode rates." Compares OUR FP8 vs official FP16 (mid models) / GPTQ-Int4
# (122B, FP16 won't fit), with self-stability + faithfulness, on ONE standardized config.
#
# DESIGN (locked in memory project_v100_benchmark_matrix):
#   - 5 models × 2 variants = 10 cells; single stack vllm-v100:vllm021-cu126 (loads all incl gemma-4).
#   - Standardized: --max-model-len 8192, ns=8, cudagraph, --skip-mm-profiling (text bench; ~45x faster
#     cold start, decode unaffected), streaming steady-state decode (NEVER tokens/wall). FP8 = coalesced ON.
#   - Prompt suite: Q1 ×5 @4096 (Axis-1 self-stability + decode mean/variance); Q2..Q5 ×1 @1024
#     (Axis-2 FP8-vs-comparator faithfulness/coherence; Q5 = write software). max_tokens is per-request
#     → mixed lengths cost no reload.
#   - Records EVERY run's full text + sha256 to manifest.csv; ch1_report.py does Axis-1/Axis-2 after.
#   - 122B FP8-vs-Int4 = SPEED+coherence only (no FP16 gold fits); quality = bit-depth principle + coherence.
#   - GUARDS: GPU-clean + CPU-clean (refuses on contended box; CPU_GATE=0 to force). Run on an IDLE box.
#
# Usage:  ./tools/ch1_reliability_bench.sh                      # all 10 cells (~3-4h idle box)
#         ONLY=q27b ./tools/ch1_reliability_bench.sh            # one model (both variants)
#         ONLY=q27b-fp8 ./tools/ch1_reliability_bench.sh        # one cell
#         CPU_GATE=0 ONLY=q27b-fp8 SUITE_SMOKE=1 ...            # quick functional smoke (short gens)
#   then: python3 tools/ch1_report.py /tmp/v100_ch1/manifest.csv
# Env: IMAGE PORT HEALTH_TIMEOUT GPUMEM MAXLEN NS ONLY CPU_GATE SUITE_SMOKE UNROLL

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
CACHE_TAG="${CACHE_TAG:-021cu126}"
PORT="${PORT:-8021}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
GPUMEM="${GPUMEM:-0.92}"
MAXLEN="${MAXLEN:-8192}"
NS="${NS:-8}"
UNROLL="${UNROLL:-4}"
ONLY="${ONLY:-}"
CPU_GATE="${CPU_GATE:-1}"
CPU_FOREIGN_PCT="${CPU_FOREIGN_PCT:-50}"
CPU_LOAD_MAX="${CPU_LOAD_MAX:-6.0}"
SUITE_SMOKE="${SUITE_SMOKE:-0}"   # 1 = override all prompt lengths to 96 tok for a fast functional smoke
TAG="${TAG:-ch1}"                              # sub-chapter id: ch1.1=vLLM 0.21, ch1.2=vLLM 0.19 (set by ch1_chain.sh)
ENGINE_LABEL="${ENGINE_LABEL:-vLLM 0.21+cu126}"  # printed in the header/manifest so the engine is unambiguous
SERVED="ch1"
OUT="${OUT:-/tmp/v100_ch1}"   # overridable so chained legs (0.21 vs 0.19) don't clobber each other
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
MANIFEST="$OUT/manifest.csv"

# ─── EDITABLE PROMPT SUITE ───────────────────────────────────────────────────
# id | reps | max_tokens | label   (text is in prompt_text() below, keyed by id)
# Q1 is the ×5 self-stability + decode anchor (long gen). Q2–Q5 ×1 faithfulness probes.
PROMPT_SPECS=(
  "1|5|4096|essay"
  "2|1|1024|factual"
  "3|1|1024|reasoning"
  "4|1|1024|structure"
  "5|1|1024|code"
)
prompt_text() {
  case "$1" in
    1) echo "Write a detailed, multi-section essay on the history, geography, economy, and culture of France. Use clear subsections with headings and develop each at length." ;;
    2) echo "Explain, step by step, how a transformer neural network processes a sentence — from tokenization and embeddings through self-attention and feed-forward layers to the output distribution. Be precise and thorough." ;;
    3) echo "A train leaves city A at 9:00 traveling 60 km/h toward city B, 280 km away. A second train leaves city B at 9:30 traveling 80 km/h toward city A on the same line. At what clock time do they meet, and how far from city A? Show every step of your reasoning, then give the final answer." ;;
    4) echo "Summarize the main causes of World War I as exactly five bullet points. Each bullet must be a single sentence. Do not add any text before or after the five bullets." ;;
    5) echo "Implement a thread-safe LRU (least-recently-used) cache in Python supporting get(key) and put(key, value) with a fixed capacity. Use only the standard library. Include docstrings and a set of pytest unit tests covering eviction, update, and concurrency." ;;
    *) echo "Hello." ;;
  esac
}
# ─────────────────────────────────────────────────────────────────────────────

# label | model path | tp | prec(fp8|fp16|int4) | comparator-group
#   comparator-group pairs each model's variants so the report knows what to diff.
# ORDER = smaller-first, fail-fast, ALL TP4 first then the TP8 flagship LAST.
# Rationale: (1) cheap models surface bugs before the slow 122B; (2) every TP4 cell fits on a
# 4-GPU half (e.g. GPU_SET=4,5,6,7) leaving the other half free, while 122B (TP8) needs the whole
# box → doing it last lets the TP4 batch run on half a box (time-safety / GPU-split). Lead with
# q27b (already smoke-validated). 122B is SPEED+coherence only (Int4 not a faithfulness gold).
CELLS=(
  "q27b-fp8|/mnt/models/Qwen/Qwen3.6-27B-FP8|4|fp8|q27b"
  "q27b-fp16|/mnt/models/Qwen/Qwen3.6-27B|4|fp16|q27b"
  "g26b-fp8|/mnt/models/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic|4|fp8|g26b"
  "g26b-fp16|/mnt/models/google/gemma-4-26B-A4B-it|4|fp16|g26b"
  "g31b-fp8|/mnt/models/RedHatAI/gemma-4-31B-it-FP8-Dynamic|4|fp8|g31b"
  "g31b-fp16|/mnt/models/google/gemma-4-31B-it|4|fp16|g31b"
  "q35b-fp8|/mnt/models/Qwen/Qwen3.6-35B-A3B-FP8|4|fp8|q35b"
  "q35b-fp16|/mnt/models/Qwen/Qwen3.6-35B-A3B|4|fp16|q35b"
  "q122b-fp8|/mnt/models/Qwen/Qwen3.5-122B-A10B-FP8|8|fp8|q122b"
  "q122b-int4|/mnt/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4|8|int4|q122b"
)

note() { echo "[ch1] $*"; }
gpu_list_for_tp() { local n="$1" i out=""; for ((i=0;i<n;i++)); do out+="${out:+,}$i"; done; echo "$out"; }
clean_box_guard() {
    local gpus="$1" used pids any=0; IFS=',' read -ra idxs <<<"$gpus"
    for i in "${idxs[@]}"; do
        used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
        pids=$(nvidia-smi --id="$i" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        [[ "${used:-9999}" -gt 2000 || "${pids:-1}" -gt 0 ]] && any=1
    done
    [[ "$any" -eq 0 ]]
}
cpu_clean_guard() {
    [[ "$CPU_GATE" == "1" ]] || return 0
    local me load foreign; me="$(id -un)"; load=$(awk '{print $1}' /proc/loadavg)
    foreign=$(ps -eo user:32,%cpu --no-headers 2>/dev/null | awk -v me="$me" -v t="$CPU_FOREIGN_PCT" '$1!=me && ($2+0)>t {n++} END{print n+0}')
    [[ "$foreign" -gt 0 ]] && { note "CPU NOT CLEAN: $foreign foreign proc(s) >${CPU_FOREIGN_PCT}% (loadavg=$load). Run on idle box or CPU_GATE=0."; return 1; }
    awk -v l="$load" -v m="$CPU_LOAD_MAX" 'BEGIN{exit !(l>m)}' && { note "CPU NOT CLEAN: loadavg=$load > $CPU_LOAD_MAX"; return 1; }
    return 0
}

# Per-FP8-cell env: union of block-FP8 (Qwen) + CT (gemma Dynamic) coalesced/resident knobs.
# All feature-detected → the irrelevant ones no-op for a given quant family. FP16/Int4 → coalesced off.
fp8_env_args() {
  echo "-e VLLM_V100_FP8_COALESCED_GEMV=1 -e VLLM_V100_FP8_COALESCED_UNROLL=$UNROLL -e VLLM_V100_FP8_COALESCED_M_UNROLL=$UNROLL -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8 -e VLLM_V100_FP8_MOE_W13_COALESCED=1 -e VLLM_V100_FP8_MOE_FALLBACK=1 -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1 -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=128 -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1 -e VLLM_V100_CT_FP8_RESIDENT=1 -e VLLM_V100_CT_FP8_RESIDENT_SELFCHECK=1 -e VLLM_V100_CT_MOE_W13_RESIDENT=1 -e VLLM_V100_CT_MOE_W13_FREE_FP16=1 -e VLLM_V100_CT_MOE_W2_GROUPED=1 -e VLLM_V100_CT_MOE_W13_COALESCED=1"
}

run_cell() {
    local label="$1" model="$2" tp="$3" prec="$4" group="$5"
    local gpus cname slog
    # GPU_SET (e.g. "4,5,6,7") pins the run to a specific GPU half — for the 4-7 hedge while the
    # other half stays free. Must match the cell's TP count; otherwise skip (e.g. 122B TP8 can't
    # run on a 4-GPU set). Unset → default contiguous 0..tp-1.
    if [[ -n "${GPU_SET:-}" ]]; then
        local nset; nset=$(awk -F, '{print NF}' <<<"$GPU_SET")
        [[ "$nset" -ne "$tp" ]] && { echo "$label: SKIP (GPU_SET=$GPU_SET has $nset GPUs, cell needs TP=$tp)" | tee -a "$SUMMARY"; return; }
        gpus="$GPU_SET"
    else
        gpus=$(gpu_list_for_tp "$tp")
    fi
    cname="ch1_${label}"; slog="$OUT/${label}_serve.log"

    docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "$label: SKIP (image $IMAGE missing)" | tee -a "$SUMMARY"; return; }
    [[ -f "$model/config.json" ]] || { echo "$label: SKIP (missing $model)" | tee -a "$SUMMARY"; return; }
    clean_box_guard "$gpus" || { echo "$label: SKIP (GPUs $gpus busy)" | tee -a "$SUMMARY"; return 1; }
    cpu_clean_guard || { echo "$label: SKIP (CPU contended)" | tee -a "$SUMMARY"; return 1; }
    for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done

    local FP8ENV=""; [[ "$prec" == "fp8" ]] && FP8ENV="$(fp8_env_args)"
    [[ "$prec" != "fp8" ]] && FP8ENV="-e VLLM_V100_FP8_COALESCED_GEMV=0"
    note "=== $label ($prec) model=$(basename "$model") TP=$tp coalesced=$([[ $prec == fp8 ]] && echo ON || echo n/a) loadavg=$(awk '{print $1}' /proc/loadavg) ==="
    docker rm -f "$cname" >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    docker run --rm -i --name "$cname" --gpus "\"device=$gpus\"" \
        -v /mnt/models:/mnt/models:ro -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
        -p ${PORT}:${PORT} --shm-size=16g \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e VLLM_ATTENTION_BACKEND=TRITON_ATTN $FP8ENV \
        "$IMAGE" \
        python3 -m fp8_w8a16_sm70.vllm_serve --model "$model" --served-model-name "$SERVED" \
            --tensor-parallel-size "$tp" --dtype float16 \
            --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --max-model-len "$MAXLEN" --max-num-seqs "$NS" --skip-mm-profiling \
            --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill \
            --host 0.0.0.0 --port "$PORT" \
        </dev/null >"$slog" 2>&1 &
    local lpid=$!

    local healthy=0 waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$lpid" 2>/dev/null || { note "  server exited before healthy"; break; }
        sleep 10; waited=$((waited+10)); (( waited % 60 == 0 )) && note "  ...loading $label (${waited}s)"
    done
    if [[ "$healthy" != 1 ]]; then
        echo "$label: FAIL (never healthy) — $slog" | tee -a "$SUMMARY"
        grep -nE "Error|Traceback|no kernel image|out of memory|assert|ImportError" "$slog" | head -6
        docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; return 1
    fi
    # warmup (JIT + cudagraph capture)
    curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(python3 -c "import json;print(json.dumps({'model':'$SERVED','messages':[{'role':'user','content':'Say hi.'}],'max_tokens':16,'temperature':0}))")" >/dev/null 2>&1 || true

    local startup; startup=$(grep -oE "init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s" "$slog" | tail -1 | grep -oE "[0-9.]+ s$")
    note "  healthy (startup ${startup:-?}); running prompt suite..."

    # ── prompt suite ──
    local spec qid reps mtok qlabel pf q1_decodes=() q1_hashes=()
    for spec in "${PROMPT_SPECS[@]}"; do
        IFS='|' read -r qid reps mtok qlabel <<<"$spec"
        [[ "$SUITE_SMOKE" == "1" ]] && mtok=96
        pf="$OUT/.prompt_${qid}.txt"; prompt_text "$qid" > "$pf"
        local r
        for r in $(seq 1 "$reps"); do
            local of="$OUT/${label}_q${qid}_run${r}.txt"
            local v; v=$(python3 - "$PORT" "$SERVED" "$mtok" "$pf" "$of" <<'PY'
import sys, json, re, time, urllib.request, hashlib
port, served, maxtok, pf, of = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
prompt = open(pf).read()
body = json.dumps({"model": served, "stream": True, "max_tokens": maxtok, "temperature": 0,
    "ignore_eos": True, "stream_options": {"include_usage": True},
    "messages": [{"role":"user","content": prompt}]}).encode()
req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
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
    dt=(tl-tf) if (tf and tl and n>1) else float("nan")
    mt=ut if ut else n
    dtps=((mt-1)/dt) if (dt and dt>0) else float("nan")
    h=hashlib.sha256(s.encode()).hexdigest()[:16]
    ok=bool(s.strip()) and n>=10
    print(("OK" if ok else "BAD")+f"\t{mt}\t{rep:.3f}\t{ttft:.2f}\t{dtps:.2f}\t{h}")
except Exception as e:
    print("BAD\t0\t1.0\tnan\tnan\tnohash")
PY
)
            local tag tok rp tt dt sha
            tag=$(printf '%s' "$v"|cut -f1); tok=$(printf '%s' "$v"|cut -f2); rp=$(printf '%s' "$v"|cut -f3)
            tt=$(printf '%s' "$v"|cut -f4); dt=$(printf '%s' "$v"|cut -f5); sha=$(printf '%s' "$v"|cut -f6)
            echo "$TAG,$label,$prec,$group,$qid,$qlabel,$r,$mtok,$tag,$tok,$dt,$tt,$rp,$sha,$of" >> "$MANIFEST"
            note "    Q$qid($qlabel) r$r: $tag ${dt} tok/s ${tok}tok rep=$rp sha=$sha"
            if [[ "$qid" == "1" && "$tag" == "OK" ]]; then q1_decodes+=("$dt"); q1_hashes+=("$sha"); fi
        done
    done

    # per-cell Q1 summary: decode mean/stdev + self-stability (distinct sha among the 5)
    local q1stat="n/a" selfstab="n/a"
    if (( ${#q1_decodes[@]} > 0 )); then
        q1stat=$(python3 -c "
import statistics as st
a=[float(x) for x in '${q1_decodes[*]}'.split()]
print(f'mean={st.mean(a):.2f} sd={(st.pstdev(a) if len(a)>1 else 0):.2f} min={min(a):.2f} max={max(a):.2f} n={len(a)}')")
        local uniq; uniq=$(printf '%s\n' "${q1_hashes[@]}" | sort -u | wc -l)
        selfstab="$uniq distinct sha / ${#q1_hashes[@]} runs $([[ $uniq -eq 1 ]] && echo '(Exact: deterministic)' || echo '(drift: Stable-at-best)')"
    fi
    echo "$label [$prec]: Q1 decode $q1stat | self-stability: $selfstab | startup=${startup:-?}" | tee -a "$SUMMARY"
    note "  stopping $cname..."; docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
}

main() {
    : > "$SUMMARY"
    echo "chapter,model,prec,group,qid,qlabel,rep_idx,max_tokens,tag,tokens,decode_tps,ttft_s,rep_score,sha256_16,outfile" > "$MANIFEST"
    echo "$TAG Reliability bench [$ENGINE_LABEL, ns=$NS, cudagraph, skip-mm, coalesced-ON(fp8)] — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image $IMAGE missing"; exit 1; }
    local row label model tp prec group
    for row in "${CELLS[@]}"; do
        IFS='|' read -r label model tp prec group <<<"$row"
        # ONLY = comma/space-separated patterns; run the cell if its label matches ANY.
        if [[ -n "$ONLY" ]]; then
            local _m=0 _p; for _p in ${ONLY//,/ }; do [[ "$label" == *"$_p"* ]] && _m=1; done
            [[ "$_m" -eq 0 ]] && continue
        fi
        run_cell "$label" "$model" "$tp" "$prec" "$group" || true
        echo "" | tee -a "$SUMMARY"
    done
    note "==== CH1 SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
    note "NEXT: python3 tools/ch1_report.py $MANIFEST   # Axis-1 self-stability + Axis-2 FP8-vs-comparator tables"
}
main "$@"
