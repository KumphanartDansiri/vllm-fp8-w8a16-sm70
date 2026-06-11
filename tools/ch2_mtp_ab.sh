#!/usr/bin/env bash
# CHAPTER 2 — MTP (speculative decode) A/B/N on OUR FP8 (coalesced) path, or stock FP16 (PREC=fp16).
# Arms, one server load each, same model/config:
#   off    : Ch1 production config (cudagraph + skip-mm [+ FP8 coalesced when PREC=fp8]) — baseline.
#   mtp<k> : same + --speculative-config '{"method":"mtp","num_speculative_tokens":k}' for k in KLIST.
# Measures the THREE things TOGETHER (never acceptance alone — the 1catai garbage trap):
#   (1) SPEEDUP   : streaming steady-state decode tok/s per k vs off, + whole-wall tok/s cross-check.
#                   MTP delivers tokens in BURSTS → chunks-vs-tokens count logged as the burst diagnostic.
#   (2) ACCEPTANCE: /metrics accepted/draft (a perf signal ONLY) + raw vllm:spec_decode_* saved per arm.
#   (3) EXACTNESS : sha + saved text per arm (off-vs-k divergence figure). Spec-decode is lossless in
#                   MATH; FP-nondeterminism makes it diverge on MoE — report honestly + check COHERENCE.
# Plus: spec-init + cudagraph-capture lines harvested from each serve log (evidence the draft path is
# actually present and running under FULL_DECODE_ONLY capture).
#
# First datapoint (2026-06-11, 35B-A3B-FP8 k=1 @512tok): off=70.51 -> mtp=70.16 = 1.00x, accept=84.9%,
# DIFF-but-coherent. High acceptance does NOT imply a wall-clock win on V100 — Ch2's real question is
# "does MTP help AT ALL on the coalesced-FP8 V100 stack?"
#
# Usage:  ./tools/ch2_mtp_ab.sh                          # 35B-A3B-FP8, off vs k=1
#         KLIST="1 2" ./tools/ch2_mtp_ab.sh              # off vs k=1 vs k=2 (3 loads)
#         PREC=fp16 MODEL=/mnt/models/Qwen/Qwen3.6-35B-A3B ./tools/ch2_mtp_ab.sh   # stock comparator
#         MODEL=/mnt/models/Qwen/Qwen3.5-122B-A10B-FP8 TP=8 KLIST="1 2" ./tools/ch2_mtp_ab.sh
# Env: IMAGE MODEL TP PREC GENTOK KLIST (legacy KSPEC honored) GPUMEM MAXLEN HEALTH_TIMEOUT PORT OUT CHUNKED
#   CHUNKED=1 drops --no-enable-chunked-prefill (vLLM default ON). Default 0 = match Ch1 + the first ch2
#   datapoint for comparability; the known crash config for the flag was 122B-hybrid @28k ctx — we run 4k.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"; CACHE_TAG="${CACHE_TAG:-021cu126}"
MODEL="${MODEL:-/mnt/models/Qwen/Qwen3.6-35B-A3B-FP8}"
PREC="${PREC:-fp8}"            # fp8 = our coalesced/resident env ON (union, feature-detected); fp16 = stock
TP="${TP:-4}"; GENTOK="${GENTOK:-512}"; GPUMEM="${GPUMEM:-0.90}"; MAXLEN="${MAXLEN:-4096}"
KLIST="${KLIST:-${KSPEC:-1}}"  # space-separated num_speculative_tokens values, one mtp arm per value
CHUNKED="${CHUNKED:-0}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"; PORT="${PORT:-8021}"; SERVED="ch2"
OUT="${OUT:-/tmp/v100_ch2_mtp}"; mkdir -p "$OUT"; SUMMARY="$OUT/SUMMARY.txt"; : > "$SUMMARY"
PROMPT="Write a detailed, multi-section essay on the history, geography, economy, and culture of France. Use clear subsections with headings and develop each at length."
note(){ echo "[ch2-mtp] $*"; }
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
gpus(){ local n="$1" i o=""; for((i=0;i<n;i++));do o+="${o:+,}$i";done; echo "$o"; }
# Union of block-FP8 (Qwen) + CT (gemma Dynamic) knobs — same set as ch1_reliability_bench so the off
# arm IS the Ch1 production cell; irrelevant ones no-op per quant family.
FP8ENV="-e VLLM_V100_FP8_COALESCED_GEMV=1 -e VLLM_V100_FP8_COALESCED_UNROLL=4 -e VLLM_V100_FP8_COALESCED_M_UNROLL=4 -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8 -e VLLM_V100_FP8_MOE_W13_COALESCED=1 -e VLLM_V100_FP8_MOE_FALLBACK=1 -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1 -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=128 -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1 -e VLLM_V100_CT_FP8_RESIDENT=1 -e VLLM_V100_CT_FP8_RESIDENT_SELFCHECK=1 -e VLLM_V100_CT_MOE_W13_RESIDENT=1 -e VLLM_V100_CT_MOE_W13_FREE_FP16=1 -e VLLM_V100_CT_MOE_W2_GROUPED=1 -e VLLM_V100_CT_MOE_W13_COALESCED=1"
[[ "$PREC" != "fp8" ]] && FP8ENV="-e VLLM_V100_FP8_COALESCED_GEMV=0"
CPFLAG=(--no-enable-chunked-prefill); [[ "$CHUNKED" == "1" ]] && CPFLAG=()

clean(){ local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader|awk '{s+=$1}END{print s+0}'); [[ "$u" -lt 2000 ]]; }

run_arm(){ # run_arm <arm> <k>   k="" -> baseline
    local arm="$1" k="${2:-}" g; g=$(gpus "$TP"); local cname="ch2_${arm}" slog="$OUT/${arm}_serve.log" sfile="$OUT/${arm}_sample.txt"
    local MARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
    [[ -n "$k" ]] && MARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${k}}")
    clean || { echo "$arm: SKIP (GPUs busy)"|tee -a "$SUMMARY"; return 1; }
    note "=== arm=$arm  model=$(basename "$MODEL") prec=$PREC TP=$TP $([[ -n $k ]] && echo "MTP k=$k" || echo baseline) ==="
    docker rm -f "$cname" >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    docker run --rm -i --name "$cname" --gpus "\"device=$g\"" \
        -v /mnt/models:/mnt/models:ro -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
        -p ${PORT}:${PORT} --shm-size=16g \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ATTENTION_BACKEND=TRITON_ATTN $FP8ENV \
        "$IMAGE" python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
            --tensor-parallel-size "$TP" --dtype float16 --max-model-len "$MAXLEN" --max-num-seqs 8 \
            --skip-mm-profiling --gpu-memory-utilization "$GPUMEM" "${CPFLAG[@]}" \
            "${MARGS[@]}" --host 0.0.0.0 --port "$PORT" \
        </dev/null >"$slog" 2>&1 &
    local lpid=$! healthy=0 w=0
    while (( w < HEALTH_TIMEOUT )); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$lpid" 2>/dev/null || { note "  server exited"; break; }; sleep 10; w=$((w+10)); ((w%60==0))&&note "  ...loading $arm (${w}s)"
    done
    if [[ "$healthy" != 1 ]]; then
        echo "$arm: FAIL (never healthy) — $slog"|tee -a "$SUMMARY"
        grep -nE "Error|Traceback|speculative|assert|no kernel|not supported|NotImplemented" "$slog"|head -8|tee -a "$SUMMARY"
        docker stop "$cname">/dev/null 2>&1||true; wait "$lpid" 2>/dev/null||true; return 1
    fi
    # evidence: spec-decode init + cudagraph capture lines from the serve log
    if [[ -n "$k" ]]; then grep -inE "specul|mtp|draft" "$slog" | head -8 > "$OUT/${arm}_spec_init.txt" || true; fi
    grep -iE "graph.*captur|captur.*graph|cudagraph" "$slog" | tail -3 > "$OUT/${arm}_cudagraph.txt" || true
    # warmup (JIT + cudagraph capture)
    curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(python3 -c "import json;print(json.dumps({'model':'$SERVED','messages':[{'role':'user','content':'hi'}],'max_tokens':16,'temperature':0}))")" >/dev/null 2>&1||true
    # timed streaming steady-state decode
    local v; v=$(python3 - "$PORT" "$SERVED" "$GENTOK" "$sfile" "$PROMPT" <<'PY'
import sys,json,time,urllib.request,hashlib
port,served,tok,sfile,prompt=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[5]
body=json.dumps({"model":served,"stream":True,"max_tokens":tok,"temperature":0,"ignore_eos":True,
 "stream_options":{"include_usage":True},"messages":[{"role":"user","content":prompt}]}).encode()
req=urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
t0=time.time();tf=tl=None;n=0;ch=[];ut=0
try:
 with urllib.request.urlopen(req,timeout=1800) as r:
  for raw in r:
   ln=raw.decode("utf-8","ignore").strip()
   if not ln.startswith("data:"):continue
   d=ln[5:].strip()
   if d=="[DONE]":break
   try:j=json.loads(d)
   except:continue
   u=j.get("usage")
   if u and u.get("completion_tokens"):ut=int(u["completion_tokens"])
   c=j.get("choices") or [];dl=c[0]["delta"].get("content") if c else None
   if dl:
    now=time.time()
    if tf is None:tf=now
    tl=now;n+=1;ch.append(dl)
 s="".join(ch);open(sfile,"w").write(s);w=s.split()
 rep=(max((w.count(x) for x in set(w)),default=0)/len(w)) if w else 1.0
 ttft=(tf-t0) if tf else float("nan")
 dt=(tl-tf) if (tf and tl and n>1) else float("nan");mt=ut if ut else n
 dtps=((mt-1)/dt) if (dt and dt>0) else float("nan")
 wall=(mt/(tl-t0)) if (tl and tl>t0) else float("nan")
 h=hashlib.sha256(s.encode()).hexdigest()[:16]
 print(("OK" if (s.strip() and n>=20) else "BAD")+f"\t{mt}\t{rep:.3f}\t{ttft:.2f}\t{dtps:.2f}\t{wall:.2f}\t{n}\t{h}")
except Exception as e:print("BAD\t0\t1.0\tnan\tnan\tnan\t0\tnohash")
PY
)
    local tag tok rep ttft dt wall nch sha
    tag=$(printf '%s' "$v"|cut -f1);tok=$(printf '%s' "$v"|cut -f2);rep=$(printf '%s' "$v"|cut -f3);ttft=$(printf '%s' "$v"|cut -f4)
    dt=$(printf '%s' "$v"|cut -f5);wall=$(printf '%s' "$v"|cut -f6);nch=$(printf '%s' "$v"|cut -f7);sha=$(printf '%s' "$v"|cut -f8)
    local acc="n/a"
    if [[ -n "$k" ]]; then
        curl -s "http://localhost:${PORT}/metrics" 2>/dev/null | grep -E "^vllm:spec_decode" > "$OUT/${arm}_metrics_spec.txt" || true
        acc=$(python3 -c "
a=d=0.0
for ln in open('$OUT/${arm}_metrics_spec.txt'):
 if ln.startswith('vllm:spec_decode_num_accepted_tokens') and 'total' in ln:
  try:a=float(ln.split()[-1])
  except:pass
 if ln.startswith('vllm:spec_decode_num_draft_tokens') and 'total' in ln:
  try:d=float(ln.split()[-1])
  except:pass
print(f'{a/d:.1%}' if d>0 else 'n/a')" 2>/dev/null||echo n/a)
    fi
    echo "$arm: $tag decode=${dt} tok/s (wall=${wall}) tok=$tok chunks=$nch ttft=${ttft}s rep=$rep sha=$sha accept=$acc" | tee -a "$SUMMARY"
    echo "$dt" > "$OUT/.${arm}_decode"; echo "$sha" > "$OUT/.${arm}_sha"; echo "$acc" > "$OUT/.${arm}_accept"
    note "  stopping $cname"; docker stop "$cname">/dev/null 2>&1||true; wait "$lpid" 2>/dev/null||true
}

main(){
    echo "CH2 MTP A/B/N — $(basename "$MODEL") prec=$PREC TP=$TP KLIST='$KLIST', cudagraph+skip-mm$([[ $PREC == fp8 ]] && echo "+coalesced"), maxlen=$MAXLEN, $GENTOK tok, chunked=$CHUNKED — $(date -u +%FT%TZ)" >> "$SUMMARY"
    docker image inspect "$IMAGE">/dev/null 2>&1||{ note "image missing";exit 1; }
    [[ -f "$MODEL/config.json" ]]||{ echo "FAIL: missing $MODEL"|tee -a "$SUMMARY";exit 1; }
    run_arm off "" || true
    local k; for k in $KLIST; do run_arm "mtp${k}" "$k" || true; done
    echo "" | tee -a "$SUMMARY"
    # verdict per k: speedup + acceptance + exactness vs off
    local od osha; od=$(cat "$OUT/.off_decode" 2>/dev/null||echo nan); osha=$(cat "$OUT/.off_sha" 2>/dev/null||echo x)
    for k in $KLIST; do
        local md msha macc spd exact
        md=$(cat "$OUT/.mtp${k}_decode" 2>/dev/null||echo nan); msha=$(cat "$OUT/.mtp${k}_sha" 2>/dev/null||echo y)
        macc=$(cat "$OUT/.mtp${k}_accept" 2>/dev/null||echo n/a)
        spd=$(python3 -c "print(f'{$md/$od:.2f}x')" 2>/dev/null||echo n/a)
        [[ "$osha" == "$msha" && "$osha" != "x" ]] && exact="EXACT (token-for-token == baseline)" || exact="DIFF (check coherence: diff $OUT/off_sample.txt $OUT/mtp${k}_sample.txt)"
        echo "k=$k: SPEEDUP off=$od -> mtp=$md tok/s = $spd | accept=$macc | EXACTNESS: $exact" | tee -a "$SUMMARY"
    done
    echo "  (acceptance is a perf signal only; usability = coherence of the mtp output; spec-init/capture evidence in ${OUT}/*_spec_init.txt, *_cudagraph.txt)" | tee -a "$SUMMARY"
    note "==== CH2 MTP SUMMARY ===="; cat "$SUMMARY"
}
main "$@"
