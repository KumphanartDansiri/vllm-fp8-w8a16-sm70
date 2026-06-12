#!/usr/bin/env bash
# Volta FP16 fused-MoE num_stages A/B (vLLM 0.21 + cu126, V100).
# HYPOTHESIS (memory project_volta_moe_fp16_patch): stock FP16 MoE decode is slow on sm_70
# because the fused_moe config-miss falls back to get_default_config's num_stages=4 (M<=32),
# which assumes cp.async (sm_80+); V100 has none, so Triton multi-buffers 96KB SMEM for no
# overlap and craters occupancy. Decode tile at M<=32: BLOCK 16/64/128, warps=4, group=1.
#
# DESIGN: hold tile FIXED, sweep num_stages on the decode-relevant M<=32 config entries only
# (M>32 entries replicate the default exactly in every arm). Arms:
#   base = no override (the recorded 15.7/10.2 tok/s comparator path; expects the
#          "Using default MoE config" warning in serve log)
#   s4   = config file replicating the default EXACTLY (incl. num_stages=4) — control arm:
#          proves the VLLM_TUNED_CONFIG_FOLDER mechanism itself is perf-neutral
#   s3 / s2 = same tile, num_stages 3 / 2 for M<=32
# Serve config = the Ch1 q35b-fp16 / g26b-fp16 cell verbatim (TP4 fp16 cudagraph
# FULL_DECODE_ONLY maxlen8192 ns8 TRITON_ATTN, chunked prefill off for comparability).
#
# Usage:  ./tools/moe_stages_ab_vllm021.sh                 # q35b, arms base s4 s3 s2
#         MODEL_KEY=g26b ./tools/moe_stages_ab_vllm021.sh  # gemma 26B-A4B
#         ARMS="base s2" NRUN=2 ...                        # subset / quicker
# Env: MODEL_KEY ARMS IMAGE PORT GPUS NRUN GENTOK MAXLEN NS GPUMEM OUT
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

MODEL_KEY="${MODEL_KEY:-q35b}"
case "$MODEL_KEY" in
  q35b) MODEL=/mnt/models/Qwen/Qwen3.6-35B-A3B;        E=256; NSHARD=128 ;;
  g26b) MODEL=/mnt/models/google/gemma-4-26B-A4B-it;   E=128; NSHARD=176 ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY (q35b|g26b)"; exit 1 ;;
esac

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
CACHE_TAG="${CACHE_TAG:-021cu126}"
ARMS="${ARMS:-base s4 s3 s2}"
TP="${TP:-4}"
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-8023}"
MAXLEN="${MAXLEN:-8192}"
NS="${NS:-8}"
GENTOK="${GENTOK:-1024}"
NRUN="${NRUN:-3}"
GPUMEM="${GPUMEM:-0.92}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="${OUT:-results/moe_stages_ab_${MODEL_KEY}_${STAMP}}"
SERVED="moeab"
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
: > "$SUMMARY"

note(){ echo "[moe-ab] $*"; }
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done

clean_box_guard() {
    local used pids any=0 i; IFS=',' read -ra idxs <<<"$GPUS"
    for i in "${idxs[@]}"; do
        used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
        pids=$(nvidia-smi --id="$i" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        [[ "${used:-9999}" -gt 2000 || "${pids:-1}" -gt 0 ]] && any=1
    done
    [[ "$any" -eq 0 ]]
}

# Emit a tuned-config JSON mirroring get_default_config(dtype=None) per M, with num_stages
# for M<=32 forced to $2. M>32 entries always replicate the default (stages=3) so prefill
# is identical across arms and the sweep isolates the decode path.
# Mode "kbest" instead applies the microbench winners (tools/moe_decode_tile_sweep.py,
# 2026-06-12 M=1..16 sweep): BLOCK_K<=64 everywhere small-M (the spill fix), per-M tile:
# M<=4 -> 16/32/64 w4 s2; M=8..64 -> 16/128/64 w8 s2 (fat-N wins 25% at M>=8).
write_cfg() {  # $1=dir $2=stages_small_m|"kbest"
    local dir="$1" st="$2"
    mkdir -p "$dir"
    python3 - "$dir" "$st" "$E" "$NSHARD" <<'PY'
import json, sys
d, st, E, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cfg = {}
for M in [1,2,4,8,16,24,32,48,64,96,128,256,512,1024,1536,2048,3072,4096]:
    bm = 16 if M<=32 else (32 if M<=96 else (64 if M<=512 else 128))
    bn = 64 if M<=64 else 128
    bk = 128 if M<=64 else 64
    nw = 4 if M<=128 else 8
    ns = int(st) if (st != "kbest" and M<=32) else 3
    if st == "kbest" and M<=64:
        if M<=4:
            bm, bn, bk, nw, ns = 16, 32, 64, 4, 2
        else:
            bm, bn, bk, nw, ns = 16, 128, 64, 8, 2
    cfg[str(M)] = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
                   "GROUP_SIZE_M": 1, "num_warps": nw, "num_stages": ns}
name = f"E={E},N={N},device_name=Tesla_V100-SXM2-32GB.json"
json.dump(cfg, open(f"{d}/{name}", "w"), indent=1)
print(name)
PY
}

run_arm() {
    local arm="$1" cfgdir_rel="" envargs=() cname slog
    cname="moeab_${MODEL_KEY}_${arm}"
    slog="$OUT/${arm}_serve.log"
    if [[ "$arm" != base ]]; then
        cfgdir_rel="$OUT/cfg_${arm}"
        local st="${arm#s}"
        local fn; fn=$(write_cfg "$cfgdir_rel" "$st") || { echo "$arm: FAIL cfg gen" | tee -a "$SUMMARY"; return 1; }
        note "$arm: wrote $cfgdir_rel/$fn"
        envargs=(-e "VLLM_TUNED_CONFIG_FOLDER=/work/$cfgdir_rel")
    fi
    clean_box_guard || { echo "$arm: SKIP (GPUs $GPUS busy)" | tee -a "$SUMMARY"; return 1; }

    note "=== arm=$arm model=$(basename "$MODEL") TP=$TP ==="
    docker rm -f "$cname" >/dev/null 2>&1 || true
    docker run --rm -i --name "$cname" --gpus "\"device=$GPUS\"" \
        -v /mnt/models:/mnt/models:ro -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
        -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
        -p ${PORT}:${PORT} --shm-size=16g \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e VLLM_ATTENTION_BACKEND=TRITON_ATTN -e VLLM_V100_FP8_COALESCED_GEMV=0 \
        "${envargs[@]}" \
        "$IMAGE" \
        python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
            --tensor-parallel-size "$TP" --dtype float16 \
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
        sleep 10; waited=$((waited+10)); (( waited % 120 == 0 )) && note "  ...loading $arm (${waited}s)"
    done
    if [[ "$healthy" != 1 ]]; then
        echo "$arm: FAIL (never healthy) — $slog" | tee -a "$SUMMARY"
        grep -nE "Error|Traceback|no kernel image|out of memory|assert" "$slog" | head -6 | tee -a "$SUMMARY"
        docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; return 1
    fi

    # PROOF the intended config path ran (config pickup is per-worker, info_once):
    local picked
    if [[ "$arm" == base ]]; then
        picked=$(grep -c "Using default MoE config" "$slog" || true)
        echo "$arm: config-miss warnings in log = $picked (expect >=1)" | tee -a "$SUMMARY"
    else
        picked=$(grep -c "Using configuration from /work/$cfgdir_rel" "$slog" || true)
        echo "$arm: override pickup lines = $picked (expect >=1)" | tee -a "$SUMMARY"
        if [[ "$picked" -eq 0 ]]; then
            echo "$arm: FAIL — override NOT picked up, results would be meaningless" | tee -a "$SUMMARY"
            docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; return 1
        fi
    fi

    # warmup (JIT + cudagraph capture)
    curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":16,\"temperature\":0}" >/dev/null 2>&1 || true

    local i v
    for i in $(seq 1 "$NRUN"); do
        v=$(python3 - "$PORT" "$SERVED" "$GENTOK" "$OUT/${arm}_run${i}.txt" <<'PY'
import hashlib, json, sys, time, urllib.request
port, served, tok, sfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
prompt = "Write a detailed, multi-section essay on the history, geography, economy, and culture of France. Use clear subsections with headings and develop each at length."
body=json.dumps({"model":served,"stream":True,"max_tokens":tok,"temperature":0,"ignore_eos":True,
 "stream_options":{"include_usage":True},"messages":[{"role":"user","content":prompt}]}).encode()
req=urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body,
                           headers={"Content-Type":"application/json"})
t0=time.time(); tf=tl=None; n=0; ch=[]; ut=0
with urllib.request.urlopen(req, timeout=2400) as r:
    for raw in r:
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
s="".join(ch); open(sfile,"w").write(s)
mt=ut or n
dec=(mt-1)/(tl-tf) if tf and tl and tl>tf and mt>1 else float("nan")
ttft=(tf-t0) if tf else float("nan")
w=s.split()
rep=(max((w.count(x) for x in set(w)),default=0)/len(w)) if w else 1.0
sha=hashlib.sha256(s.encode()).hexdigest()[:16]
print(f"{mt}\t{dec:.2f}\t{ttft:.2f}\t{rep:.3f}\t{sha}")
PY
)
        echo "$arm run${i}: tok=$(cut -f1 <<<"$v") decode=$(cut -f2 <<<"$v") tok/s ttft=$(cut -f3 <<<"$v")s rep=$(cut -f4 <<<"$v") sha=$(cut -f5 <<<"$v")" | tee -a "$SUMMARY"
    done
    docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true
    sleep 5
}

echo "MoE num_stages A/B — $MODEL_KEY ($MODEL) E=$E N=$NSHARD TP=$TP ns=$NS gentok=$GENTOK — $(date -u +%FT%TZ)" | tee -a "$SUMMARY"
for arm in $ARMS; do
    run_arm "$arm"
    echo "" >> "$SUMMARY"
done

python3 - "$SUMMARY" <<'PY' | tee -a "$SUMMARY"
import re, statistics, sys
text=open(sys.argv[1]).read()
arms={}
for m in re.finditer(r"^(\w+) run\d+: tok=\d+ decode=([0-9.]+)", text, re.M):
    arms.setdefault(m.group(1), []).append(float(m.group(2)))
print("== MEANS ==")
for a, v in arms.items():
    print(f"{a}: decode_mean={statistics.mean(v):.2f} min={min(v):.2f} max={max(v):.2f} n={len(v)}")
PY
note "done: $SUMMARY"
