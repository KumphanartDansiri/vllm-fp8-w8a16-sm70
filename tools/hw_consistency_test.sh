#!/usr/bin/env bash
# HARDWARE-CONSISTENCY TEST — run the SAME model on DIFFERENT physical GPU groups CONCURRENTLY and
# compare outputs. Distinguishes benign FP/all-reduce nondeterminism from a FAILING GPU (silent data
# corruption) on the aging V100s. Idea (user, 2026-06-10): the 8-card box = independent test rigs;
# cross-group agreement localizes a bad card.
#
# PROBE = a DETERMINISTIC path (dense FP8 = g31b-fp8, proven Exact 5/5) so any cross-group difference
# is PURE HARDWARE, not the FP16 warmup drift. temp=0, fixed prompt, ignore_eos.
#
# TEST A (TP4 cross-group): group A=GPU0-3, group B=GPU4-7, same model, CONCURRENT, ROUNDS rounds.
# TEST B (TP2 four pairs):  {0,1}{2,3}{4,5}{6,7}, same model, CONCURRENT, ROUNDS rounds (covers all 8).
#
# READ:
#   - every group every round same sha            -> HARDWARE CONSISTENT (no bad card). ✓
#   - one group drifts/garbage, others exact       -> that group has a SUSPECT GPU. <-- fault
#   - each group internally exact but groups differ -> likely NVLink topology (benign); confirm w/ dcgmi diag
# Compare only within a TP size (TP4 vs TP2 legitimately differ).
#
# Usage:  ./tools/hw_consistency_test.sh                 # both tests, g31b-fp8, 3 rounds
#         TEST=A ./tools/hw_consistency_test.sh          # just the TP4 cross-group test
#         MODEL=/mnt/models/google/gemma-4-31B-it PREC=fp16 ./tools/hw_consistency_test.sh   # FP16 stress variant
# Env: IMAGE MODEL PREC ROUNDS GENTOK TEST HEALTH_TIMEOUT
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
CACHE_TAG="${CACHE_TAG:-021cu126}"
MODEL="${MODEL:-/mnt/models/RedHatAI/gemma-4-31B-it-FP8-Dynamic}"
PREC="${PREC:-fp8}"
ROUNDS="${ROUNDS:-3}"
GENTOK="${GENTOK:-512}"           # short — we need determinism, not depth
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"
TEST="${TEST:-AB}"
ENTRY="${ENTRY:-our}"             # our = fp8_w8a16_sm70 plugin loaded; stock = pure upstream vllm (no plugin)
GPUMEM="${GPUMEM:-0.90}"
OUT="${OUT:-/tmp/v100_hw_consistency}"; mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"; : > "$SUMMARY"
PROMPT="Write a detailed multi-section essay on the history, geography, economy, and culture of France, with clear subsections."
note(){ echo "[hwtest] $*"; }
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done

fp8_env(){ [[ "$PREC" == "fp8" ]] && echo "-e VLLM_V100_FP8_COALESCED_GEMV=1 -e VLLM_V100_FP8_COALESCED_UNROLL=4 -e VLLM_V100_FP8_COALESCED_M_UNROLL=4 -e VLLM_V100_FP8_COALESCED_GEMV_M_MAX=8 -e VLLM_V100_FP8_MOE_W13_COALESCED=1 -e VLLM_V100_FP8_MOE_FALLBACK=1 -e VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM=1 -e VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS=128 -e VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=1 -e VLLM_V100_CT_FP8_RESIDENT=1 -e VLLM_V100_CT_MOE_W13_RESIDENT=1 -e VLLM_V100_CT_MOE_W13_FREE_FP16=1 -e VLLM_V100_CT_MOE_W2_GROUPED=1 -e VLLM_V100_CT_MOE_W13_COALESCED=1" || echo "-e VLLM_V100_FP8_COALESCED_GEMV=0"; }

serve(){ # label gpus port tp
    local label="$1" gpus="$2" port="$3" tp="$4" cname="hwt_${label}"
    docker rm -f "$cname" >/dev/null 2>&1 || true
    if [[ "$ENTRY" == "stock" ]]; then
        # PURE STOCK vLLM — our fp8_w8a16_sm70 module is NOT loaded at all (no PYTHONPATH, no patches,
        # no kernel). Isolates the hardware from OUR plugin: an FP16 model here exercises only upstream
        # vLLM, so cross-group consistency = a plugin-independent hardware verdict.
        # shellcheck disable=SC2086
        docker run --rm -i --name "$cname" --gpus "\"device=$gpus\"" \
            -v /mnt/models:/mnt/models:ro -p ${port}:${port} --shm-size=16g \
            -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
            -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
            -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
            "$IMAGE" vllm serve "$MODEL" --served-model-name hwt \
                --tensor-parallel-size "$tp" --dtype float16 --attention-backend TRITON_ATTN \
                --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
                --max-model-len 4096 --max-num-seqs 8 --skip-mm-profiling \
                --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill --host 0.0.0.0 --port "$port" \
            </dev/null >"$OUT/${label}_serve.log" 2>&1 &
    else
        # OUR plugin (fp8_w8a16_sm70) — the FP8 path runs our W8A16 kernel.
        # shellcheck disable=SC2086
        docker run --rm -i --name "$cname" --gpus "\"device=$gpus\"" \
            -v /mnt/models:/mnt/models:ro -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
            -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
            -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
            -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
            -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-inductor:/tmp/torchinductor_root" \
            -p ${port}:${port} --shm-size=16g \
            -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
            -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ATTENTION_BACKEND=TRITON_ATTN $(fp8_env) \
            "$IMAGE" python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name hwt \
                --tensor-parallel-size "$tp" --dtype float16 \
                --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
                --max-model-len 4096 --max-num-seqs 8 --skip-mm-profiling \
                --gpu-memory-utilization "$GPUMEM" --no-enable-chunked-prefill --host 0.0.0.0 --port "$port" \
            </dev/null >"$OUT/${label}_serve.log" 2>&1 &
    fi
    echo $!
}
wait_healthy(){ # port pid label
    local port="$1" pid="$2" label="$3" w=0
    while (( w < HEALTH_TIMEOUT )); do
        curl -sf "http://localhost:${port}/health" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || { note "  $label exited before healthy"; return 1; }
        sleep 10; w=$((w+10))
    done; return 1
}
gen_sha(){ # port -> prints sha256-16 of the generated text (temp=0, fixed prompt)
    python3 - "$1" "$GENTOK" "$PROMPT" <<'PY'
import sys,json,urllib.request,hashlib
port,tok,prompt=sys.argv[1],int(sys.argv[2]),sys.argv[3]
body=json.dumps({"model":"hwt","max_tokens":tok,"temperature":0,"ignore_eos":True,
  "messages":[{"role":"user","content":prompt}]}).encode()
req=urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
try:
    d=json.load(urllib.request.urlopen(req,timeout=600)); s=d["choices"][0]["message"]["content"]
    print(hashlib.sha256(s.encode()).hexdigest()[:16] + ("\tCOHERENT" if len(s.split())>20 else "\tSHORT/BAD"))
except Exception as e: print("ERR\t"+str(e)[:60])
PY
}

run_group_test(){ # testname; args: label:gpus:port:tp ...
    local testname="$1"; shift
    local specs=("$@") pids=() labels=() ports=()
    note "=== $testname : launching ${#specs[@]} concurrent groups ($MODEL $PREC) ==="
    local sp label gpus port tp pid
    for sp in "${specs[@]}"; do IFS=: read -r label gpus port tp <<<"$sp"
        pid=$(serve "$label" "$gpus" "$port" "$tp"); pids+=("$pid"); labels+=("$label"); ports+=("$port")
        note "  launched $label on GPU[$gpus] port $port (pid $pid)"
    done
    # wait all healthy
    local i ok=1
    for i in "${!labels[@]}"; do
        if wait_healthy "${ports[$i]}" "${pids[$i]}" "${labels[$i]}"; then note "  ${labels[$i]} healthy"; else note "  ${labels[$i]} FAILED to start"; ok=0; fi
    done
    if [[ "$ok" == 1 ]]; then
        # warmup each
        for i in "${!ports[@]}"; do gen_sha "${ports[$i]}" >/dev/null 2>&1 || true; done
        echo "## $testname ($MODEL $PREC, $ROUNDS rounds, $GENTOK tok)" | tee -a "$SUMMARY"
        declare -A allsha
        local r
        for r in $(seq 1 "$ROUNDS"); do
            for i in "${!ports[@]}"; do
                local res sha coh; res=$(gen_sha "${ports[$i]}"); sha=$(printf '%s' "$res"|cut -f1); coh=$(printf '%s' "$res"|cut -f2)
                echo "  round$r ${labels[$i]} (GPU group): sha=$sha $coh" | tee -a "$SUMMARY"
                allsha["${labels[$i]}"]+="$sha "
            done
        done
        # verdict: across ALL groups+rounds, how many distinct sha?
        local everything; everything=$(for i in "${!labels[@]}"; do echo "${allsha[${labels[$i]}]}"; done)
        local distinct; distinct=$(echo $everything | tr ' ' '\n' | grep -v '^$' | sort -u | wc -l)
        echo "  -> distinct sha across all groups×rounds: $distinct" | tee -a "$SUMMARY"
        if [[ "$distinct" == 1 ]]; then
            echo "  VERDICT: ✅ ALL groups×rounds IDENTICAL -> hardware CONSISTENT, no bad card." | tee -a "$SUMMARY"
        else
            echo "  VERDICT: ⚠️ $distinct distinct outputs. Check per-group: a group that ALONE differs/garbages = suspect GPU;" | tee -a "$SUMMARY"
            echo "           groups each-internally-consistent-but-differ = likely NVLink topology (run dcgmi diag to confirm)." | tee -a "$SUMMARY"
            for i in "${!labels[@]}"; do
                local gd; gd=$(echo ${allsha[${labels[$i]}]} | tr ' ' '\n' | grep -v '^$' | sort -u | wc -l)
                echo "           ${labels[$i]}: $gd distinct across rounds ($([[ $gd == 1 ]] && echo deterministic || echo DRIFTS))" | tee -a "$SUMMARY"
            done
        fi
    fi
    for label in "${labels[@]}"; do docker stop "hwt_${label}" >/dev/null 2>&1 || true; done
    echo "" | tee -a "$SUMMARY"
}

clean_box(){ local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader|awk '{s+=$1}END{print s+0}'); [[ "$u" -lt 2000 ]]; }
main(){
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { note "image missing"; exit 1; }
    clean_box || { note "GPUs busy — need an idle box"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 1; }
    [[ "$TEST" == *A* ]] && run_group_test "TEST A: TP4 cross-group (0-3 vs 4-7)" "A_gpu0123:0,1,2,3:8031:4" "B_gpu4567:4,5,6,7:8032:4"
    [[ "$TEST" == *B* ]] && run_group_test "TEST B: TP2 four pairs (all 8 GPUs)" "P01:0,1:8041:2" "P23:2,3:8042:2" "P45:4,5:8043:2" "P67:6,7:8044:2"
    note "==== HW-CONSISTENCY SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
}
main "$@"
