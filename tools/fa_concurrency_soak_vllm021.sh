#!/usr/bin/env bash
# FA-V100 concurrency soak — the last gate before docs/promotion (audit Turn 14+).
#
# Correctness is closed (Sq<Sk + mixed-batch gates pass); this measures SERVING
# BEHAVIOR UNDER LOAD: concurrent prefill+decode coherence, per-user throughput,
# block_size=256 fragmentation in practice, scheduler/preemption behavior.
#
# THREE ARMS decompose the two effects (each arm = fresh server):
#   triton16  : VLLM_V100_FLASH_ATTN=0, --block-size 16   (today's production config)
#   triton256 : VLLM_V100_FLASH_ATTN=0, --block-size 256  (block-size effect alone)
#   fa256     : VLLM_V100_FLASH_ATTN=1, --block-size 256  (FA effect on top)
#
# PER ARM:
#   phase S (sanity, Codex): single-user 24k prompt -> TTFT anchor against the
#           known baselines (triton ~51.8s, fa ~19.45s @24,040 tok).
#   phase C (soak): USERS concurrent clients; each sends ONE long prompt
#           (staggered sizes 6k/12k/18k/24k) then FOLLOWUPS short decode-heavy
#           requests. Every request logs ACTUAL prompt_tokens + completion_tokens
#           (from API usage, Codex), TTFT, decode tok/s, coherence.
#   capture: KV-pool + max-concurrency banner, route/fallback/densify banners,
#           preemption warnings.
#
# Usage:  ./tools/fa_concurrency_soak_vllm021.sh              # all 3 arms
#         ONLY=fa256 ./tools/fa_concurrency_soak_vllm021.sh   # one arm
# Env: IMAGE PORT HEALTH_TIMEOUT GPUMEM MAXLEN USERS FOLLOWUPS MODEL ONLY
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
FA_DIR="${FA_DIR:-/home/kumphanartd/flash-attention-v100}"
PORT="${PORT:-8024}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3600}"
GPUMEM="${GPUMEM:-0.90}"
MAXLEN="${MAXLEN:-28672}"
USERS="${USERS:-4}"
FOLLOWUPS="${FOLLOWUPS:-2}"
MODEL="${MODEL:-/mnt/models/zai-org/GLM-4.5-Air-FP8}"
TP=8
ONLY="${ONLY:-}"
CACHE_TAG="${CACHE_TAG:-021}"
SERVED="fasoak"

OUT=/tmp/v100_fa_soak
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
: > "$SUMMARY"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
note() { echo "[fa-soak] $*"; }

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
    local label="$1" faflag="$2" blocksz="$3" cname slog
    cname="fasoak_${label}"; slog="$OUT/${label}_serve.log"
    : > "$OUT/${label}_requests.tsv"
    note "=== arm=$label FA=$faflag block=$blocksz (TP=$TP users=$USERS followups=$FOLLOWUPS) ==="
    docker rm -f "$cname" >/dev/null 2>&1 || true
    docker run --rm -i --name "$cname" --gpus '"device=0,1,2,3,4,5,6,7"' \
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
        -e VLLM_V100_CT_MOE_W2_GROUPED=1 \
        -e VLLM_V100_CT_MOE_PREFILL_TILED=1 -e VLLM_V100_CT_MOE_PREFILL_FUSED=1 \
        -e VLLM_V100_CT_CHANNEL_WMMA=1 \
        -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        "$IMAGE" \
        python3 -m fp8_w8a16_sm70.vllm_serve --model "$MODEL" --served-model-name "$SERVED" \
            --tensor-parallel-size "$TP" --dtype float16 \
            --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --max-model-len "$MAXLEN" --max-num-seqs 8 --block-size "$blocksz" \
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
        grep -nE "Error|Traceback|no kernel image|out of memory" "$slog" | head -6
        docker stop "$cname" >/dev/null 2>&1 || true; wait "$lpid" 2>/dev/null || true; return 1
    fi
    note "  healthy after ${waited}s"

    # KV pool / concurrency banners (block-size effect shows here)
    local kvline
    kvline=$(grep -m1 -E "GPU KV cache size|Maximum concurrency" "$slog" | tr -d '\n' || true)
    grep -m2 -E "GPU KV cache size|Maximum concurrency" "$slog" | sed 's/^/    /' || true

    # warmup
    curl -s "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":16,\"temperature\":0}" >/dev/null 2>&1 || true

    # phases S + C — concurrent python client, per-request TSV with real token counts
    python3 - "$PORT" "$SERVED" "$USERS" "$FOLLOWUPS" "$OUT/${label}_requests.tsv" <<'PY'
import json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

port, served, users, followups, tsv = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])

def make_prompt(ptok, seed):
    words = int(ptok / 1.31)
    para = (f"The {seed} expedition crossed the silent valley while measuring "
            "glacier retreat, soil moisture, and the migration of mountain birds. ")
    body = " ".join((para * (words // len(para.split()) + 1)).split()[:words])
    return body + "\n\nIn one short sentence, what was being measured?"

def request(tag, prompt, maxtok, check):
    req = {"model": served, "messages": [{"role": "user", "content": prompt}],
           "max_tokens": maxtok, "temperature": 0, "stream": True,
           "stream_options": {"include_usage": True}}
    t0 = time.time(); tf = None; chunks = []; usage = None
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions",
            data=json.dumps(req).encode(),
            headers={"Content-Type": "application/json"}), timeout=1800)
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            p = line[5:].strip()
            if p == "[DONE]": break
            try: j = json.loads(p)
            except Exception: continue
            if j.get("usage"): usage = j["usage"]
            ch = j.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if tf is None: tf = time.time()
                chunks.append(ch[0]["delta"]["content"])
    except Exception as e:
        return f"{tag}\tERROR\t-\t-\t-\t-\t{type(e).__name__}: {e}"
    tend = time.time(); text = "".join(chunks)
    ttft = (tf - t0) if tf else float("nan")
    n = len(chunks)
    dtps = (n - 1) / (tend - tf) if tf and n > 1 else float("nan")
    pt = (usage or {}).get("prompt_tokens", "?")
    ct = (usage or {}).get("completion_tokens", "?")
    ok = "OK" if (check(text) and len(text) > 5) else "SUSPECT"
    snip = re.sub(r"\s+", " ", text)[:80]
    return (f"{tag}\t{ok}\tpt={pt}\tct={ct}\tttft={ttft:.2f}s\t"
            f"decode={dtps:.2f}t/s\t{snip}")

long_check = lambda t: bool(re.search(r"glacier|moisture|bird|measur", t, re.I))
short_check = lambda t: ("1" in t and "5" in t)
lines = []

# ── phase S: single-user 24k sanity anchor ──────────────────────────────────
lines.append(request("S\tu0\tlong24k", make_prompt(24000, "anchor"), 64, long_check))
print(lines[-1], flush=True)

# ── phase C: concurrent mixed soak ──────────────────────────────────────────
sizes = [6000, 12000, 18000, 24000]
def user_session(u):
    out = []
    seed = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"][u % 6]
    out.append(request(f"C\tu{u}\tlong{sizes[u % len(sizes)]//1000}k",
                       make_prompt(sizes[u % len(sizes)], seed), 64, long_check))
    for f in range(followups):
        out.append(request(f"C\tu{u}\tshort{f}",
                           "List the numbers one to five as digits.", 128, short_check))
    return out

with ThreadPoolExecutor(max_workers=users) as ex:
    for fut in [ex.submit(user_session, u) for u in range(users)]:
        for ln in fut.result():
            lines.append(ln); print(ln, flush=True)

with open(tsv, "w") as f:
    f.write("\n".join(lines) + "\n")
ok = sum("\tOK\t" in l for l in lines); tot = len(lines)
print(f"AGG\t{ok}/{tot} OK", flush=True)
PY

    # serve-log forensics
    local routed densify fellback preempt
    routed=$(grep -c "prefill -> flash_attn_v100" "$slog" 2>/dev/null || true)
    densify=$(grep -c "densify q stride" "$slog" 2>/dev/null || true)
    fellback=$(grep -c "fallback to Triton" "$slog" 2>/dev/null || true)
    preempt=$(grep -ciE "preempt" "$slog" 2>/dev/null || true)
    {
      echo "$label: kv-banner: ${kvline:-<none>}"
      echo "$label: route=$routed densify=$densify fallback=$fellback preempt=$preempt"
      echo "$label: $(grep -c $'\tOK\t' "$OUT/${label}_requests.tsv" || true)/$(wc -l < "$OUT/${label}_requests.tsv") requests OK — details $OUT/${label}_requests.tsv"
    } | tee -a "$SUMMARY"

    docker stop "$cname" >/dev/null 2>&1 || true
    wait "$lpid" 2>/dev/null || true
    sleep 5
}

note "results -> $OUT (SUMMARY.txt)"
clean_box_guard || { note "ABORT: box not clean (need all 8 GPUs)"; exit 3; }
[[ -f "$FA_DIR/build/lib.linux-x86_64-cpython-312/flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so" ]] \
  || { note "ABORT: ai-bond cu126 build missing in $FA_DIR/build"; exit 4; }
docker run --rm -v "$FA_DIR":/fasrc:ro -v "$OUT":/out alpine sh -c \
  "mkdir -p /out/pylib && cp /fasrc/build/lib.linux-x86_64-cpython-312/flash_attn_v100_cuda.*.so /out/pylib/ && chown -R $(id -u):$(id -g) /out/pylib" \
  || { note "ABORT: failed to stage flash_attn_v100_cuda.so"; exit 4; }

# arm-label | FA flag | block size
for arm in "triton16|0|16" "triton256|0|256" "fa256|1|256"; do
    IFS='|' read -r label faflag blocksz <<<"$arm"
    [[ -n "$ONLY" && "$ONLY" != "$label" ]] && continue
    run_arm "$label" "$faflag" "$blocksz"
done
note "=== SUMMARY ==="
cat "$SUMMARY"