#!/usr/bin/env bash
# FA-V100 FLEET VERIFICATION — overnight batch (audit Turn 19+).
#
# Purpose (user directive): verify the FA path on ALL eligible local models BEFORE
# any upstream feedback to ai-bond — one consolidated report with full-fleet evidence.
# Per model: full triton-vs-fa A/B (reuses fa_prefill_ttft_ab_vllm021.sh) under the
# promoted config (cudagraph, full FP8 env set, chunked default, block 256).
# Evidence per model lands in /tmp/v100_fa_fleet/<tag>/ (copy durably after).
#
# Already verified elsewhere (skipped): GLM-4.5-Air-FP8 (headline 2.66x),
# Qwen3.5-122B-A10B-FP8 (eager + cudagraph). Excluded: GLM-4.7-Flash (MLA).
# Gemma-4: full-attn layers route, sliding layers fall back per-call (P3 pending)
# -> the run validates the MIXED route/fallback behavior + coherence.
#
# Shared-box etiquette: re-checks the clean-box guard BEFORE EACH MODEL; if busy,
# waits up to BUSY_WAIT_MAX then skips that model (recorded as SKIPPED-busy).
#
# Usage:  ./tools/fa_fleet_verify_vllm021.sh                # full fleet
#         ONLY_TAG=qwen36-27b ./tools/fa_fleet_verify_vllm021.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

OUT=/tmp/v100_fa_fleet
AB=/tmp/v100_fa_ttft_ab          # the A/B harness's own out dir (copied per model)
mkdir -p "$OUT"
FLEET_SUMMARY="$OUT/FLEET_SUMMARY.txt"
: > "$FLEET_SUMMARY"
BUSY_WAIT_MAX="${BUSY_WAIT_MAX:-3600}"
ONLY_TAG="${ONLY_TAG:-}"
note() { echo "[fleet] $*" | tee -a "$FLEET_SUMMARY"; }

# tag | model path | TP | maxlen | prompt_tokens | notes
FLEET=(
  "qwen25-7b|/mnt/models/Qwen/Qwen2.5-7B-Instruct|2|16384|12000|D=128 dense"
  "dsr1-qwen32b|/mnt/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|4|16384|12000|D=128 dense"
  "qwen3-30b-a3b|/mnt/models/Qwen/Qwen3-30B-A3B-Instruct-2507|4|16384|12000|D=128 MoE"
  "qwen3-coder-30b|/mnt/models/Qwen/Qwen3-Coder-30B-A3B-Instruct|4|16384|12000|D=128 MoE"
  "mixtral-8x7b|/mnt/models/Mixtral-8x7B-Instruct|4|16384|12000|D=128 MoE fp16"
  "qwen36-27b-fp8|/mnt/models/Qwen/Qwen3.6-27B-FP8|4|16384|12000|D=256 GDN-hybrid"
  "qwen36-35b-fp8|/mnt/models/Qwen/Qwen3.6-35B-A3B-FP8|4|16384|12000|D=256 GDN-hybrid MoE"
  "gemma4-31b-fp8|/mnt/models/RedHatAI/gemma-4-31B-it-FP8-Dynamic|4|16384|12000|D=256 full+sliding (sliding->fallback)"
  "q122b-int4|/mnt/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4|8|16384|12000|D=256 GDN-hybrid Int4"
)

wait_for_clean_box() {
    local waited=0 i used pids any
    while (( waited < BUSY_WAIT_MAX )); do
        any=0
        for i in 0 1 2 3 4 5 6 7; do
            used=$(nvidia-smi --id="$i" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
            pids=$(nvidia-smi --id="$i" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
            [[ "${used:-9999}" -gt 2000 || "${pids:-1}" -gt 0 ]] && any=1
        done
        [[ "$any" -eq 0 ]] && return 0
        sleep 60; waited=$((waited+60))
    done
    return 1
}

for entry in "${FLEET[@]}"; do
    IFS='|' read -r tag model tp maxlen ptok notes <<<"$entry"
    [[ -n "$ONLY_TAG" && "$ONLY_TAG" != "$tag" ]] && continue
    [[ -d "$model" ]] || { note "$tag: SKIPPED (model dir missing: $model)"; continue; }
    if ! wait_for_clean_box; then
        note "$tag: SKIPPED-busy (box not clean within ${BUSY_WAIT_MAX}s)"; continue
    fi
    note "=== $tag ($notes) TP=$tp maxlen=$maxlen prompt=$ptok — $(date -u +%H:%M:%SZ) ==="
    MODE=cudagraph MODEL="$model" TP="$tp" MAXLEN="$maxlen" PROMPT_TOKENS="$ptok" \
      TRIALS=2 MAXTOK=64 SKIP_MM=1 GPUMEM=0.92 \
      ./tools/fa_prefill_ttft_ab_vllm021.sh >"$OUT/${tag}_ab.log" 2>&1
    rc=$?
    mkdir -p "$OUT/$tag"
    cp "$AB"/SUMMARY.txt "$AB"/*_trials.txt "$OUT/$tag/" 2>/dev/null
    # keep serve-log forensics (banners + errors), not the full logs
    for arm in triton fa; do
        { grep -E "fa_v100_prefill|Setting attention block size|GPU KV cache size|Maximum concurrency" "$AB/${arm}_serve.log" 2>/dev/null | head -20
          grep -E "Error|Traceback" "$AB/${arm}_serve.log" 2>/dev/null | head -10
        } > "$OUT/$tag/${arm}_forensics.txt"
    done
    {
      echo "--- $tag (rc=$rc) ---"
      cat "$AB/SUMMARY.txt" 2>/dev/null
      echo "fa fallback reasons: $(grep -hoE 'fallback to Triton \([^)]*\)' "$AB/fa_serve.log" 2>/dev/null | sort | uniq -c | tr '\n' ';')"
    } >> "$FLEET_SUMMARY"
    note "$tag done (rc=$rc)"
done

note "FLEET COMPLETE — $(date -u +%H:%M:%SZ). Summary: $FLEET_SUMMARY"
cat "$FLEET_SUMMARY"