#!/usr/bin/env bash
# ======================================================================================
# ai-bond flash-attention-v100 : clean-box BUILD + go/no-go MICROBENCH driver (Route A).
#
# Self-contained: the user launches THIS on the shared V100 box. It guards the box,
# captures full env/git state, builds ai-bond, runs the correctness gate (test.py),
# the adapter-contract smoke (fa_v100_paged_smoke.py), and the prefill microbench
# (fa_v100_microbench.py). All output is tee'd to a timestamped results file.
#
# Route A defaults: block_size=256, fp16 KV, no FP8-KV, no split-KV, no cascade.
#
# Env knobs:
#   AIBOND_DIR   (default /home/kumphanartd/flash-attention-v100)
#   VLLM_DIR     (default /home/kumphanartd/vllm-0.21)  -- for git-hash capture only
#   FORCE=1      run even if other GPU compute processes are present (son's training)
#   SKIP_BUILD=1 skip the pip build (already installed)
#   AUTOPATCH=0  disable the BLOCK_N_128 contingency on smoke failure (default: ON)
#   PREPATCH=1   apply BLOCK_N_128=128 BEFORE the first build (skip the stock attempt)
#   SEQLEN HQ HK D LAYERS  -- microbench overrides
#
# Known-likely issue (audit Turn 7, Claude+Codex static-verified): at D=128 the KV tile
# BLOCK_N_128=160 does NOT divide the 256-token page; tile 1 (rows 160..319) straddles
# the page boundary and the single-pointer linear load reads the WRONG physical block
# for non-contiguous block tables. The smoke (D=128, interleaved block_table, garbage
# tails) triggers exactly this. Contingency = BLOCK_N_128 160->128 in include/forward.h.
# This driver auto-applies it on stock-smoke failure, clean-rebuilds, re-gates, benches —
# one GPU window yields: bug confirmed + fix validated + numbers.
# ======================================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIBOND_DIR="${AIBOND_DIR:-/home/kumphanartd/flash-attention-v100}"
VLLM_DIR="${VLLM_DIR:-/home/kumphanartd/vllm-0.21}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${HERE}/fa_v100_microbench_${TS}.txt"

SEQLEN="${SEQLEN:-26000}"; HQ="${HQ:-12}"; HK="${HK:-1}"; D="${D:-128}"; LAYERS="${LAYERS:-46}"

log() { echo "$@" | tee -a "$OUT"; }
hr()  { log "----------------------------------------------------------------------"; }

_bn128() { grep -oE '#define BLOCK_N_128 +[0-9]+' "$AIBOND_DIR/include/forward.h" 2>/dev/null | grep -oE '[0-9]+$' || echo '?'; }
BUILD_VARIANT="source(BLOCK_N_128=$(_bn128))"

build_aibond() {
  # --no-deps: torch/numpy already in the image; ai-bond's dep list tries to
  # reinstall debian-managed wheel/setuptools and fails (RECORD file not found)
  ( cd "$AIBOND_DIR" && pip install . --no-build-isolation --no-deps -v ) 2>&1 | tee -a "$OUT" | tail -8
  local rc=${PIPESTATUS[0]}
  if [[ "$rc" != "0" ]]; then
    log "    pip build FAILED (exit $rc) — a stale installed flash_attn_v100 could otherwise mask this."
    return 1
  fi
  if ! python3 -c "import flash_attn_v100" 2>/dev/null; then
    log "    flash_attn_v100 import failed after build."
    return 1
  fi
  log "    build OK ($BUILD_VARIANT): $(python3 -c 'import flash_attn_v100 as f; print(getattr(f,"__doc__",""))' 2>/dev/null)"
  return 0
}

run_testpy() {
  if [[ -f "$AIBOND_DIR/test.py" ]]; then
    ( cd "$AIBOND_DIR" && python3 test.py ) 2>&1 | tee -a "$OUT" | tail -25
    local rc=${PIPESTATUS[0]}
    log "    [test.py exit: $rc]"
    [[ "$rc" == "0" ]] || return 1
  else
    log "    WARN: test.py not found in $AIBOND_DIR — correctness gate SKIPPED."
  fi
  return 0
}

apply_blockn128_patch() {
  local F="$AIBOND_DIR/include/forward.h"
  if ! grep -q '#define BLOCK_N_128 160' "$F"; then
    log "    patch target '#define BLOCK_N_128 160' not found (already patched or source changed)."
    return 1
  fi
  cp "$F" "$F.bak_fa_audit"
  sed -i 's/#define BLOCK_N_128 160/#define BLOCK_N_128 128/' "$F"
  rm -rf "$AIBOND_DIR/build"   # headers are not dependency-tracked by setuptools; force full rebuild
  BUILD_VARIANT="patched(BLOCK_N_128=128)"
  log "    CONTINGENCY APPLIED: BLOCK_N_128 160->128 (tile now divides 256 page)."
  log "    (backup: include/forward.h.bak_fa_audit ; revert via git checkout -- include/forward.h)"
  return 0
}

log "== ai-bond FA-V100 build+microbench :: ${TS} =="
log "results -> $OUT"
hr

# --- 1. clean-box guard -------------------------------------------------------------
log "[1] CLEAN-BOX GUARD"
PROCS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
if [[ -n "${PROCS//[$' \t\r\n']/}" ]]; then
  log "    other GPU compute processes detected:"
  echo "$PROCS" | sed 's/^/      /' | tee -a "$OUT"
  if [[ "${FORCE:-0}" != "1" ]]; then
    log "    ABORT: box not clean. Re-run with FORCE=1 to override (will contend w/ training)."
    exit 3
  fi
  log "    FORCE=1 set -> continuing despite contention."
else
  log "    GPU clean (no other compute processes)."
fi
hr

# --- 2. toolchain pin-check ---------------------------------------------------------
log "[2] TOOLCHAIN PIN-CHECK"
NVCC_VER="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
log "    nvcc release: ${NVCC_VER:-<none>} (need <= 12.9 for compute_70)"
if [[ -n "$NVCC_VER" ]]; then
  awk "BEGIN{exit !($NVCC_VER > 12.9)}" && log "    WARN: CUDA > 12.9 — compute_70 dropped in CUDA 13; build will likely fail."
fi
python3 - <<'PY' 2>&1 | tee -a "$OUT"
import torch
print(f"    torch: {torch.__version__}  cuda_avail={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"    device: {p.name}  SM {p.major}.{p.minor}")
    assert p.major == 7 and p.minor == 0, "NOT a Volta SM70 device — ai-bond is sm_70 only"
else:
    raise SystemExit("    ABORT: no CUDA device visible")
PY
[[ ${PIPESTATUS[0]:-1} -ne 0 ]] && { log "    ABORT: torch/device check failed."; exit 4; }
hr

# --- 3. env + git state capture (Codex) ---------------------------------------------
log "[3] ENV + GIT STATE"
log "    --- nvidia-smi ---"
nvidia-smi 2>&1 | sed 's/^/    /' | tee -a "$OUT" >/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    GPU: /' | tee -a "$OUT"
log "    nvcc: $(nvcc --version 2>/dev/null | tail -1)"
log "    python: $(python3 --version 2>&1)"
ab_hash="$(git -C "$AIBOND_DIR" rev-parse --short HEAD 2>/dev/null || echo '<n/a>')"
vl_hash="$(git -C "$VLLM_DIR" rev-parse --short HEAD 2>/dev/null || echo '<n/a>')"
log "    ai-bond ($AIBOND_DIR) HEAD: $ab_hash"
log "    vllm    ($VLLM_DIR) HEAD: $vl_hash"
log "    bench shapes: SEQLEN=$SEQLEN HQ=$HQ HK=$HK D=$D LAYERS=$LAYERS block=256 (Route A)"
hr

# --- 4. build ai-bond ----------------------------------------------------------------
log "[4] BUILD ai-bond"
if [[ "${PREPATCH:-0}" == "1" ]]; then
  log "    PREPATCH=1 -> applying BLOCK_N_128=128 contingency BEFORE first build."
  apply_blockn128_patch || { log "    ABORT: prepatch failed."; exit 5; }
fi
if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  log "    SKIP_BUILD=1 -> skipping pip build."
else
  build_aibond || { log "    ABORT: build failed."; exit 5; }
fi
hr

# --- 5. correctness gate : ai-bond test.py ------------------------------------------
log "[5] CORRECTNESS GATE (ai-bond test.py) — HARD GATE"
run_testpy || { log "    ABORT: ai-bond test.py FAILED — do NOT proceed to smoke/microbench."; exit 7; }
hr

# --- 6. adapter-contract smoke ------------------------------------------------------
log "[6] ADAPTER-CONTRACT SMOKE (paged + block_table + seqused_k + synth cu_seqlens_k + in-place out)"
python3 "$HERE/fa_v100_paged_smoke.py" 2>&1 | tee -a "$OUT"
SMOKE_RC=${PIPESTATUS[0]}
log "    [smoke exit: $SMOKE_RC] (build: $BUILD_VARIANT)"
if [[ "$SMOKE_RC" != "0" ]]; then
  if [[ "$SMOKE_RC" == "1" && "${AUTOPATCH:-1}" == "1" && "$(_bn128)" == "160" ]]; then
    log "    smoke FAILED on stock build — consistent with the PREDICTED D=128 tile/page"
    log "    straddle (BLOCK_N_128=160 does not divide page 256; audit Turn 7)."
    log "    Applying contingency + clean rebuild + full re-gate..."
    apply_blockn128_patch || { log "    ABORT: contingency patch failed."; exit 6; }
    build_aibond || { log "    ABORT: patched build failed."; exit 5; }
    run_testpy   || { log "    ABORT: test.py FAILED on patched build."; exit 7; }
    python3 "$HERE/fa_v100_paged_smoke.py" 2>&1 | tee -a "$OUT"
    SMOKE_RC=${PIPESTATUS[0]}
    log "    [patched smoke exit: $SMOKE_RC]"
    if [[ "$SMOKE_RC" != "0" ]]; then
      log "    ABORT: smoke STILL failing with BLOCK_N_128=128 — not (only) the straddle bug."
      exit 6
    fi
    log "    >>> STRADDLE BUG CONFIRMED + FIX VALIDATED: stock FAIL -> patched PASS. <<<"
  else
    log "    STOP: adapter contract not satisfied — do NOT proceed to microbench/integration."
    exit 6
  fi
fi
hr

# --- 7. go/no-go microbench ---------------------------------------------------------
log "[6b] LONG-SEQ + LAYOUT GATE (paged 512->24k, interleaved-KV views, qkv-split q)"
python3 "$HERE/fa_v100_longseq_check.py" 2>&1 | tee -a "$OUT" | grep -E "LONGSEQ|Traceback"
LONG_RC=${PIPESTATUS[0]}
log "    [longseq exit: $LONG_RC]"
if [[ "$LONG_RC" != "0" ]]; then
  log "    ABORT: long-seq/layout correctness gate FAILED — the e2e-discovered failure"
  log "    modes (24k garbage, qkv-split strided q) are guarded here now."
  exit 9
fi
hr

log "[7] PREFILL MICROBENCH (Route A, build: $BUILD_VARIANT)"
python3 "$HERE/fa_v100_microbench.py" \
  --seqlen "$SEQLEN" --hq "$HQ" --hk "$HK" --d "$D" --layers "$LAYERS" 2>&1 | tee -a "$OUT"
BENCH_RC=${PIPESTATUS[0]}
if [[ "$BENCH_RC" != "0" ]]; then
  log "    ABORT: microbench FAILED (exit $BENCH_RC) — no go/no-go numbers produced."
  exit 8
fi
hr
log "DONE. Full log: $OUT"
log "Decision: correctness PASS (5+6) AND microbench faster than ~42s residual => integrate adapter."
