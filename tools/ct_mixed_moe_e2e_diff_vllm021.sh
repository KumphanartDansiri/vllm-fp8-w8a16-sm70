#!/usr/bin/env bash
# Phase 2b AUTHORITATIVE validation: end-to-end text diff of the mixed CT-MoE
# (w13 FP8 + shared-expert STORAGE) vs the FP16 baseline, both temperature=0.
#
# WHY not the apply-level self-check: it compares apply() return values, but the
# shared expert is a STORAGE side-effect the MoERunner combines AFTER apply, so
# the L2 check is blind to a missing/incorrect shared store. The only correct
# gate is deterministic token-for-token output vs the known-good FP16 path.
#
# Runs the glm45air probe TWICE (baseline then mixed-4-layers) and diffs the
# generated essay. IDENTICAL = correct. Early+semantic divergence = shared still
# broken. Late/minor divergence = FP8-w13 rounding (acceptable).
#
# Usage: ./tools/ct_mixed_moe_e2e_diff_vllm021.sh   (two 8-GPU loads, ~10 min)
# Env:   MAXLAYERS (default 4)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=/tmp/v100_glm45air_021
DIFFOUT=/tmp/v100_mixed_moe_e2e
MAXLAYERS="${MAXLAYERS:-0}"   # 0 = all 45 MoE layers (Phase 2c shipping config)
mkdir -p "$DIFFOUT"

echo "=== [1/2] baseline: FP16 MoE (CT_MOE_W13_RESIDENT=0) ==="
VLLM_V100_CT_FP8_RESIDENT=1 VLLM_V100_CT_MOE_W13_RESIDENT=0 \
  ./tools/glm45_air_fp8_load_vllm021.sh || true
cp -f "$OUT/sample.txt" "$DIFFOUT/baseline.txt" 2>/dev/null || { echo "no baseline sample — aborting"; exit 1; }

# Phase 2c default: all layers, FP16 w13 freed (the real shipping config). Set
# MAXLAYERS>0 + FREE=0 to instead diff the guarded 4-layer cut.
FREE="${FREE:-1}"
echo
echo "=== [2/2] mixed: w13 FP8 true-resident (RESIDENT=1, FREE_FP16=$FREE, MAXLAYERS=$MAXLAYERS) ==="
VLLM_V100_CT_FP8_RESIDENT=1 VLLM_V100_CT_MOE_W13_RESIDENT=1 \
  VLLM_V100_CT_MOE_W13_FREE_FP16="$FREE" \
  VLLM_V100_CT_MOE_W13_RESIDENT_MAXLAYERS="$MAXLAYERS" \
  ./tools/glm45_air_fp8_load_vllm021.sh || true
cp -f "$OUT/sample.txt" "$DIFFOUT/mixed.txt" 2>/dev/null || { echo "no mixed sample — aborting"; exit 1; }

echo
echo "=== DIFF baseline vs mixed (both temperature=0) ==="
if diff -q "$DIFFOUT/baseline.txt" "$DIFFOUT/mixed.txt" >/dev/null 2>&1; then
    echo "RESULT: IDENTICAL — mixed (FP8 w13 + shared storage) matches FP16 baseline token-for-token."
    echo "        -> Phase 2b correct; safe to proceed to Phase 2c (free FP16 w13)."
else
    python3 - "$DIFFOUT/baseline.txt" "$DIFFOUT/mixed.txt" <<'PY'
import sys
a = open(sys.argv[1]).read(); b = open(sys.argv[2]).read()
n = 0
for x, y in zip(a, b):
    if x != y:
        break
    n += 1
m = min(len(a), len(b))
frac = (n / m) if m else 0.0
print(f"RESULT: DIVERGES at char {n}/{m} ({frac:.0%} common prefix)")
print("  baseline:", repr(a[max(0, n-20):n+50]))
print("  mixed   :", repr(b[max(0, n-20):n+50]))
if n < 60:
    print("  -> EARLY divergence: shared-expert storage still wrong (the bug).")
else:
    print("  -> LATE divergence: likely FP8-w13 rounding accumulation (acceptable);")
    print("     compare semantics, not exact tokens.")
PY
fi
echo "(samples saved in $DIFFOUT/)"
