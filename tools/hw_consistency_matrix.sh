#!/usr/bin/env bash
# Hardware × plugin disambiguation matrix (cross-group TP4, 0-3 vs 4-7, 3 rounds each):
#   DETERMINISTIC model (q35b, Exact in Ch1) — expect BIT-IDENTICAL cross-group:
#     - q35b FP16 STOCK  : pure upstream vLLM, our plugin NOT loaded -> identical => HARDWARE clean, plugin-independent
#     - q35b FP8  OUR    : our W8A16 kernel in the path           -> identical => our kernel is deterministic too
#   DRIFTING model (g26b MoE, drifts in Ch1) — expect COHERENT drift; the question is WHY it drifts:
#     - g26b FP16 STOCK  : if stock MoE ALSO drifts -> the drift is ROUTING nondeterminism, NOT our plugin
#     - g26b FP8  OUR    : the "grey" one; if it drifts like stock + stays coherent on both halves -> benign routing,
#                          not our edge case. (Garbage/crash on a group would indicate a real kernel problem.)
# READ TOGETHER: deterministic pair identical => hardware ruled out (both stock & our paths). Drifting pair both
# coherent & stock-drifts-too => g26b-fp8 drift = MoE routing, our kernel exonerated.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run(){ # title model prec entry outdir
    echo; echo "############## $1 ##############"
    OUT="$5" MODEL="$2" PREC="$3" ENTRY="$4" TEST=A ROUNDS=3 GENTOK=512 HEALTH_TIMEOUT=1500 \
        ./tools/hw_consistency_test.sh || true
    sleep 15   # let GPU memory release before the next concurrent-serve pair
}
run "q35b FP16 — STOCK vLLM (no plugin) — deterministic"        /mnt/models/Qwen/Qwen3.6-35B-A3B          fp16 stock /tmp/hwm_q35b_fp16
run "q35b FP8  — OUR plugin — deterministic"                    /mnt/models/Qwen/Qwen3.6-35B-A3B-FP8      fp8  our   /tmp/hwm_q35b_fp8
run "g26b FP16 — STOCK vLLM (no plugin) — drifting (MoE)"       /mnt/models/google/gemma-4-26B-A4B-it     fp16 stock /tmp/hwm_g26b_fp16
run "g26b FP8  — OUR plugin — drifting (the GREY one)"          /mnt/models/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic fp8 our /tmp/hwm_g26b_fp8

echo; echo "================= MATRIX SUMMARY ================="
for d in /tmp/hwm_q35b_fp16 /tmp/hwm_q35b_fp8 /tmp/hwm_g26b_fp16 /tmp/hwm_g26b_fp8; do
    echo "----- $(basename "$d") -----"; cat "$d/SUMMARY.txt" 2>/dev/null || echo "(no summary)"; echo
done
echo "READ: deterministic (q35b) both stock+our IDENTICAL => hardware clean + plugin clean."
echo "      drifting (g26b) both COHERENT + stock-also-drifts => g26b-fp8 drift = MoE routing, not our kernel."
