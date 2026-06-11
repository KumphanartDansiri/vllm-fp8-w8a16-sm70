#!/usr/bin/env bash
# CH2 CHAIN — overnight MTP sweep, one launch, unattended (~3-4h on an idle box).
# Answers the Ch2 question raised by the first datapoint (35B k=1 @512tok = 1.00x, accept 84.9%):
# "does MTP help AT ALL on the coalesced-FP8 V100 stack?" — and if not, WHERE does the gain die.
#
# Cells (each = tools/ch2_mtp_ab.sh, own OUT dir, GENTOK=1024 steady-state):
#   1. q35b-fp8   KLIST="1 2"  TP4 — re-anchor k=1 at 1024 tok + first k=2 (deeper speculation).
#   2. q122b-fp8  KLIST="1 2"  TP8 — FLAGSHIP. 0.18-stack MTP headline was 45-47 vs 34.76 (1.36x);
#                 coalesced baseline is now ~44 — does MTP still add on top, at TP8 comm-heavy?
#   3. q27b-fp8   KLIST="1"    TP4 — hybrid/dense-ish path datapoint (MTP was lossless on dense FP16).
#   4. q35b-fp16  KLIST="1"    TP4 — STOCK comparator, no our-kernels: if stock FP16 also shows ~1.00x,
#                 the no-win is V100/engine-wide, NOT our FP8 stack. Also the "FP16-MoE-MTP-diverges-too"
#                 exactness datapoint (exonerates the plugin).
#   5. g31b-fp8   KLIST="1"    TP4 — EXPLORATORY gemma-4 mtp module via the same spec-config; may FAIL
#                 to load (unproven combo) — last on purpose, failure doesn't block anything.
#
# Results land DURABLY in results/ch2_mtp_<date>/ (NOT /tmp). Per-cell SUMMARY.txt + samples + serve
# logs + spec/capture evidence; aggregated CHAIN_SUMMARY.txt at the end.
#
# Launch (user, idle box):   tmux new-session -d -s ch2chain ./tools/ch2_chain.sh
# Watch:                     tmux attach -t ch2chain    (or tail -f results/ch2_mtp_*/chain_console.log)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE="${BASE:-results/ch2_mtp_$(date +%Y%m%d)}"; mkdir -p "$BASE"
exec > >(tee -a "$BASE/chain_console.log") 2>&1
leg(){ echo; echo "[ch2-chain $(date -u +%H:%M:%SZ)] ===== $* ====="; echo; }

# clean-box guard: refuse to start a multi-hour chain on a busy box (per-cell guards re-check too)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader|awk '{s+=$1}END{print s+0}')
[[ "$USED" -ge 2000 ]] && { echo "ABORT: GPUs busy (${USED} MiB in use) — run on an idle box."; exit 1; }

GENTOK="${GENTOK:-1024}"

leg "CELL 1/5 — q35b-fp8 (off vs k=1 vs k=2) — re-anchor + deeper speculation"
OUT="$BASE/q35b_fp8" MODEL=/mnt/models/Qwen/Qwen3.6-35B-A3B-FP8 TP=4 PREC=fp8 \
  KLIST="1 2" GENTOK="$GENTOK" ./tools/ch2_mtp_ab.sh || true

leg "CELL 2/5 — q122b-fp8 TP8 (off vs k=1 vs k=2) — the flagship MTP question"
OUT="$BASE/q122b_fp8" MODEL=/mnt/models/Qwen/Qwen3.5-122B-A10B-FP8 TP=8 PREC=fp8 \
  KLIST="1 2" GENTOK="$GENTOK" ./tools/ch2_mtp_ab.sh || true

leg "CELL 3/5 — q27b-fp8 (off vs k=1) — hybrid/dense-ish path"
OUT="$BASE/q27b_fp8" MODEL=/mnt/models/Qwen/Qwen3.6-27B-FP8 TP=4 PREC=fp8 \
  KLIST="1" GENTOK="$GENTOK" ./tools/ch2_mtp_ab.sh || true

leg "CELL 4/5 — q35b-fp16 STOCK (off vs k=1) — engine-wide vs our-stack disambiguator"
OUT="$BASE/q35b_fp16" MODEL=/mnt/models/Qwen/Qwen3.6-35B-A3B TP=4 PREC=fp16 \
  KLIST="1" GENTOK="$GENTOK" ./tools/ch2_mtp_ab.sh || true

leg "CELL 5/5 — g31b-fp8 EXPLORATORY (off vs k=1) — gemma-4 mtp module, may FAIL"
OUT="$BASE/g31b_fp8" MODEL=/mnt/models/RedHatAI/gemma-4-31B-it-FP8-Dynamic TP=4 PREC=fp8 \
  KLIST="1" GENTOK="$GENTOK" ./tools/ch2_mtp_ab.sh || true

leg "CHAIN DONE — aggregating"
AGG="$BASE/CHAIN_SUMMARY.txt"; : > "$AGG"
for d in q35b_fp8 q122b_fp8 q27b_fp8 q35b_fp16 g31b_fp8; do
    echo "──── $d ────" >> "$AGG"
    cat "$BASE/$d/SUMMARY.txt" >> "$AGG" 2>/dev/null || echo "(no summary — cell did not run)" >> "$AGG"
    echo >> "$AGG"
done
cat "$AGG"
echo "Durable results: $BASE/  (CHAIN_SUMMARY.txt + per-cell SUMMARY.txt/samples/serve logs/spec evidence)"
echo "Divergence figures: diff <cell>/off_sample.txt <cell>/mtp1_sample.txt  (or ch1_exactness_viz.py)"
