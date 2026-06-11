#!/usr/bin/env bash
# CH1 CHAIN — one launch, two engine passes back-to-back, unattended:
#   LEG 1: vLLM 0.21+cu126 — FULL Chapter 1 (all 10 cells, publishable headline).
#   LEG 2: vLLM 0.19+cu126 — Qwen3.6 engine-comparison (q27b + q35b, TP4; the two models proven on
#          0.19). 122B (Qwen3.5, TP8) and gemma (needs tf5 on 0.19) are NOT in the 0.19 leg — they
#          stay in the 0.21 leg / a later pass.
# Decompose: 0.19+cu126 vs 0.21+cu126 at FIXED wheel = isolates the engine (cu126≈cu128 already shown).
#
# CPU_GATE=0 by default here: the box has a stable 2-core idle drain (son's wedged extensionHost);
# env-control proved <1% drift → deltas valid, absolutes carry a small stable (conservative) offset.
# Each leg writes its OWN OUT dir (no clobber) + a REPORT.txt. ~4-5h total.
#
# Usage:  ./tools/ch1_chain.sh
#   then: cat /tmp/v100_ch1_021/REPORT.txt  /tmp/v100_ch1_019cu126/REPORT.txt
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CPU_GATE="${CPU_GATE:-0}"
leg() { echo; echo "[chain $(date -u +%H:%M:%SZ)] ===== $* ====="; echo; }

leg "CH1.1 — vLLM 0.21+cu126 : FULL Chapter 1 (all 10 cells)"
OUT=/tmp/v100_ch1.1_021 TAG=ch1.1 ENGINE_LABEL="vLLM 0.21+cu126" \
  IMAGE=vllm-v100:vllm021-cu126 CACHE_TAG=021cu126 \
  ./tools/ch1_reliability_bench.sh || true
python3 tools/ch1_report.py /tmp/v100_ch1.1_021/manifest.csv > /tmp/v100_ch1.1_021/REPORT.txt 2>&1 || true
echo "[chain] ch1.1 report -> /tmp/v100_ch1.1_021/REPORT.txt"

leg "CH1.2 — vLLM 0.19+cu126 : Qwen3.6 engine-comparison (q27b + q35b, TP4)"
OUT=/tmp/v100_ch1.2_019cu126 TAG=ch1.2 ENGINE_LABEL="vLLM 0.19+cu126" \
  IMAGE=vllm-v100-py312:vllm019-cu126 CACHE_TAG=019cu126 ONLY=q27b,q35b \
  ./tools/ch1_reliability_bench.sh || true
python3 tools/ch1_report.py /tmp/v100_ch1.2_019cu126/manifest.csv > /tmp/v100_ch1.2_019cu126/REPORT.txt 2>&1 || true
echo "[chain] ch1.2 report -> /tmp/v100_ch1.2_019cu126/REPORT.txt"

leg "CHAIN DONE"
echo "  CH1.1 vLLM 0.21+cu126 (Chapter 1)      : /tmp/v100_ch1.1_021/REPORT.txt   + manifest.csv"
echo "  CH1.2 vLLM 0.19+cu126 (engine compare) : /tmp/v100_ch1.2_019cu126/REPORT.txt + manifest.csv"
echo "  Engine delta = compare q27b/q35b FP8 decode across CH1.1 vs CH1.2 (0.21 vs 0.19, fixed cu126)."
