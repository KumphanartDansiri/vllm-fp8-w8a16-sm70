#!/usr/bin/env bash
# Fan the Volta MoE shell-walk tuner across all 8 V100s.
# Each (shape, M) is an independent tuning job (own shell-walk, own best config),
# so we shard the 2 shapes x 9 batch sizes = 18 jobs round-robin (descending M, so
# the heavy jobs start first) over the GPUs, then merge per-M results into the two
# canonical JSONs. Synthetic-kernel tuning => no model mount, fast container start.
#
# Usage:  ./tools/moe_volta_tune_fleet.sh
# Env: IMAGE NGPU MS
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"

IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
NGPU="${NGPU:-8}"
MS="${MS:-64 48 32 24 16 8 4 2 1}"        # descending: longest pole first
STAMP="$(date -u +%Y%m%d_%H%M%S)"
FLEET="results/moe_volta_fleet_${STAMP}"
mkdir -p "$FLEET"
LOG="$FLEET/fleet.log"

# shape key : E : TOPK : K : NSHARD
SHAPES=(
  "q35b:256:8:2048:128"
  "g26b:128:8:2816:176"
)

note(){ echo "[fleet $(date +%T)] $*" | tee -a "$LOG"; }

# ── build the job list (descending M, both shapes), assign round-robin to GPUs ──
JOBS=()
for m in $MS; do
  for s in "${SHAPES[@]}"; do JOBS+=("$s:$m"); done
done
declare -a GPU_JOBS
for i in "${!JOBS[@]}"; do
  g=$(( i % NGPU ))
  GPU_JOBS[$g]="${GPU_JOBS[$g]:-} ${JOBS[$i]}"
done

note "image=$IMAGE ngpu=$NGPU  ${#JOBS[@]} jobs over $NGPU GPUs"
for g in $(seq 0 $((NGPU-1))); do note "  GPU$g:${GPU_JOBS[$g]:-}"; done

# ── one background worker per GPU; runs its jobs sequentially ──
run_gpu() {
  local g="$1"; shift
  for job in $@; do
    IFS=':' read -r skey e topk k nshard m <<<"$job"
    local outdir="$FLEET/${skey}_M${m}"
    mkdir -p "$outdir"
    note "GPU$g START $skey M=$m"
    docker run --rm --gpus "\"device=$g\"" \
      -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
      --shm-size=4g \
      -e TUNE_E="$e" -e TUNE_TOPK="$topk" -e TUNE_K="$k" -e TUNE_NSHARD="$nshard" \
      -e TUNE_M="$m" \
      "$IMAGE" \
      python3 tools/moe_volta_tune.py "/work/$outdir" \
      >"$outdir/console.log" 2>&1 \
      && note "GPU$g DONE  $skey M=$m -> $(grep -E '^M=' "$outdir/tune.log" | tail -1)" \
      || note "GPU$g FAIL  $skey M=$m (see $outdir/console.log)"
  done
}

for g in $(seq 0 $((NGPU-1))); do
  run_gpu "$g" ${GPU_JOBS[$g]:-} &
done
wait
note "all jobs finished; merging per-shape JSONs"

# ── merge each shape's per-M JSONs into one canonical file ──
for s in "${SHAPES[@]}"; do
  IFS=':' read -r skey e topk k nshard <<<"$s"
  python3 - "$FLEET" "$skey" "$e" "$nshard" "$PROJECT_ROOT" <<'PY' | tee -a "$LOG"
import glob, json, os, sys
fleet, skey, E, N, root = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
fname = f"E={E},N={N},device_name=Tesla_V100-SXM2-32GB.json"
merged, tv = {}, None
for jf in sorted(glob.glob(f"{fleet}/{skey}_M*/{fname}")):
    d = json.load(open(jf))
    tv = d.pop("triton_version", tv)
    merged.update(d)
if not merged:
    print(f"{skey}: NO per-M JSONs found — check console logs"); sys.exit(0)
outdir = os.path.join(root, f"results/moe_volta_tune_{skey}")
os.makedirs(outdir, exist_ok=True)
ordered = {k: merged[k] for k in sorted(merged, key=lambda x: int(x))}
payload = {"triton_version": tv, **ordered}
outp = os.path.join(outdir, fname)
json.dump(payload, open(outp, "w"), indent=4)
print(f"{skey}: merged {len(merged)} batch sizes -> {outp}")
for mk in sorted(merged, key=lambda x: int(x)):
    c = merged[mk]
    print(f"   M={mk:>4}: {c['BLOCK_SIZE_M']}/{c['BLOCK_SIZE_N']}/{c['BLOCK_SIZE_K']} "
          f"w{c['num_warps']} s{c['num_stages']} g{c['GROUP_SIZE_M']}")
PY
done

note "FLEET COMPLETE -> $FLEET ; merged JSONs in results/moe_volta_tune_{q35b,g26b}/"
echo "early-stop verdicts:" | tee -a "$LOG"
grep -h "shell early-stop" "$FLEET"/*/tune.log 2>/dev/null | sort | uniq -c | tee -a "$LOG"
