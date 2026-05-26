#!/usr/bin/env bash
# tools/bench_1catai.sh -- file-captured serve + curl-loop helper for the
# 1Cat-vLLM v1.0.0 wheel image (parallel to tools/bench_v100.sh).
#
# Same /tmp/v100_bench/<TS>_<tag>/ layout so the GPT-fires-curl workflow
# stays identical. No monkey-patch env vars (1catai ships native FP8 SM70
# MoE + FA2-v100, so there are no VLLM_V100_FP8_* knobs to record).
#
# Usage:
#   ./tools/bench_1catai.sh serve <tag> [extra vllm serve args...]
#       Launch `vllm serve` inside the 1catai image, capturing console to
#       /tmp/v100_bench/<YYYYMMDD_HHMMSS>_1catai_<tag>/serve.log. Writes
#       config.txt with git/env/command in the same dir, symlinks
#       /tmp/v100_bench/latest to it.
#       Example:
#         ./tools/bench_1catai.sh serve default_tp8
#
#   ./tools/bench_1catai.sh curls [<tag_or_path>] [<n_curls>] [<prompt>]
#       Wait for /health, run 1 warmup + N timed curls, capture curls.log
#       + summary.txt + serve_extract.log in the same dir. Same shape as
#       bench_v100.sh.
#
# Env (serve-time defaults):
#   MODEL=/mnt/models/Qwen3.5-122B-A10B-FP8
#   PORT=8000  GPUS=0,1,2,3,4,5,6,7  TP_SIZE=8
#   MAX_MODEL_LEN=32768
#   EXTRA_SERVE_ARGS=""              (passed verbatim, e.g. "--enforce-eager")
#
# Env (curl-time):
#   MAX_TOKENS=200   IGNORE_EOS=0    (matches bench_v100.sh semantics)
#
# Note on defaults: by design this script does NOT pass
# --enforce-eager / --attention-backend / --quantization. The whole point
# of running the 1catai stack is to see whether its native defaults (FA2-v100,
# native FP8 SM70 MoE, possibly cudagraphs) load and serve. Use
# EXTRA_SERVE_ARGS to override.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
BENCH_ROOT="${V100_BENCH_ROOT:-/tmp/v100_bench}"
mkdir -p "$BENCH_ROOT"

resolve_run_dir() {
    local arg="${1:-}"
    if [[ -z "$arg" ]]; then
        local latest="$BENCH_ROOT/latest"
        if [[ ! -L "$latest" ]]; then
            echo "[bench_1catai] no /tmp/v100_bench/latest symlink; pass a tag or path" >&2
            return 1
        fi
        readlink -f "$latest"
        return 0
    fi
    if [[ -d "$arg" ]]; then
        echo "$arg"
        return 0
    fi
    if [[ -d "$BENCH_ROOT/$arg" ]]; then
        echo "$BENCH_ROOT/$arg"
        return 0
    fi
    local match
    match="$(ls -1dt "$BENCH_ROOT"/*_1catai_"$arg" 2>/dev/null | head -1 || true)"
    if [[ -z "$match" ]]; then
        match="$(ls -1dt "$BENCH_ROOT"/*_"$arg" 2>/dev/null | head -1 || true)"
    fi
    if [[ -n "$match" && -d "$match" ]]; then
        echo "$match"
        return 0
    fi
    echo "[bench_1catai] no run dir found for '$arg' under $BENCH_ROOT" >&2
    return 1
}

write_config() {
    local out="$1"
    local mode="$2"
    {
        echo "# bench_1catai.sh run config"
        echo "mode             : $mode"
        echo "stack            : 1Cat-vLLM v1.0.0 wheel image (py3.12 + torch 2.9.1+cu128)"
        echo "timestamp        : $(date -Iseconds)"
        echo "host             : $(hostname)"
        echo "project_root     : $PROJECT_ROOT"
        echo "git_commit       : $(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo 'n/a')"
        echo "git_branch       : $(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'n/a')"
        echo "git_status_short : $(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | wc -l) files modified"
        echo ""
        echo "# resolved bench vars"
        echo "MODEL             = ${MODEL:-/mnt/models/Qwen3.5-122B-A10B-FP8}"
        echo "PORT              = ${PORT:-8000}"
        echo "GPUS              = ${GPUS:-0,1,2,3,4,5,6,7}"
        echo "TP_SIZE           = ${TP_SIZE:-8}"
        echo "MAX_MODEL_LEN     = ${MAX_MODEL_LEN:-32768}"
        echo "EXTRA_SERVE_ARGS  = ${EXTRA_SERVE_ARGS:-}"
        echo "MAX_TOKENS        = ${MAX_TOKENS:-200}"
    } > "$out"
}

mode="${1:-help}"
shift || true

case "$mode" in
    serve)
        TAG="${1:-default}"
        shift || true
        MODEL="${MODEL:-/mnt/models/Qwen3.5-122B-A10B-FP8}"
        PORT="${PORT:-8000}"
        GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
        TP_SIZE="${TP_SIZE:-8}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
        EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"

        TS="$(date +%Y%m%d_%H%M%S)"
        RUN_DIR="$BENCH_ROOT/${TS}_1catai_${TAG}"
        mkdir -p "$RUN_DIR"
        ln -sfn "$RUN_DIR" "$BENCH_ROOT/latest"

        CONFIG="$RUN_DIR/config.txt"
        SERVE_LOG="$RUN_DIR/serve.log"
        write_config "$CONFIG" "serve"
        {
            echo ""
            echo "# serve command (1catai-vllm wheel image, no monkey-patches)"
            echo "GPUS=\"$GPUS\" ./docker/run_docker_1catai.sh serve \\"
            echo "    --model $MODEL --tensor-parallel-size $TP_SIZE \\"
            echo "    --max-num-seqs 1 --gpu-memory-utilization 0.80 \\"
            echo "    --max-model-len $MAX_MODEL_LEN \\"
            echo "    --host 0.0.0.0 --port $PORT $EXTRA_SERVE_ARGS $*"
        } >> "$CONFIG"

        echo "[bench_1catai] run dir : $RUN_DIR"
        echo "[bench_1catai] serve log: $SERVE_LOG"
        echo "[bench_1catai] latest   : $BENCH_ROOT/latest -> $RUN_DIR"
        echo "[bench_1catai] config   : $CONFIG"
        echo ""

        # Split EXTRA_SERVE_ARGS on whitespace into a real array, so the env
        # var can carry multiple flags without per-arg quoting gymnastics.
        EXTRA_ARGS=()
        if [[ -n "$EXTRA_SERVE_ARGS" ]]; then
            # shellcheck disable=SC2206
            EXTRA_ARGS=( $EXTRA_SERVE_ARGS )
        fi

        (
            cd "$PROJECT_ROOT"
            GPUS="$GPUS" \
            ./docker/run_docker_1catai.sh serve \
                --model "$MODEL" \
                --tensor-parallel-size "$TP_SIZE" \
                --max-num-seqs 1 --gpu-memory-utilization 0.80 \
                --max-model-len "$MAX_MODEL_LEN" \
                --host 0.0.0.0 --port "$PORT" \
                "${EXTRA_ARGS[@]}" \
                "$@"
        ) 2>&1 | tee "$SERVE_LOG"
        ;;

    curls)
        RUN_DIR="$(resolve_run_dir "${1:-}")"
        shift || true
        N_CURLS="${1:-4}"; shift || true
        PROMPT="${1:-The capital of France is}"; shift || true
        MODEL="${MODEL:-/mnt/models/Qwen3.5-122B-A10B-FP8}"
        PORT="${PORT:-8000}"
        MAX_TOKENS="${MAX_TOKENS:-200}"

        CURLS_LOG="$RUN_DIR/curls.log"
        SUMMARY="$RUN_DIR/summary.txt"
        EXTRACT="$RUN_DIR/serve_extract.log"
        SERVE_LOG="$RUN_DIR/serve.log"

        if [[ -f "$RUN_DIR/config.txt" ]]; then
            {
                echo ""
                echo "# curls invocation"
                echo "curls_timestamp : $(date -Iseconds)"
                echo "N_CURLS         : $N_CURLS"
                echo "MAX_TOKENS      : $MAX_TOKENS"
                echo "IGNORE_EOS      : ${IGNORE_EOS:-0}"
                echo "prompt          : $(echo "$PROMPT" | head -c 120)"
            } >> "$RUN_DIR/config.txt"
        else
            write_config "$RUN_DIR/config.txt" "curls-only"
        fi

        echo "[bench_1catai] run dir   : $RUN_DIR"
        echo "[bench_1catai] curls log : $CURLS_LOG"
        echo "[bench_1catai] N         : $N_CURLS  max_tokens=$MAX_TOKENS"

        echo "[bench_1catai] waiting on http://localhost:$PORT/health ..."
        deadline=$(( $(date +%s) + 900 ))
        until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
            if [[ $(date +%s) -gt $deadline ]]; then
                echo "[bench_1catai] /health did not come up within 15 min; aborting" >&2
                exit 1
            fi
            sleep 2
        done
        echo "[bench_1catai] /health ready"

        IGNORE_EOS="${IGNORE_EOS:-0}"
        BODY=$(IGNORE_EOS="$IGNORE_EOS" python3 -c "
import json, os, sys
payload = {
    'model': sys.argv[1],
    'prompt': sys.argv[2],
    'max_tokens': int(sys.argv[3]),
    'temperature': 0,
    'stream': False,
}
if os.environ.get('IGNORE_EOS', '0') not in ('0', 'off', 'false', ''):
    payload['ignore_eos'] = True
print(json.dumps(payload))
" "$MODEL" "$PROMPT" "$MAX_TOKENS")

        (
            echo "[bench_1catai] $(date -Iseconds) curls starting in $RUN_DIR"
            echo "[bench_1catai] prompt: $(echo "$PROMPT" | head -c 80)..."

            echo "=== warmup (untimed, not in mean) ==="
            curl -s "http://localhost:${PORT}/v1/completions" \
                -H 'Content-Type: application/json' \
                -d "$BODY" > /dev/null
            echo "warmup done"

            walls=()
            tokens=()
            for i in $(seq 1 "$N_CURLS"); do
                echo "=== Curl $i (measured) ==="
                start_s=$(date +%s.%N)
                response=$(curl -s "http://localhost:${PORT}/v1/completions" \
                    -H 'Content-Type: application/json' \
                    -d "$BODY")
                end_s=$(date +%s.%N)

                wall=$(python3 -c "print(round($end_s - $start_s, 3))")
                ct=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['usage']['completion_tokens'])")
                preview=$(echo "$response" | python3 -c "import sys, json; print(repr(json.load(sys.stdin)['choices'][0]['text'][:60]))")

                echo "  wall_s        = $wall"
                echo "  completion    = $ct tokens"
                echo "  first 60 chars= $preview"
                walls+=("$wall")
                tokens+=("$ct")
            done

            python3 - <<PY > "$SUMMARY"
walls = [$(IFS=,; echo "${walls[*]}")]
tokens = [$(IFS=,; echo "${tokens[*]}")]
per_run = [t / w for t, w in zip(tokens, walls)]
print('=== summary (measured curls only; warmup excluded) ===')
print('n_measured       :', len(walls))
print('per-curl wall_s  :', ['%.3f' % w for w in walls])
print('per-curl tokens  :', tokens)
print('per-curl tok/s   :', ['%.3f' % x for x in per_run])
print('mean tok/s       : %.3f' % (sum(per_run) / len(per_run)))
print('aggregate tok/s  : %.3f' % (sum(tokens) / sum(walls)))
PY
            cat "$SUMMARY"
        ) 2>&1 | tee "$CURLS_LOG"

        # Simpler extract than bench_v100.sh: no V100-FP8-* monkey-patch
        # banners exist in this stack. Capture engine/attn/MoE-config lines
        # plus the standard vLLM throughput logger.
        if [[ -f "$SERVE_LOG" ]]; then
            grep -E \
                -e 'Initializing.*LLM engine' \
                -e 'Started engine' \
                -e 'attention backend' \
                -e 'Attention backend' \
                -e 'fused_moe.*config' \
                -e 'MoE config' \
                -e 'Using default MoE' \
                -e 'cudagraph' \
                -e 'CUDAGraph' \
                -e 'capture' \
                -e 'loggers\.py:[0-9]+' \
                -e 'Application startup complete' \
                -e 'tok/s\|tokens/s' \
                -e 'flash_attn_v100' \
                -e 'fp8_sm70' \
                -e 'Fp8SM70' \
                -e 'ERROR' \
                "$SERVE_LOG" > "$EXTRACT" 2>/dev/null || true
            echo "[bench_1catai] high-signal serve extract: $EXTRACT"
            echo "[bench_1catai] grep'd $(wc -l < "$EXTRACT") lines"
        else
            echo "[bench_1catai] no serve.log in $RUN_DIR (serve not started via this script?)"
        fi
        echo "[bench_1catai] artifacts in $RUN_DIR :"
        ls -la "$RUN_DIR"
        ;;

    *)
        cat >&2 <<USAGE
usage: $0 <serve|curls> [args...]

  serve <tag>                       launch 1catai-vllm \`vllm serve\`, capture
                                    to /tmp/v100_bench/<TS>_1catai_<tag>/serve.log
  curls [<tag|path|empty>] [N] [prompt]
                                    1 warmup + N timed curls into the
                                    matching run dir; default N=4

env (serve-time):
  MODEL=/mnt/models/Qwen3.5-122B-A10B-FP8
  PORT=8000  GPUS=0,1,2,3,4,5,6,7  TP_SIZE=8
  MAX_MODEL_LEN=32768
  EXTRA_SERVE_ARGS="--enforce-eager"   (verbatim flags, optional)

env (curl-time):
  MAX_TOKENS=200  IGNORE_EOS=0
USAGE
        exit 1
        ;;
esac
