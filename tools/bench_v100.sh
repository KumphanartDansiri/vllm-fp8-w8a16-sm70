#!/usr/bin/env bash
# tools/bench_v100.sh -- file-captured serve + curl-loop helper for V100
# FP8 W8A16 benchmarking. Solves terminal-truncation and label-swap risks
# from Stage 2C peer review. Every run lands in a single timestamped
# directory containing the resolved config, the serve console, the curl
# loop, an extract of high-signal serve lines, and a numerical summary.
#
# Usage:
#   ./tools/bench_v100.sh serve <tag>
#       Launch the V100 FP8 serve in foreground, capturing console to
#       /tmp/v100_bench/<YYYYMMDD_HHMMSS>_<tag>/serve.log. Also writes
#       config.txt (git commit, env, command) into the same dir.
#       The latest run path is symlinked to /tmp/v100_bench/latest.
#       Example:
#         VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=0 \
#           ./tools/bench_v100.sh serve fast0
#
#   ./tools/bench_v100.sh curls [<tag_or_path>] [<n_curls>] [<prompt>]
#       Wait for the serve's /health endpoint, then run 1 warmup + N timed
#       curls, capture output to curls.log and a summary.txt in the same
#       directory as serve.log. Auto-extracts high-signal serve lines
#       into serve_extract.log.
#       <tag_or_path>: either a full path under /tmp/v100_bench/, or a
#       tag to look up the latest matching dir, or empty to use
#       /tmp/v100_bench/latest.
#       Example:
#         ./tools/bench_v100.sh curls fast0
#         ./tools/bench_v100.sh curls   # uses latest serve
#
# Resulting layout:
#   /tmp/v100_bench/<YYYYMMDD_HHMMSS>_<tag>/
#     config.txt           # git commit + resolved env + serve command
#     serve.log            # full serve stdout/stderr
#     curls.log            # full curl-loop stdout/stderr
#     serve_extract.log    # high-signal lines (banner, MoE profile, throughput)
#     summary.txt          # per-curl tok/s, mean (measured curls only)
#
# Env (all serve-time; defaults shown):
#   MODEL=/mnt/models/Qwen3.5-122B-A10B-FP8
#   PORT=8000
#   GPUS=0,1,2,3,4,5,6,7
#   TP_SIZE=8
#   QUANT=fp8                   (serve-time; set QUANT=none to omit --quantization)
#   MAX_MODEL_LEN=32768
#   MAX_TOKENS=200             (curl-time)
#   VLLM_V100_FP8_* (anything matching this prefix is recorded in config.txt)

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
            echo "[bench_v100] no /tmp/v100_bench/latest symlink; pass a tag or path" >&2
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
    match="$(ls -1dt "$BENCH_ROOT"/*_"$arg" 2>/dev/null | head -1 || true)"
    if [[ -n "$match" && -d "$match" ]]; then
        echo "$match"
        return 0
    fi
    echo "[bench_v100] no run dir found for '$arg' under $BENCH_ROOT" >&2
    return 1
}

write_config() {
    local out="$1"
    local mode="$2"
    {
        echo "# bench_v100.sh run config"
        echo "mode             : $mode"
        echo "timestamp        : $(date -Iseconds)"
        echo "host             : $(hostname)"
        echo "project_root     : $PROJECT_ROOT"
        echo "git_commit       : $(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo 'n/a')"
        echo "git_branch       : $(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'n/a')"
        echo "git_status_short : $(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | wc -l) files modified"
        echo ""
        echo "# resolved env (V100 FP8 vars only)"
        # `|| true`: grep returns 1 when no monkey-patch env vars are
        # exported (e.g. running a baseline against an unrelated model);
        # set -euo pipefail would otherwise abort write_config and the
        # whole script before the docker run command ever fires.
        env | grep -E '^VLLM_V100_FP8_' | sort || true
        echo ""
        echo "# resolved bench vars"
        echo "MODEL          = ${MODEL:-/mnt/models/Qwen3.5-122B-A10B-FP8}"
        echo "PORT           = ${PORT:-8000}"
        echo "GPUS           = ${GPUS:-0,1,2,3,4,5,6,7}"
        echo "TP_SIZE        = ${TP_SIZE:-8}"
        echo "QUANT          = ${QUANT:-fp8}"
        echo "MAX_MODEL_LEN  = ${MAX_MODEL_LEN:-32768}"
        echo "MAX_TOKENS     = ${MAX_TOKENS:-200}"
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
        QUANT="${QUANT:-fp8}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
        # ENFORCE_EAGER=1 (default) preserves the documented sm_70 path.
        # Set ENFORCE_EAGER=0 to exercise the cudagraph path — known to
        # crash on our cu128 stack at compiler_interface.py:382 due to a
        # vllm 0.18 / torch 2.10 / py3.10 standalone_compile.FakeTensorMode
        # mismatch (see no_eager_attempt.log). Useful for confirming that
        # failure mode against a fresh model arch.
        ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
        QUANT_ARGS=()
        QUANT_DESC="omitted"
        if [[ -n "$QUANT" && "$QUANT" != "none" ]]; then
            QUANT_ARGS=(--quantization "$QUANT")
            QUANT_DESC="--quantization $QUANT"
        fi
        EAGER_ARGS=()
        EAGER_DESC="--enforce-eager"
        if [[ "$ENFORCE_EAGER" == "0" || "$ENFORCE_EAGER" == "off" || "$ENFORCE_EAGER" == "false" ]]; then
            EAGER_DESC="(eager disabled — cudagraph path)"
        else
            EAGER_ARGS=(--enforce-eager)
        fi

        TS="$(date +%Y%m%d_%H%M%S)"
        RUN_DIR="$BENCH_ROOT/${TS}_${TAG}"
        mkdir -p "$RUN_DIR"
        ln -sfn "$RUN_DIR" "$BENCH_ROOT/latest"

        CONFIG="$RUN_DIR/config.txt"
        SERVE_LOG="$RUN_DIR/serve.log"
        write_config "$CONFIG" "serve"
        {
            echo ""
            echo "# serve command"
            echo "GPUS=\"$GPUS\" ./docker/run_docker.sh serve \\"
            echo "    --model $MODEL $QUANT_DESC --dtype float16 $EAGER_DESC \\"
            echo "    --attention-backend TRITON_ATTN --tensor-parallel-size $TP_SIZE \\"
            echo "    --max-num-seqs 1 --gpu-memory-utilization 0.80 \\"
            echo "    --max-model-len $MAX_MODEL_LEN --no-enable-chunked-prefill \\"
            # NOTE: --disable-custom-all-reduce was historically passed for
            # sm_70 safety. Source-read in Stage 2D Step 2C.1 found that vLLM
            # 0.18's CustomAllreduce auto-disables anyway on this DGX-1 V100
            # topology (NOT fully NVLink-connected at TP=8), so the flag is
            # dead weight here. Removed by default; pass it as a trailing
            # arg to this script if a future host needs it back.
            echo "    --host 0.0.0.0 --port $PORT $*"
        } >> "$CONFIG"

        echo "[bench_v100] run dir : $RUN_DIR"
        echo "[bench_v100] serve log: $SERVE_LOG"
        echo "[bench_v100] latest   : $BENCH_ROOT/latest -> $RUN_DIR"
        echo "[bench_v100] config   : $CONFIG"
        echo ""

        (
            cd "$PROJECT_ROOT"
            GPUS="$GPUS" \
            ./docker/run_docker.sh serve \
                --model "$MODEL" \
                "${QUANT_ARGS[@]}" --dtype float16 "${EAGER_ARGS[@]}" \
                --attention-backend TRITON_ATTN \
                --tensor-parallel-size "$TP_SIZE" \
                --max-num-seqs 1 --gpu-memory-utilization 0.80 \
                --max-model-len "$MAX_MODEL_LEN" --no-enable-chunked-prefill \
                --host 0.0.0.0 --port "$PORT" \
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

        # Update the existing config.txt (created by `serve`) with the
        # curl-side resolved settings, or write a fresh one if `serve`
        # wasn't run via this script.
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

        echo "[bench_v100] run dir   : $RUN_DIR"
        echo "[bench_v100] curls log : $CURLS_LOG"
        echo "[bench_v100] N         : $N_CURLS  max_tokens=$MAX_TOKENS"

        # Wait on /health (the vLLM serve registers /health as GET).
        echo "[bench_v100] waiting on http://localhost:$PORT/health ..."
        deadline=$(( $(date +%s) + 900 ))   # 15-minute cap for cold start
        until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
            if [[ $(date +%s) -gt $deadline ]]; then
                echo "[bench_v100] /health did not come up within 15 min; aborting" >&2
                exit 1
            fi
            sleep 2
        done
        echo "[bench_v100] /health ready"

        # IGNORE_EOS=1 disables natural EOS so the model is forced to
        # produce exactly MAX_TOKENS regardless of when it would have
        # stopped on its own. Needed for cross-config wall comparisons
        # (different TP shard arithmetic can break a logit tie at temp=0
        # and cause early EOS in one config but not another -- e.g.
        # Qw3.6-35B FP16 TP=4 stopped at 80 tokens while TP=8 ran to 200).
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
            echo "[bench_v100] $(date -Iseconds) curls starting in $RUN_DIR"
            echo "[bench_v100] prompt: $(echo "$PROMPT" | head -c 80)..."

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

        # Extract high-signal lines from the serve log into serve_extract.log
        # so the proof-of-config (fast_route_prep banner, MoE profile, engine
        # throughput logger) is co-located with the curl timings.
        #
        # Important: DECODE-BREAKDOWN multi-line blocks have an indented body
        # (e.g. "  Qwen3NextSparseMoeBlock ... ms/token", "    + moe_router
        # ..."). The grep below needs to catch those too, not just the
        # "DECODE-BREAKDOWN" header line, otherwise the breakdown body is
        # silently dropped from the extract.
        if [[ -f "$SERVE_LOG" ]]; then
            grep -E \
                -e 'fast_route_prep=' \
                -e 'V100-FP8-MOE-GROUPED' \
                -e 'V100-FP8-DBG-SHARED' \
                -e 'loggers\.py:259' \
                -e 'V100-FP8-MOE-PROFILE' \
                -e 'DECODE-BREAKDOWN' \
                -e '^\(Worker_TP[0-9]+ pid=[0-9]+\)[[:space:]]+(\+ |\+-- |Qwen|LogitsProcessor|Other|Total[[:space:]])' \
                -e 'attached hooks' \
                -e 'patched PWAL fired' \
                -e 'Application startup complete' \
                -e 'Stage 2D Step' \
                -e 'cross-cutting attribution' \
                -e 'row_parallel_ar' \
                "$SERVE_LOG" > "$EXTRACT" 2>/dev/null || true
            echo "[bench_v100] high-signal serve extract: $EXTRACT"
            echo "[bench_v100] grep'd $(wc -l < "$EXTRACT") lines"
        else
            echo "[bench_v100] no serve.log in $RUN_DIR (serve not started via this script?)"
        fi
        echo "[bench_v100] artifacts in $RUN_DIR :"
        ls -la "$RUN_DIR"
        ;;

    *)
        cat >&2 <<USAGE
usage: $0 <serve|curls> [args...]

  serve <tag>                       launch V100 FP8 serve, capture to
                                    /tmp/v100_bench/<TS>_<tag>/serve.log
  curls [<tag|path|empty>] [N] [prompt]
                                    run 1 warmup + N timed curls into
                                    the matching run dir; default N=4

env (serve-time):
  MODEL=/mnt/models/Qwen3.5-122B-A10B-FP8
  PORT=8000  GPUS=0,1,2,3,4,5,6,7  TP_SIZE=8  QUANT=fp8
  MAX_MODEL_LEN=32768
  QUANT=none omits --quantization for native FP16/BF16 checkpoints.
  VLLM_V100_FP8_* (any matching var is recorded in config.txt)

env (curl-time):
  MAX_TOKENS=200  (temperature is always 0 / greedy)

example A/B:
  Terminal 1: VLLM_V100_FP8_MOE_FAST_ROUTE_PREP=0 \\
              $0 serve fast0
  Terminal 2: $0 curls fast0
  (ctrl-c serve, change env, repeat with fast1)
USAGE
        exit 1
        ;;
esac
