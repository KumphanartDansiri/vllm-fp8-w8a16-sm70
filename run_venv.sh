#!/usr/bin/env bash
# Alternative to Docker: local uv venv in your home, no root needed.
# Follows the AGENTS.md recommended workflow.
#
# Usage:
#   ./run_venv.sh setup   # one-time: create venv at ./.venv and install torch
#   ./run_venv.sh test    # run the FP8->FP16 hello-world test
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install with:"
    echo '    curl -LsSf https://astral.sh/uv/install.sh | sh'
    exit 1
fi

# Make nvcc visible.
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

case "${1:-test}" in
    setup)
        uv venv --python 3.12 "$VENV"
        source "$VENV/bin/activate"
        # CUDA 12.4 wheels to match host CUDA.
        uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
        uv pip install ninja
        echo
        echo "Setup done. Now: ./run_venv.sh test"
        ;;
    test)
        if [[ ! -f "$VENV/bin/activate" ]]; then
            echo "venv not found at $VENV — run: $0 setup" >&2
            exit 1
        fi
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
        python test_fp8.py
        ;;
    *)
        echo "usage: $0 {setup|test}" >&2
        exit 1
        ;;
esac
