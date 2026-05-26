#!/usr/bin/env bash
# Build the NCCL-bump sibling image (vllm-v100-py312-nccl-test:cu128) and
# validate the override took. Baseline image (vllm-v100-py312-test:cu128)
# is not modified.
#
# Logs everything to ~/nccl_bump_build_<timestamp>.log so a tmux scrollback
# loss doesn't lose the build record.
#
# Usage: ./tools/build_nccl_bump.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

IMAGE="vllm-v100-py312-nccl-test:cu128"
DOCKERFILE="$PROJECT_ROOT/docker/Dockerfile.vllm018_py312_nccl"
LOG="$HOME/nccl_bump_build_$(date +%Y%m%d_%H%M%S).log"

{
    echo "=== Build start: $(date) ==="
    echo "Image:      $IMAGE"
    echo "Dockerfile: $DOCKERFILE"
    echo "Log:        $LOG"
    echo "Baseline:   vllm-v100-py312-test:cu128 (untouched)"
    echo
} | tee "$LOG"

# docker/ is sufficient as build context since the Dockerfile has no COPY
# (the base image already has everything). Small context = fast upload.
docker build -t "$IMAGE" -f "$DOCKERFILE" "$PROJECT_ROOT/docker" 2>&1 | tee -a "$LOG"

{
    echo
    echo "=== Post-build validation ==="
} | tee -a "$LOG"

docker run --rm -i "$IMAGE" python3 - <<'PY' 2>&1 | tee -a "$LOG"
import ctypes
import os
import subprocess
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("nccl (torch compiled API):", torch.cuda.nccl.version())
print("arch:", torch.cuda.get_arch_list())

# On-disk inspection: does the override actually live at the path torch will dlopen?
try:
    import nvidia.nccl as _nccl_pkg
    nccl_root = _nccl_pkg.__path__[0]
    print("nccl on-disk path:", nccl_root)
    lib_dir = os.path.join(nccl_root, "lib")
    if os.path.isdir(lib_dir):
        print("nccl lib contents:")
        for f in sorted(os.listdir(lib_dir)):
            full = os.path.join(lib_dir, f)
            target = ""
            if os.path.islink(full):
                target = " -> " + os.readlink(full)
            print(" ", f, target)
        lib_path = os.path.join(lib_dir, "libnccl.so.2")
        lib = ctypes.CDLL(lib_path)
        version = ctypes.c_int()
        rc = lib.ncclGetVersion(ctypes.byref(version))
        print("ncclGetVersion rc:", rc)
        print("ncclGetVersion raw:", version.value)
        major = version.value // 10000
        minor = (version.value % 10000) // 100
        patch = version.value % 100
        print("ncclGetVersion parsed:", (major, minor, patch))
except Exception as e:
    print("on-disk probe failed:", e)
PY

{
    echo
    echo "=== Build complete: $(date) ==="
    echo "Image tagged: $IMAGE"
    echo "Log saved:    $LOG"
    echo
    echo "Next steps (when you're back):"
    echo "  1. Inspect the validation output above. torch's compiled NCCL API may"
    echo "     still say 2.27.5; ncclGetVersion from libnccl.so should be 2.30.4."
    echo "  2. Smoke-test by serving the 122B-FP8 model with this image:"
    echo "       # Edit IMAGE in docker/run_docker_vllm018_py312.sh temporarily,"
    echo "       # OR run docker directly with --env NCCL_DEBUG=VERSION to see"
    echo "       # the runtime banner on first init."
    echo "  3. If smoke matrix improves, promote this to the new baseline."
    echo "  4. If smoke matrix regresses, the baseline image is still intact;"
    echo "     just don't use this one."
} | tee -a "$LOG"
