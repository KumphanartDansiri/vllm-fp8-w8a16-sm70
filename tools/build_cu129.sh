#!/usr/bin/env bash
# Build the cu129 sibling image (vllm-v100-py312-test:cu129).
# Baseline cu128 image (vllm-v100-py312-test:cu128) is not modified.
#
# This is the NVIDIA-recommended V100 endgame: driver R580 + CUDA 12.9
# toolkit. On the current R535 host, the image relies on NVIDIA's CUDA
# Forward Compatibility shim (cuda-compat-12-9) which is installed in the
# Dockerfile. If forward-compat fails at runtime, the bare-metal R580
# upgrade is the next required step.
#
# Logs everything to ~/cu129_build_<timestamp>.log.
#
# Usage: ./tools/build_cu129.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

IMAGE="vllm-v100-py312-test:cu129"
DOCKERFILE="$PROJECT_ROOT/docker/Dockerfile.vllm018_py312_cu129"
LOG="$HOME/cu129_build_$(date +%Y%m%d_%H%M%S).log"

{
    echo "=== Build start: $(date) ==="
    echo "Image:      $IMAGE"
    echo "Dockerfile: $DOCKERFILE"
    echo "Log:        $LOG"
    echo "Baseline:   vllm-v100-py312-test:cu128 (untouched)"
    echo
    echo "Expected duration: 15-25 min (base image pull + torch+cu129 wheel"
    echo "download + vllm install). Parallel-safe with GPU work."
    echo
} | tee "$LOG"

# docker/ is sufficient as build context. The Dockerfile has no COPY ops.
docker build -t "$IMAGE" -f "$DOCKERFILE" "$PROJECT_ROOT/docker" 2>&1 | tee -a "$LOG"

{
    echo
    echo "=== Post-build validation (no GPU) ==="
} | tee -a "$LOG"

# Post-build validation runs with --gpus all so torch.cuda.get_arch_list()
# returns the real compile-time arch set (it returns [] without GPU access).
# This is also the forward-compat probe: if R535 can't satisfy cu129 toolkit
# requirements, the matmul below will raise CUDA_ERROR_INSUFFICIENT_DRIVER.
docker run --rm -i --gpus all "$IMAGE" python3 - <<'PY' 2>&1 | tee -a "$LOG"
import os
import sys
import torch
import vllm

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.get_arch_list():", torch.cuda.get_arch_list())
print("torch compile-time NCCL:", torch.cuda.nccl.version())
print("vllm:", vllm.__version__)

# sm_70 check, now with GPU access so the arch list is populated.
arch_list = torch.cuda.get_arch_list()
if "sm_70" not in arch_list:
    print(f"FAIL: sm_70 NOT in arch_list: {arch_list}")
    sys.exit(1)
print(f"OK: sm_70 in arch_list")

# Check forward-compat shim is present on disk.
compat_dir = "/usr/local/cuda-12.9/compat"
if os.path.isdir(compat_dir):
    print("cuda-compat-12-9 directory:", compat_dir)
    print("contents:")
    for f in sorted(os.listdir(compat_dir)):
        print(" ", f)
else:
    print("WARNING: cuda-compat-12-9 directory not found at expected path")

# Forward-compat probe: does cu129 actually work on the current host driver?
# If R535 is too old and cuda-compat-12-9 shim isn't loading, this will
# raise CUDA_ERROR_INSUFFICIENT_DRIVER (35) and tell us we're in scenario B.
print()
print("=== Forward-compat probe ===")
try:
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    x = torch.ones(1, device="cuda")
    y = (x @ x).item()
    print(f"matmul on cuda:0 = {y}  ->  SCENARIO A (cu129 works on R535)")
except Exception as e:
    print(f"FAIL on GPU op: {type(e).__name__}: {e}")
    print("  ->  SCENARIO B (forward-compat insufficient, R580 upgrade required)")
    sys.exit(2)
PY

{
    echo
    echo "=== Build complete: $(date) ==="
    echo "Image tagged: $IMAGE"
    echo "Log saved:    $LOG"
    echo
    echo "Next: GPU smoke test on R535 (current driver)."
    echo "  Scenario A (works): forward-compat shim functional, run smoke matrix."
    echo "  Scenario B (driver-too-old): wait for R580 host upgrade, retry."
    echo
    echo "Quick GPU compat probe (no model needed):"
    echo "  docker run --rm --gpus all $IMAGE python3 -c \\"
    echo "      \"import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); \\"
    echo "       x = torch.ones(1, device='cuda'); print('matmul:', (x @ x).item())\""
    echo
    echo "If that prints True, 8, and matmul: 1.0 -> scenario A, proceed to smoke."
    echo "If it raises CUDA_ERROR_INSUFFICIENT_DRIVER (35) or similar -> scenario B."
} | tee -a "$LOG"
