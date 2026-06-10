#!/usr/bin/env bash
# Phase 2d pre-flight: numerical test of the FP16 grouped-routed GEMM
# (fp16_grouped_routed_gemm) that REPLACES the per-expert Python loop for GLM w2
# (down). Compares the kernel against the exact per-expert dequant-free FP16
# routed reference `expert_out[r] = hidden[r] @ w2[expert_ids[r]].T`.
#
# WHY this kernel (vs the FP8 grouped A.3 path): w2's contraction dim K=I/TP=176
# is NOT 128-aligned and the weight is kept FP16 (no FP8 rounding on w2). This
# kernel uses scalar __half loads (not uint4), so it handles K=176 directly with
# no padding, while still killing the `torch.unique` CPU sync + per-expert matmul
# loop that pinned mixed decode at ~1 tok/s.
#
# Coverage: aligned + partial N (the partial-CTA masking), the GLM K=176 tail,
# k_split=1 and 2, invalid expert ids (-1 -> row stays 0), and R>0/R=0.
#
# Runs in the stock vllm021 image, ~seconds on ONE GPU. Usage:
#   ./tools/ct_fp16_w2_grouped_numtest_vllm021.sh
# Env: IMAGE GPU CACHE_TAG

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"
GPU="${GPU:-0}"
CACHE_TAG="${CACHE_TAG:-021}"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE missing"; exit 1; }
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk '{print $1+0}')
[[ "${used:-9999}" -le 2000 ]] || echo "WARN: GPU $GPU has ${used} MiB used (shared box?). Set GPU= to a free one."

docker run --rm -i --name ct_fp16_w2_numtest --gpus "\"device=$GPU\"" \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python3 - <<'PY'
import torch, torch.nn.functional as F
# Same cached extension vllm_serve uses (no recompile if source unchanged).
from fp8_w8a16_sm70.ext_loader import load_kernel
ext = load_kernel(name="fp8_dequant_ext_vllm")
assert hasattr(ext, "fp16_grouped_routed_gemm"), \
    "fp16_grouped_routed_gemm not in extension — rebuild (source changed?)"

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False

def run_case(E, N, K, R, k_split, seed=0, with_invalid=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    # w2: FP16 [E, N, K] (N = hidden H, K = intermediate-shard I)
    w2 = torch.randn(E, N, K, generator=g, device=dev, dtype=torch.float16) * 0.1
    A  = torch.randn(R, K, generator=g, device=dev, dtype=torch.float16) * 0.1
    expert_ids = torch.randint(0, E, (R,), generator=g, device=dev, dtype=torch.int64)
    if with_invalid and R >= 2:
        expert_ids[0] = -1          # invalid route -> output row must stay 0

    # Reference: exact per-expert FP16 routed GEMM (FP32 accum), what the kernel replaces.
    ref = torch.zeros(R, N, device=dev, dtype=torch.float32)
    for r in range(R):
        e = int(expert_ids[r].item())
        if e < 0:
            continue
        ref[r] = (A[r].float() @ w2[e].float().T)

    out = ext.fp16_grouped_routed_gemm(
        A.contiguous(), expert_ids.contiguous(), w2.contiguous(), k_split).float()

    denom = ref.norm().clamp_min(1e-12)
    l2 = ((out - ref).norm() / denom).item()
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item() if denom > 1e-9 else 1.0
    # If row0 was forced invalid, assert it is exactly zero.
    zero_ok = True
    if with_invalid and R >= 2:
        zero_ok = bool((out[0].abs().max() == 0))
    ok = (cos > 0.9999 and l2 < 2e-3 and zero_ok)
    tag = "PASS" if ok else "FAIL"
    inv = " invalid-row0" if with_invalid else ""
    npart = "" if (N % 128 == 0) else " [partial-N]"
    print(f"[{tag}] E={E} N={N} K={K} R={R} ksplit={k_split}{inv}{npart} | "
          f"L2rel={l2:.2e} cos={cos:.6f} zero_ok={zero_ok}")
    return ok

print("-- fp16_grouped_routed_gemm vs per-expert FP16 routed reference (GLM w2 shapes) --")
allok = True
# GLM-Air w2: N=H=4096, K=I/TP=176, E=128 experts. Decode + small-batch + prefill R.
allok &= run_case(E=128, N=4096, K=176, R=1,    k_split=1)   # decode (M=1, 1 routed row shown)
allok &= run_case(E=128, N=4096, K=176, R=8,    k_split=1)   # decode M=1 topk=8
allok &= run_case(E=128, N=4096, K=176, R=64,   k_split=1)
allok &= run_case(E=128, N=4096, K=176, R=512,  k_split=1)   # prefill-ish
allok &= run_case(E=128, N=4096, K=176, R=8,    k_split=2)   # K=176 not /2-even? 176/2=88 ok
allok &= run_case(E=128, N=4096, K=176, R=64,   k_split=2)
allok &= run_case(E=128, N=4096, K=176, R=64,   k_split=1, with_invalid=True)  # -1 route
allok &= run_case(E=32,  N=4000, K=176, R=64,   k_split=1)   # partial-N (4000 % 128 = 32)
allok &= run_case(E=8,   N=2048, K=256, R=32,   k_split=2)   # aligned sanity
allok &= run_case(E=8,   N=4096, K=176, R=0,    k_split=1)   # empty -> no launch, zeros
print("\nRESULT:", "ALL PASS — fp16_grouped_routed_gemm matches the per-expert reference"
      if allok else "FAIL — grouped FP16 w2 kernel diverges (see rows above)")
import sys; sys.exit(0 if allok else 1)
PY
rc=$?
echo "(exit $rc)"; exit $rc
