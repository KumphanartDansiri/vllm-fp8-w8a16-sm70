#!/usr/bin/env bash
# Phase 2 pre-flight: numerical test of the grouped-routed MoE kernel
# (fp8_w8a16_grouped_routed_gemm_a3) for CHANNEL-scale experts, BEFORE wiring it
# into GLM-Air. Mirrors the dense Linear numtest: compare the kernel against a
# per-expert dequant-FP16 routed reference, across aligned and partial N.
#
# WHY (GPT): the dense partial-N hazard is fixed in A.1/A.2/A.3, but the grouped
# kernel was NOT hardened yet (deferred — it's shared with the validated Qwen
# block-FP8 MoE path). GLM expert w13 N per TP8 shard = 2*1408/8 = 352 (NOT
# 128-aligned), so the grouped kernel will hit the same partial-CTA hazard. This
# test proves: (a) grouped kernel is correct for channel-scale on ALIGNED N
# (gated), and (b) it FAILS on partial N today (xfail) — the exact thing Phase 2
# kernel-hardening must flip to PASS before GLM-Air integration.
#
# Runs in the stock vllm021 image, ~seconds on ONE GPU. Usage:
#   ./tools/ct_fp8_grouped_moe_numtest_vllm021.sh
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

docker run --rm -i --name ct_grouped_numtest --gpus "\"device=$GPU\"" \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python3 - <<'PY'
import torch, torch.nn.functional as F
# Load the SAME cached extension vllm_serve uses (no recompile if source unchanged).
from fp8_w8a16_sm70.ext_loader import load_kernel
ext = load_kernel(name="fp8_dequant_ext_vllm")

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False

def run_case(E, N, K, R, k_split, seed=0, gated=True):
    g = torch.Generator(device=dev).manual_seed(seed)
    # per-expert per-output-row CHANNEL scale [E, N, 1] (positive)
    scale = torch.rand(E, N, 1, generator=g, device=dev, dtype=torch.float16) * 0.02 + 0.01
    w_ref = torch.randn(E, N, K, generator=g, device=dev, dtype=torch.float16) * 0.1
    w_q   = (w_ref / scale).to(torch.float8_e4m3fn)            # [E,N,K] FP8 (resident)
    A     = torch.randn(R, K, generator=g, device=dev, dtype=torch.float16) * 0.1
    expert_ids = torch.randint(0, E, (R,), generator=g, device=dev, dtype=torch.int64)

    # Reference: routed per-expert dequant-FP16 GEMM. out[r] = A[r] @ (w_q[e].f16*scale[e]).T
    ref = torch.empty(R, N, device=dev, dtype=torch.float32)
    for r in range(R):
        e = int(expert_ids[r].item())
        wdq = w_q[e].to(torch.float16) * scale[e]              # [N,K]
        ref[r] = (A[r].float() @ wdq.float().T)

    # Kernel (channel): scale expanded to [E, N, K/128], block_h=1, block_w=128.
    Kb = K // 128
    scale_blk = scale.expand(E, N, Kb).contiguous()            # [E,N,Kb] FP16
    w_u8 = w_q.view(torch.uint8).contiguous()                  # [E,N,K] bytes
    out = ext.fp8_w8a16_grouped_routed_gemm_a3(
        A.contiguous(), expert_ids.contiguous(), w_u8, scale_blk,
        N, K, 1, 128, k_split)                                 # -> [R,N] fp16

    a = out.float(); b = ref
    l2  = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    ok  = (cos > 0.9999 and l2 < 2e-2)
    aligned = (N % 128 == 0)
    label = ("PASS" if ok else "FAIL") if gated else ("xPASS" if ok else "XFAIL")
    print(f"[{label}] E={E} N={N:>5} K={K} R={R} ksplit={k_split} "
          f"{'aligned' if aligned else 'PARTIAL-N'} | L2rel={l2:.4f} cos={cos:.5f}")
    return ok

# All cases GATE the suite. Aligned shapes validate channel-scale correctness;
# the partial-N shapes (GLM expert w13: N=352 shared, N=2736 dense per TP8 shard)
# exercise the partial-CTA path hardened in fp8_w8a16_grouped_routed_gemm_a3 —
# garbage/nan before the fix, must PASS now (the proof the hardening worked).
print("-- channel-scale, aligned + partial N (all gated) --")
allok = True
allok &= run_case(E=8, N=1408, K=4096, R=16, k_split=8)
allok &= run_case(E=8, N=4096, K=4096, R=8,  k_split=8)
allok &= run_case(E=4, N=1408, K=2048, R=12, k_split=4)
allok &= run_case(E=8, N=352,  K=4096, R=16, k_split=8)   # shared gate_up shard (partial-N)
allok &= run_case(E=8, N=352,  K=4096, R=64, k_split=8)   # more routed rows
allok &= run_case(E=8, N=2736, K=4096, R=16, k_split=8)   # dense gate_up shard (partial-N)

print("\nRESULT:", "ALL PASS (incl. partial-N) — grouped channel kernel matches per-expert dequant reference"
      if allok else "FAIL — see rows above")
PY
rc=$?
echo "(exit $rc)"; exit $rc
