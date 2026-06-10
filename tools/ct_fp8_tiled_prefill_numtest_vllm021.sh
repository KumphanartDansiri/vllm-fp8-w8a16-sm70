#!/usr/bin/env bash
# Phase 4 Stage 1 correctness test: the TILED PREFILL path (per-expert a2(w13) +
# cuBLAS(w2), grouped-by-expert) must match the per-row grouped path AND an FP32
# reference, at prefill R. Runs the SAME `_v100_ct_mixed_moe_routed` twice — once
# with VLLM_V100_CT_MOE_PREFILL_TILED forced on, once off — on synthetic GLM-Air
# expert shapes (2I=352 partial-N, I=176, H=4096). No model, no 8-GPU.
#
# Usage: ./tools/ct_fp8_tiled_prefill_numtest_vllm021.sh
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

docker run --rm -i --name ct_tiled_prefill_numtest --gpus "\"device=$GPU\"" \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python3 - <<'PY'
import types, torch, torch.nn.functional as F
import fp8_w8a16_sm70.compressed_tensors_v100 as ctv
from fp8_w8a16_sm70.compressed_tensors_v100 import _v100_ct_mixed_moe_routed
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False

def build_layer(E, I, H, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    N13 = 2 * I
    s13 = torch.rand(E, N13, 1, generator=g, device=dev, dtype=torch.float16) * 0.02 + 0.01
    w13_ref = torch.randn(E, N13, H, generator=g, device=dev, dtype=torch.float16) * 0.1
    w13_q = (w13_ref / s13).to(torch.float8_e4m3fn)
    Kb = H // 128
    mix = {"u8": w13_q.view(torch.uint8).contiguous(),
           "scale": s13.expand(E, N13, Kb).contiguous(), "N": N13, "H": H, "validated": None}
    w2 = torch.randn(E, H, I, generator=g, device=dev, dtype=torch.float16) * 0.1
    layer = types.SimpleNamespace(_v100_w13_mix=mix,
        w2_weight=types.SimpleNamespace(data=w2),
        apply_router_weight_on_input=False, activation="silu", shared_experts=None)
    return layer, w13_q, s13, w2

def ref_moe(x, topk_w, topk_ids, w13_q, s13, w2, I):
    M, H = x.shape; topk = topk_ids.size(-1)
    w13_dq = w13_q.to(torch.float32) * s13.to(torch.float32); w2f = w2.float()
    out = torch.zeros(M, H, device=x.device, dtype=torch.float32)
    for m in range(M):
        for j in range(topk):
            e = int(topk_ids[m, j].item()); wt = float(topk_w[m, j].item())
            gu = x[m].float() @ w13_dq[e].T
            h = F.silu(gu[:I]) * gu[I:]
            out[m] += wt * (h @ w2f[e].T)
    return out

def run_case(E, I, H, M, topk, seed=0):
    layer, w13_q, s13, w2 = build_layer(E, I, H, seed)
    g = torch.Generator(device=dev).manual_seed(seed + 100)
    x = torch.randn(M, H, generator=g, device=dev, dtype=torch.float16) * 0.1
    topk_ids = torch.stack([torch.randperm(E, generator=g, device=dev)[:topk] for _ in range(M)]).to(torch.int64)
    topk_w = torch.rand(M, topk, generator=g, device=dev, dtype=torch.float16) * 0.5 + 0.1
    R = M * topk

    ctv._CT_MOE_PREFILL_TILED_MIN_R = 1   # force the tiled branch to be reachable
    ctv._CT_MOE_PREFILL_TILED = True
    tiled = _v100_ct_mixed_moe_routed(layer, x, topk_w, topk_ids).float()
    ctv._CT_MOE_PREFILL_TILED = False     # per-row grouped path
    grouped = _v100_ct_mixed_moe_routed(layer, x, topk_w, topk_ids).float()
    ref = ref_moe(x, topk_w, topk_ids, w13_q, s13, w2, I)

    def l2(a, b): return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    tg = l2(tiled, grouped); tr = l2(tiled, ref); gr = l2(grouped, ref)
    cos = F.cosine_similarity(tiled.flatten(), grouped.flatten(), dim=0).item()
    ok = (tg < 5e-3 and tr < 2e-2 and cos > 0.9999)
    print(f"[{'PASS' if ok else 'FAIL'}] E={E} I={I} H={H} M={M} topk={topk} R={R} | "
          f"tiled-vs-grouped L2={tg:.4f} cos={cos:.5f} | tiled-vs-ref={tr:.4f} grouped-vs-ref={gr:.4f}")
    return ok

print("-- tiled prefill (per-expert a2+cuBLAS) vs per-row grouped vs FP32 ref, GLM-Air shapes --")
allok = True
allok &= run_case(E=8,  I=176, H=4096, M=32,  topk=8)   # R=256 (= MIN_R boundary)
allok &= run_case(E=8,  I=176, H=4096, M=128, topk=8)   # R=1024
allok &= run_case(E=128,I=176, H=4096, M=64,  topk=8)   # R=512, 128 experts (GLM-Air)
allok &= run_case(E=16, I=176, H=4096, M=256, topk=4)   # R=1024
allok &= run_case(E=8,  I=256, H=4096, M=64,  topk=2)   # 2I=512 aligned-N sanity
print("\nRESULT:", "ALL PASS — tiled prefill matches per-row grouped + reference"
      if allok else "FAIL — tiled prefill diverges (see rows)")
import sys; sys.exit(0 if allok else 1)
PY
rc=$?
echo "(exit $rc)"; exit $rc
