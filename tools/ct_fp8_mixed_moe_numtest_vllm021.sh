#!/usr/bin/env bash
# Phase 2b integration test: validate the MIXED routed MoE forward
# (_v100_ct_mixed_moe_routed: route-prep -> w13 FP8 grouped kernel -> silu(gate)*up
# -> w2 FP16 per-expert -> route_w -> scatter) against a reference FP16 MoE, on
# SMALL synthetic data. No model, no 8-GPU, no OOM — this is what lets us trust
# the true w13-resident path (which frees FP16 w13 and removes the per-layer
# fallback). Mirrors GLM-Air expert shapes: 2I=352 (partial-N for w13), I=176, H=4096.
#
# Usage: ./tools/ct_fp8_mixed_moe_numtest_vllm021.sh
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

docker run --rm -i --name ct_mixed_moe_numtest --gpus "\"device=$GPU\"" \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e VLLM_V100_CT_MOE_W13_CHUNK="${W13_CHUNK:-64}" \
    "$IMAGE" python3 - <<'PY'
import types, torch, torch.nn.functional as F
from fp8_w8a16_sm70.compressed_tensors_v100 import _v100_ct_mixed_moe_routed

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False

def build_layer(E, I, H, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    N13 = 2 * I
    # w13: per-expert per-row CHANNEL scale; FP8 weight [E, 2I, H]
    s13 = torch.rand(E, N13, 1, generator=g, device=dev, dtype=torch.float16) * 0.02 + 0.01
    w13_ref = torch.randn(E, N13, H, generator=g, device=dev, dtype=torch.float16) * 0.1
    w13_q = (w13_ref / s13).to(torch.float8_e4m3fn)            # [E,2I,H] FP8
    Kb = H // 128
    mix = {
        "u8": w13_q.view(torch.uint8).contiguous(),
        "scale": s13.expand(E, N13, Kb).contiguous(),
        "N": N13, "H": H, "validated": None,
    }
    # w2: FP16 [E, H, I] (the mixed path keeps w2 FP16)
    w2 = torch.randn(E, H, I, generator=g, device=dev, dtype=torch.float16) * 0.1
    layer = types.SimpleNamespace(
        _v100_w13_mix=mix,
        w2_weight=types.SimpleNamespace(data=w2),
        apply_router_weight_on_input=False,
        activation="silu",
    )
    return layer, w13_q, s13, w2

def ref_moe(x, topk_w, topk_ids, w13_q, s13, w2, I):
    M, H = x.shape
    topk = topk_ids.size(-1)
    w13_dq = w13_q.to(torch.float32) * s13.to(torch.float32)   # [E,2I,H]
    w2f = w2.float()
    out = torch.zeros(M, H, device=x.device, dtype=torch.float32)
    for m in range(M):
        for j in range(topk):
            e = int(topk_ids[m, j].item()); wt = float(topk_w[m, j].item())
            gate_up = x[m].float() @ w13_dq[e].T               # [2I]
            h = F.silu(gate_up[:I]) * gate_up[I:]              # [I]
            out[m] += wt * (h @ w2f[e].T)                      # [H]
    return out

def run_case(E, I, H, M, topk, seed=0):
    layer, w13_q, s13, w2 = build_layer(E, I, H, seed)
    g = torch.Generator(device=dev).manual_seed(seed + 100)
    x = torch.randn(M, H, generator=g, device=dev, dtype=torch.float16) * 0.1
    topk_ids = torch.stack([torch.randperm(E, generator=g, device=dev)[:topk] for _ in range(M)]).to(torch.int64)
    topk_w = torch.rand(M, topk, generator=g, device=dev, dtype=torch.float16) * 0.5 + 0.1

    mixed = _v100_ct_mixed_moe_routed(layer, x, topk_w, topk_ids).float()
    ref = ref_moe(x, topk_w, topk_ids, w13_q, s13, w2, I)
    l2 = ((mixed - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    cos = F.cosine_similarity(mixed.flatten(), ref.flatten(), dim=0).item()
    ok = (cos > 0.9999 and l2 < 2e-2)
    print(f"[{'PASS' if ok else 'FAIL'}] E={E} I={I} H={H} M={M} topk={topk} | "
          f"L2rel={l2:.4f} cos={cos:.5f}")
    return ok

print("-- mixed routed MoE (w13 FP8 + w2 FP16) vs reference FP16 MoE --")
print("   (VLLM_V100_CT_MOE_W13_CHUNK forced small so R>chunk cases exercise the")
print("    routed-row chunking that fixes the gridDim.y>65535 long-context bug)")
allok = True
allok &= run_case(E=8,  I=176, H=4096, M=1,  topk=2)   # decode-like, GLM shapes (2I=352 partial-N)
allok &= run_case(E=8,  I=176, H=4096, M=4,  topk=2)
allok &= run_case(E=8,  I=176, H=4096, M=16, topk=4)
allok &= run_case(E=16, I=176, H=4096, M=8,  topk=2)
allok &= run_case(E=8,  I=256, H=4096, M=4,  topk=2)   # 2I=512 aligned (sanity)
allok &= run_case(E=8,  I=176, H=4096, M=100, topk=4)  # R=400 >> chunk(64) -> multi-chunk w13 path
print("\nRESULT:", "ALL PASS — mixed routed MoE matches reference (route/w13/act/w2/scatter correct)"
      if allok else "FAIL — integration bug in the mixed forward (see rows above)")
PY
rc=$?
echo "(exit $rc)"; exit $rc
