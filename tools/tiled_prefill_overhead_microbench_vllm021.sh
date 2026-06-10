#!/usr/bin/env bash
# Phase 4 Stage 1 overhead breakdown (Codex caution: "measure grouping/sort overhead
# separately so we don't hide a new bottleneck"). Stage 1 cut 26k TTFT 169s->74s
# (2.3x), below the ~7x MoE-kernel ceiling — this bench shows WHERE the gap is:
# grouping (argsort/unique/cumsum) vs gather/scatter vs the per-expert a2(w13) loop
# vs the per-expert cuBLAS(w2) loop, at GLM-Air 26k-prefill R=209584.
#
# Usage: ./tools/tiled_prefill_overhead_microbench_vllm021.sh
# Env: IMAGE GPU CACHE_TAG R(default 209584)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"
IMAGE="${IMAGE:-vllm-v100:vllm021-cu126}"; GPU="${GPU:-0}"; CACHE_TAG="${CACHE_TAG:-021}"
R="${R:-209584}"
for s in torchext triton torch inductor; do mkdir -p "$HOME/.cache/vllm-v100-${CACHE_TAG}-$s"; done
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE missing"; exit 1; }

docker run --rm -i --name tiled_prefill_overhead --gpus "\"device=$GPU\"" \
    -v "$PROJECT_ROOT":/work -w /work -e PYTHONPATH=/work/src -e R_ROWS="$R" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torchext:/root/.cache/torch_extensions" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-triton:/root/.triton" \
    -v "$HOME/.cache/vllm-v100-${CACHE_TAG}-torch:/root/.cache/torch" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$IMAGE" python3 - <<'PY'
import os, torch
from fp8_w8a16_sm70.ext_loader import load_kernel
ext = load_kernel(name="fp8_dequant_ext_vllm")
dev="cuda"; torch.backends.cuda.matmul.allow_tf32=False
R=int(os.environ.get("R_ROWS","209584")); E=128; I=176; H=4096; N13=2*I
print(f"GLM-Air 26k-prefill MoE, ONE LAYER: R={R} routed rows, E={E}, w13[{N13},{H}] FP8, w2[{H},{I}] FP16")
g=torch.Generator(device=dev).manual_seed(0)
s13=(torch.rand(E,N13,1,generator=g,device=dev,dtype=torch.float16)*0.02+0.01)
w13u8=(torch.randn(E,N13,H,generator=g,device=dev,dtype=torch.float16)*0.1).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
s13b=s13.expand(E,N13,H//128).contiguous()
w2=(torch.randn(E,H,I,generator=g,device=dev,dtype=torch.float16)*0.1).contiguous()
route_x=torch.randn(R,H,generator=g,device=dev,dtype=torch.float16)*0.1
expert_ids=torch.randint(0,E,(R,),generator=g,device=dev,dtype=torch.int64)

def t(fn,it=10,wu=3):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it

# ---- component timing of the tiled path ----
order=torch.argsort(expert_ids); eids_s=expert_ids.index_select(0,order)
rx_s=route_x.index_select(0,order).contiguous()
uniq,counts=torch.unique_consecutive(eids_s,return_counts=True)
starts=torch.cumsum(counts,0)-counts
ul,sl,cl=uniq.tolist(),starts.tolist(),counts.tolist()

def grouping():
    o=torch.argsort(expert_ids); es=expert_ids.index_select(0,o)
    rs=route_x.index_select(0,o).contiguous()
    u,c=torch.unique_consecutive(es,return_counts=True); st=torch.cumsum(c,0)-c
    return o,rs,u,st,c
def w13_loop():
    out=torch.empty(R,N13,device=dev,dtype=torch.float16)
    for e,st,cnt in zip(ul,sl,cl):
        out[st:st+cnt]=ext.fp8_w8a16_gemm_a2(rx_s[st:st+cnt].contiguous(),w13u8[e].reshape(-1),s13b[e].reshape(-1),N13,H,1,128)
    return out
w13o=w13_loop()
hid_s=(torch.nn.functional.silu(w13o[:,:I])*w13o[:,I:]).contiguous()
def w2_loop():
    out=torch.empty(R,H,device=dev,dtype=torch.float16)
    for e,st,cnt in zip(ul,sl,cl):
        out[st:st+cnt]=hid_s[st:st+cnt]@w2[e].T
    return out
def scatter():
    eo=torch.empty(R,H,device=dev,dtype=torch.float16); eo.index_copy_(0,order,hid_s[:, :H] if H<=hid_s.size(1) else torch.empty(R,H,device=dev,dtype=torch.float16)); return eo

m_group=t(lambda: grouping(),it=10,wu=3)
m_w13=t(w13_loop,it=5,wu=2)
m_act=t(lambda:(torch.nn.functional.silu(w13o[:,:I])*w13o[:,I:]).contiguous(),it=10,wu=3)
m_w2=t(w2_loop,it=5,wu=2)
eo=torch.empty(R,H,device=dev,dtype=torch.float16); src=torch.randn(R,H,device=dev,dtype=torch.float16)
m_scat=t(lambda: eo.index_copy_(0,order,src),it=20,wu=5)  # [R,H] scatter (unsort) only
tiled_total=m_group+m_w13+m_act+m_w2+m_scat

# ---- per-row grouped path (current decode kernel) at same R, chunked ----
def grouped_w13():
    parts=[]
    for i in range(0,R,60000):
        j=min(i+60000,R); parts.append(ext.fp8_w8a16_grouped_routed_gemm_a3(route_x[i:j].contiguous(),expert_ids[i:j].contiguous(),w13u8,s13b,N13,H,1,128,8))
    return torch.cat(parts,0)
def grouped_w2():
    parts=[]
    for i in range(0,R,60000):
        j=min(i+60000,R); parts.append(ext.fp16_grouped_routed_gemm(hid_s[i:j].contiguous(),expert_ids[i:j].contiguous(),w2,1))
    return torch.cat(parts,0)
m_gw13=t(grouped_w13,it=3,wu=1); m_gw2=t(grouped_w2,it=3,wu=1)
grouped_total=m_gw13+m_gw2

print(f"\n== TILED path components (ms/layer) ==")
print(f"  grouping(sort+gather+unique+cumsum) : {m_group:8.2f}")
print(f"  w13 per-expert a2 loop              : {m_w13:8.2f}")
print(f"  activation                          : {m_act:8.2f}")
print(f"  w2 per-expert cuBLAS loop           : {m_w2:8.2f}")
print(f"  scatter(index_copy)                 : {m_scat:8.2f}")
print(f"  TILED TOTAL                         : {tiled_total:8.2f}")
print(f"\n== PER-ROW grouped path (current) (ms/layer) ==")
print(f"  grouped w13 : {m_gw13:8.2f}   grouped w2 : {m_gw2:8.2f}   TOTAL : {grouped_total:8.2f}")
print(f"\n== speedup (per-row / tiled) : {grouped_total/tiled_total:.2f}x  (x45 layers => tiled {tiled_total*45/1000:.1f}s vs grouped {grouped_total*45/1000:.1f}s) ==")
print("   overhead share of tiled = grouping+scatter =", f"{(m_group+m_scat)/tiled_total*100:.0f}%")
PY
rc=$?; echo "(exit $rc)"; exit $rc
