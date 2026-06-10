#!/usr/bin/env python3
"""NCU target: GLM-Air grouped MoE w13 decode kernel (fp8_w8a16_grouped_routed_gemm_a3).
Brackets ONE kernel call with cudaProfilerStart/Stop. Run under
`ncu --profile-from-start off` to measure sectors/request + DRAM% — the
coalescing falsifiable bar (A.3 dense was 25.3 sectors / 17.8% DRAM)."""
import torch
from fp8_w8a16_sm70.ext_loader import load_kernel
ext = load_kernel(name="fp8_dequant_ext_vllm")
dev = "cuda"
E, R, N13, K13 = 128, 8, 352, 4096          # GLM-Air TP=8 w13 decode, topk=8
g = torch.Generator(device=dev).manual_seed(0)
s13 = (torch.rand(E, N13, 1, generator=g, device=dev, dtype=torch.float16) * 0.02 + 0.01
       ).expand(E, N13, K13 // 128).contiguous()
w13 = (torch.randn(E, N13, K13, generator=g, device=dev, dtype=torch.float16) * 0.1
       ).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
x13 = torch.randn(R, K13, generator=g, device=dev, dtype=torch.float16) * 0.1
eids = torch.randint(0, E, (R,), generator=g, device=dev, dtype=torch.int64)
for _ in range(50):
    ext.fp8_w8a16_grouped_routed_gemm_a3(x13, eids, w13, s13, N13, K13, 1, 128, 8)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
ext.fp8_w8a16_grouped_routed_gemm_a3(x13, eids, w13, s13, N13, K13, 1, 128, 8)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
