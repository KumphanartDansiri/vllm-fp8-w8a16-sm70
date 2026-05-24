"""
FP8W8A16Linear: a drop-in PyTorch nn.Module that does FP8 W8A16 forward on V100.

Wraps our custom CUDA kernel so it can be used like nn.Linear inside any model:

    layer = FP8W8A16Linear.from_safetensors(st_path, "model.x.weight",
                                            "model.x.weight_scale_inv")
    y = layer(x)                              # x: [..., K] fp16   y: [..., N] fp16

Loads on construction:
    weight_fp8     : [N, K]  uint8 (raw FP8 E4M3-FN bytes)
    weight_scale   : [Nb, Kb] fp16 (cast from BF16 if needed; ~lossless in practice)

Forward:
    1. Reshape input to (M, K), cast to fp16 if needed.
    2. Call fused FP8 W8A16 GEMM kernel (CUDA cores).
    3. Reshape output back to (..., N).
    4. Add bias if present (in fp16).
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

try:
    from safetensors import safe_open
    _HAVE_SAFETENSORS = True
except ImportError:
    _HAVE_SAFETENSORS = False


class FP8W8A16Linear(nn.Module):
    def __init__(
        self,
        ext,                              # the loaded torch extension (provides fp8_w8a16_gemm)
        weight_fp8: torch.Tensor,         # [N, K] uint8 raw FP8 bytes
        weight_scale: torch.Tensor,       # [Nb, Kb] fp16
        block_h: int = 128,
        block_w: int = 128,
        bias: Optional[torch.Tensor] = None,  # [N] fp16, or None
    ):
        super().__init__()
        assert weight_fp8.dtype == torch.uint8, "weight_fp8 must be uint8 (FP8 reinterpreted)"
        assert weight_scale.dtype == torch.float16, "weight_scale must be fp16"
        assert weight_fp8.ndim == 2, "weight_fp8 must be 2D [N, K]"
        N, K = weight_fp8.shape
        Nb = (N + block_h - 1) // block_h
        Kb = (K + block_w - 1) // block_w
        assert weight_scale.shape == (Nb, Kb), \
            f"weight_scale shape {weight_scale.shape} != expected ({Nb}, {Kb})"

        self.ext = ext
        self.N, self.K = N, K
        self.block_h, self.block_w = block_h, block_w

        # Buffers — not trainable. Flatten so the kernel sees contiguous 1D layouts.
        self.register_buffer("weight_fp8",   weight_fp8.contiguous().reshape(-1))
        self.register_buffer("weight_scale", weight_scale.contiguous().reshape(-1))
        if bias is not None:
            assert bias.shape == (N,)
            self.register_buffer("bias", bias.to(torch.float16).contiguous())
        else:
            self.bias = None

    @classmethod
    def from_safetensors(
        cls,
        ext,
        st_path,
        weight_key: str,
        scale_key: str,
        block_h: int = 128,
        block_w: int = 128,
        bias_key: Optional[str] = None,
        device: str = "cuda",
    ):
        if not _HAVE_SAFETENSORS:
            raise RuntimeError("safetensors not installed")
        with safe_open(Path(st_path), framework="pt") as f:
            W = f.get_tensor(weight_key)           # float8_e4m3fn
            S = f.get_tensor(scale_key)            # bf16 typically
            B = f.get_tensor(bias_key) if bias_key else None
        # Convert to the dtypes our kernel expects.
        W_u8 = W.view(torch.uint8).to(device).contiguous()
        S_f16 = S.to(torch.float16).to(device).contiguous()
        bias = None
        if B is not None:
            bias = B.to(torch.float16).to(device).contiguous()
        return cls(ext, W_u8, S_f16, block_h, block_w, bias)

    # Dispatch thresholds tuned from the A.1/A.2/A.3 benchmarks on the 4B
    # model's [2560, 9216] down_proj layer:
    #   M <=  4 : A.3 K_SPLIT=8   (low-M, K-axis CTA splitting wins big)
    #   M ==  8 : A.3 K_SPLIT=4   (transition)
    #   M <  64 : A.1             (single-row CTAs, enough already)
    #   M >= 64 : A.2             (M-tiling amortizes W reads)
    # Different layer shapes / different N may shift these crossovers
    # slightly, but the pattern (small M wants more CTAs, large M wants
    # less W traffic) holds.
    DISPATCH_M_A3_K8  = 4    # M <= this -> A.3 with K_SPLIT=8
    DISPATCH_M_A3_K4  = 8    # M <= this -> A.3 with K_SPLIT=4
    DISPATCH_M_A2     = 64   # M >= this -> A.2 (or WMMA when shapes permit)
    # WMMA POC tile constants (must match fp8_dequant.cu)
    WMMA_TILE_M       = 64
    WMMA_TILE_N       = 64
    WMMA_TILE_K       = 16

    def _k_split_ok(self, k_split: int) -> bool:
        """A.3 needs K % (k_split * block_w) == 0 for scale alignment."""
        return (self.K % (k_split * self.block_w)) == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., K] -> out: [..., N]. Cast to fp16 internally if needed.
        Dispatches across A.1 / A.2 / A.3 based on batch size to land on the
        fastest kernel variant for the workload."""
        assert x.shape[-1] == self.K, \
            f"input last dim {x.shape[-1]} != K={self.K}"
        orig_shape = x.shape
        x2d = x.reshape(-1, self.K)
        if x2d.dtype != torch.float16:
            x2d = x2d.to(torch.float16)
        x2d = x2d.contiguous()

        M = x2d.size(0)
        ext = self.ext

        # Resolve which kernel to call. Fall back gracefully if optional A.x
        # bindings aren't present (older build) or if K isn't compatible with
        # the desired K_SPLIT.
        have_a1   = hasattr(ext, "fp8_w8a16_gemm_a1")
        have_a2   = hasattr(ext, "fp8_w8a16_gemm_a2")
        have_a3   = hasattr(ext, "fp8_w8a16_gemm_a3")
        have_wmma = hasattr(ext, "fp8_w8a16_gemm_wmma_poc")

        # WMMA POC constraints: M%64==0, N%64==0, K%16==0, block 128×128.
        wmma_layer_ok = (
            have_wmma
            and (self.N % self.WMMA_TILE_N) == 0
            and (self.K % self.WMMA_TILE_K) == 0
            and self.block_h == 128 and self.block_w == 128
        )

        if have_a3 and M <= self.DISPATCH_M_A3_K8 and self._k_split_ok(8):
            out = ext.fp8_w8a16_gemm_a3(
                x2d, self.weight_fp8, self.weight_scale,
                self.N, self.K, self.block_h, self.block_w, 8)
        elif have_a3 and M <= self.DISPATCH_M_A3_K4 and self._k_split_ok(4):
            out = ext.fp8_w8a16_gemm_a3(
                x2d, self.weight_fp8, self.weight_scale,
                self.N, self.K, self.block_h, self.block_w, 4)
        elif wmma_layer_ok and M >= self.WMMA_TILE_M:
            # WMMA + A.2 tail when M is not 64-aligned.
            M_aligned = (M // self.WMMA_TILE_M) * self.WMMA_TILE_M
            M_tail    = M - M_aligned
            x_main = x2d[:M_aligned].contiguous()
            out_main = ext.fp8_w8a16_gemm_wmma_poc(
                x_main, self.weight_fp8, self.weight_scale,
                self.N, self.K, self.block_h, self.block_w)
            if M_tail > 0:
                x_tail = x2d[M_aligned:].contiguous()
                out_tail = ext.fp8_w8a16_gemm_a2(
                    x_tail, self.weight_fp8, self.weight_scale,
                    self.N, self.K, self.block_h, self.block_w)
                out = torch.cat([out_main, out_tail], dim=0)
            else:
                out = out_main
        elif have_a2 and M >= self.DISPATCH_M_A2:
            out = ext.fp8_w8a16_gemm_a2(
                x2d, self.weight_fp8, self.weight_scale,
                self.N, self.K, self.block_h, self.block_w)
        elif have_a1:
            out = ext.fp8_w8a16_gemm_a1(
                x2d, self.weight_fp8, self.weight_scale,
                self.N, self.K, self.block_h, self.block_w)
        else:
            out = ext.fp8_w8a16_gemm(
                x2d, self.weight_fp8, self.weight_scale,
                self.N, self.K, self.block_h, self.block_w)

        out = out.reshape(orig_shape[:-1] + (self.N,))
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (f"N={self.N}, K={self.K}, "
                f"block=[{self.block_h},{self.block_w}], "
                f"bias={self.bias is not None}")
