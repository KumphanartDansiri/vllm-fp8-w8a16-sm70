"""FP8 W8A16 quantization kernel for V100 (sm_70) integrated with vLLM 0.18.

Public surface:
  - load_kernel():     JIT-compile and return the CUDA extension
  - FP8W8A16Linear:    nn.Module drop-in for FP8 block-quantized Linear layers

The vLLM monkey-patch entry point lives in `fp8_w8a16_sm70.vllm_serve`. Run it
directly as a script to start a vLLM OpenAI-compatible server with FP8 support
on V100.
"""
from .ext_loader import load_kernel
from .module import FP8W8A16Linear

__all__ = ["load_kernel", "FP8W8A16Linear"]
__version__ = "0.6.0"
