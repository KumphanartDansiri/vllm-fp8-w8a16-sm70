"""JIT-compile the FP8 W8A16 sm_70 CUDA kernel.

All test/bench/tool scripts call `load_kernel()` instead of duplicating the
torch.utils.cpp_extension.load(...) recipe with hard-coded kernel paths. This
centralizes the build flags and the kernel-source location, so when either
moves the rest of the codebase doesn't need to change.

Each caller passes its own `name` so the extension is cached in a distinct
directory under torch_extensions/ — useful when iterating on the kernel and
running multiple processes that should each pick up the latest source.
"""
from pathlib import Path

from torch.utils.cpp_extension import load


_KERNEL_DIR = Path(__file__).resolve().parent
_KERNEL_SRC = _KERNEL_DIR / "fp8_dequant.cu"


def load_kernel(name: str = "fp8_dequant_ext", verbose: bool = False):
    """JIT-compile fp8_dequant.cu for sm_70 (V100) and return the loaded module.

    The compiled .so is cached under ~/.cache/torch_extensions/<name>/ (or
    wherever torch's extension cache resolves to).  Re-import is free unless
    the source file has changed.
    """
    return load(
        name=name,
        sources=[str(_KERNEL_SRC)],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_70,code=sm_70",
            "--use_fast_math",
        ],
        extra_cflags=["-O3"],
        verbose=verbose,
    )


def kernel_source_path() -> Path:
    """Absolute path to fp8_dequant.cu. Useful for diagnostics / scripts that
    need to reference the .cu directly (e.g., dev_sanity.py)."""
    return _KERNEL_SRC
