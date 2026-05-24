"""
serve_fp8_v100.py
─────────────────
Monkey-patches vLLM 0.18.0's FP8 dispatch so it loads + serves a DeepSeek-style
block-FP8 W8A16 model on Volta (sm_70 / V100) GPUs.

Without these patches, vLLM rejects sm_70 for FP8 (min capability 75) and
takes a Marlin path that requires sm_75+ instructions. We:

    1. Lower Fp8Config.get_min_capability  ->  70
    2. Force use_marlin = False on V100 (no Marlin layout transformation)
    3. Replace Fp8LinearMethod.apply() with our V100 W8A16 kernel
       (auto-dispatches A.1 / A.2 / A.3 by M, same logic as FP8W8A16Linear)

Then we hand off to vLLM's existing CLI. From the user's view, this is just
`python serve_fp8_v100.py --model ... --quantization fp8 ...` and vLLM
behaves normally, except FP8 actually works on V100.

This script touches NO vLLM source files. Patches are applied at import time.
Reverting = stop using this wrapper.

Compatible with: vllm 0.18.0, torch 2.10.0+cu128 (matches AIAGENT_ENV.md).

Usage:
    /home/aiagent/vllm-env/bin/python serve_fp8_v100.py \\
        --model /mnt/models/Qwen3.5-4B-FP8 \\
        --quantization fp8 \\
        --dtype float16 \\
        --enforce-eager \\
        --attention-backend TRITON_ATTN \\
        --max-num-seqs 1 \\
        --tensor-parallel-size 1
"""
import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


# ─── Compile our FP8 W8A16 kernel ────────────────────────────────────────────
HERE = Path(__file__).resolve().parent

print("[serve_fp8_v100] Compiling FP8 W8A16 kernel for sm_70 ...", flush=True)
_ext = load(
    name="fp8_dequant_ext_vllm",
    sources=[str(HERE / "fp8_dequant.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_70,code=sm_70",
        "--use_fast_math",
    ],
    extra_cflags=["-O3"],
    verbose=False,
)
print("[serve_fp8_v100] Kernel compiled OK.", flush=True)


def _is_volta() -> bool:
    """True iff the first visible GPU is sm_70 (V100). vLLM assumes all visible
    GPUs are homogeneous; we follow the same convention."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return (major, minor) == (7, 0)


# ─── Dispatch thresholds (mirrors FP8W8A16Linear) ───────────────────────────
# M <=  4 : A.3 K_SPLIT=8 (K-axis CTA splitting, decode)
# M <=  8 : A.3 K_SPLIT=4 (low-M transition)
# M < 64  : A.1 (vectorized, one row per CTA)
# M >= 64 : WMMA (Phase A.4 POC, tensor cores) for the 64-aligned region;
#           tail (M % 64) falls back to A.2.
_DISPATCH_M_A3_K8 = 4
_DISPATCH_M_A3_K4 = 8
_DISPATCH_M_A2    = 64

# WMMA POC tile sizes (must match constants in fp8_dequant.cu)
_WMMA_TILE_M = 64
_WMMA_TILE_N = 64
_WMMA_TILE_K = 16
_HAS_WMMA    = hasattr(_ext, "fp8_w8a16_gemm_wmma_poc")
# Dispatch threshold for WMMA. Default = tile M (64). Set very high (e.g.,
# FP8_WMMA_MIN_M=99999) to fully disable WMMA for A/B comparison.
_WMMA_MIN_M  = int(os.environ.get("FP8_WMMA_MIN_M", str(_WMMA_TILE_M)))

# Per-process counters for per-variant call rate observability. Each TP
# worker tracks its own counts; periodic summary printed every COUNTER_LOG_EVERY
# calls on rank 0.
_VARIANT_COUNTS = {"A.3 k=8": 0, "A.3 k=4": 0, "A.1": 0, "A.2": 0,
                   "WMMA": 0, "WMMA+A.2(tail)": 0}
_VARIANT_COUNTER_LOG_EVERY = int(os.environ.get("FP8_WMMA_COUNTER_LOG_EVERY", "1000"))
_VARIANT_TOTAL = 0


# Per-(prefix, rank) state tracking for the apply-stats instrumentation.
# Each set holds keys that have already triggered a particular log policy,
# so we don't spam the log with thousands of repetitive lines per layer.
_APPLY_LOGGED_FIRST  = set()   # any call (warmup or inference) — first per layer
_APPLY_LOGGED_DECODE = set()   # first M==1 (decode) call per layer
_APPLY_LOGGED_WARN   = set()   # first WARN-level call per layer
_APPLY_LOGGED_BAD    = set()   # first BAD-level call per layer
_APPLY_CALL_COUNT    = {}      # (prefix, rank) -> call index

# Thresholds (overridable via env). FP16 saturates at ~65504; we want to
# catch trends well before that. Default WARN at 1k, BAD at 10k.
_APPLY_WARN = float(os.environ.get("VLLM_V100_FP8_APPLY_WARN", "1000"))
_APPLY_BAD  = float(os.environ.get("VLLM_V100_FP8_APPLY_MAG",  "10000"))


def _maybe_log_apply_stats(layer, x_in, out_pre, out_post, variant):
    """Log per-(layer, rank) activation/output statistics from _our_apply.

    Enabled by VLLM_V100_FP8_DEBUG_APPLY=on. Logs on:
      - FIRST   : first call for (prefix, rank) — baseline.
      - DECODE  : first M==1 call for (prefix, rank) — decode may diverge
                  from prefill even if prefill looks clean.
      - WARN    : first call where out_pre abs_max > VLLM_V100_FP8_APPLY_WARN
                  (default 1000) — magnitude trending toward FP16 saturation.
      - BAD     : first call where out_pre has NaN/Inf OR abs_max >
                  VLLM_V100_FP8_APPLY_MAG (default 10000).
      - CAST    : out_pre was fully finite but out_post (after dtype cast
                  back to x.dtype) became non-finite — caught at the cast.

    Each event fires AT MOST once per (prefix, rank). Use to locate the
    first layer where outputs explode during real TP=4 inference.
    """
    if os.environ.get("VLLM_V100_FP8_DEBUG_APPLY", "off").lower() in (
            "0", "off", "false", ""):
        return

    prefix = getattr(layer, "prefix", "<?>")
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
    except Exception:
        rank = -1
    key = (prefix, rank)
    call_idx = _APPLY_CALL_COUNT.get(key, 0)
    _APPLY_CALL_COUNT[key] = call_idx + 1

    # x is already reshaped to [-1, K] inside _our_apply. Use shape[0] as M.
    M = int(x_in.shape[0])

    # First gate: is this a candidate for logging at all?
    is_first        = key not in _APPLY_LOGGED_FIRST
    is_first_decode = (M == 1) and (key not in _APPLY_LOGGED_DECODE)

    # Compute fast probe on out_pre to decide WARN/BAD.
    out_total = out_pre.numel()
    out_finite_mask = torch.isfinite(out_pre)
    out_finite = int(out_finite_mask.sum())
    if out_finite == 0:
        out_abs_max_pre = float("inf")
    else:
        out_abs_max_pre = float(out_pre[out_finite_mask].abs().max())
    is_bad  = (out_finite < out_total) or (out_abs_max_pre > _APPLY_BAD)
    is_warn = (not is_bad) and (out_abs_max_pre > _APPLY_WARN)
    is_warn_first = is_warn and (key not in _APPLY_LOGGED_WARN)
    is_bad_first  = is_bad  and (key not in _APPLY_LOGGED_BAD)

    # Cast divergence: pre-cast clean but post-cast became non-finite.
    cast_diverged = False
    if out_finite == out_total:
        post_finite_count = int(torch.isfinite(out_post).sum())
        cast_diverged = post_finite_count < out_total
    else:
        post_finite_count = -1

    if not (is_first or is_first_decode or is_warn_first or
            is_bad_first or cast_diverged):
        return

    # Compute the remaining stats only now that we're committed to logging.
    nan_count = int(torch.isnan(out_pre).sum())
    inf_count = int(torch.isinf(out_pre).sum())
    if out_finite > 0:
        out_abs_pre = out_pre[out_finite_mask].abs()
        out_mean_abs_pre = float(out_abs_pre.mean())
    else:
        out_mean_abs_pre = float("nan")

    x_total = x_in.numel()
    x_finite_mask = torch.isfinite(x_in)
    x_finite = int(x_finite_mask.sum())
    if x_finite > 0:
        x_abs = x_in[x_finite_mask].abs()
        x_abs_max = float(x_abs.max())
        x_mean_abs = float(x_abs.mean())
    else:
        x_abs_max = float("nan")
        x_mean_abs = float("nan")

    post_finite_mask = torch.isfinite(out_post)
    post_finite_count = int(post_finite_mask.sum())
    post_total = out_post.numel()
    if post_finite_count > 0:
        post_selected = out_post.masked_select(post_finite_mask)
        post_abs_max = float(post_selected.abs().max())
    else:
        post_abs_max = float("nan")

    tags = []
    if is_first:        tags.append("FIRST");  _APPLY_LOGGED_FIRST.add(key)
    if is_first_decode: tags.append("DECODE"); _APPLY_LOGGED_DECODE.add(key)
    if is_warn_first:   tags.append("WARN");   _APPLY_LOGGED_WARN.add(key)
    if is_bad_first:    tags.append("BAD");    _APPLY_LOGGED_BAD.add(key)
    if cast_diverged:   tags.append("CAST")
    tag = "+".join(tags)

    print(
        f"[V100-FP8-APPLY {tag} rank={rank} call={call_idx} M={M} variant={variant}] {prefix}\n"
        f"  x:        dtype={x_in.dtype} shape={tuple(x_in.shape)} "
        f"finite={x_finite}/{x_total} abs_max={x_abs_max:.3e} mean_abs={x_mean_abs:.3e}\n"
        f"  out_pre:  dtype={out_pre.dtype} shape={tuple(out_pre.shape)} "
        f"finite={out_finite}/{out_total} nan={nan_count} inf={inf_count} "
        f"abs_max={out_abs_max_pre:.3e} mean_abs={out_mean_abs_pre:.3e}\n"
        f"  out_post: dtype={out_post.dtype} "
        f"finite={post_finite_count}/{post_total} abs_max={post_abs_max:.3e}",
        flush=True,
    )


def _our_apply(self, layer, x, bias=None):
    """Replacement for Fp8LinearMethod.apply that uses our V100 W8A16 kernel.

    Expects (after patched process_weights_after_loading):
      - layer.weight          : torch.float8_e4m3fn, shape [N, K]
      - layer.weight_scale_inv: float16, shape [N/block_h, K/block_w] (we cast
                                from the model's BF16 once at load time)
      - layer.weight_block_size: list[int] of length 2, e.g. [128, 128]
      - layer.input_size_per_partition  == K
      - layer.output_size_per_partition == N
    """
    N = layer.output_size_per_partition
    K = layer.input_size_per_partition
    block_h, block_w = layer.weight_block_size

    # Flatten input to [M, K] in FP16. vLLM may pass BF16 if --dtype bfloat16,
    # in which case we cast — slight cost, acceptable on V100 (it uses FP16
    # tensor cores not BF16 anyway, so model dtype is usually FP16).
    orig_shape = x.shape
    x2d = x.reshape(-1, K)
    if x2d.dtype != torch.float16:
        x2d = x2d.to(torch.float16)
    x2d = x2d.contiguous()
    M = x2d.size(0)

    # Raw uint8 view of FP8 bytes (kernel ABI) and contiguous flattened layout.
    weight_u8 = layer.weight.view(torch.uint8).reshape(-1).contiguous()
    scales    = layer.weight_scale_inv.reshape(-1).contiguous()

    # Dispatch: A.3 (k_split=8 or 4) for low M, A.1 for middle, WMMA + tail
    # for high M when shape constraints permit.
    # A.3 requires K % (k_split * block_w) == 0.
    # WMMA POC requires M%64==0, N%64==0, K%16==0, block_h==block_w==128.
    def k_split_ok(k_split):
        return (K % (k_split * block_w)) == 0

    wmma_layer_ok = (
        _HAS_WMMA
        and (N % _WMMA_TILE_N) == 0
        and (K % _WMMA_TILE_K) == 0
        and block_h == 128 and block_w == 128
    )

    if M <= _DISPATCH_M_A3_K8 and k_split_ok(8):
        out = _ext.fp8_w8a16_gemm_a3(
            x2d, weight_u8, scales, N, K, block_h, block_w, 8)
        variant = "A.3 k=8"
    elif M <= _DISPATCH_M_A3_K4 and k_split_ok(4):
        out = _ext.fp8_w8a16_gemm_a3(
            x2d, weight_u8, scales, N, K, block_h, block_w, 4)
        variant = "A.3 k=4"
    elif wmma_layer_ok and M >= _WMMA_MIN_M:
        # WMMA path with M-tail fallback to A.2 when M is not 64-aligned.
        M_aligned = (M // _WMMA_TILE_M) * _WMMA_TILE_M
        M_tail    = M - M_aligned
        x_main = x2d[:M_aligned].contiguous()
        out_main = _ext.fp8_w8a16_gemm_wmma_poc(
            x_main, weight_u8, scales, N, K, block_h, block_w)
        if M_tail > 0:
            x_tail = x2d[M_aligned:].contiguous()
            out_tail = _ext.fp8_w8a16_gemm_a2(
                x_tail, weight_u8, scales, N, K, block_h, block_w)
            out = torch.cat([out_main, out_tail], dim=0)
            variant = "WMMA+A.2(tail)"
        else:
            out = out_main
            variant = "WMMA"
    elif M >= _DISPATCH_M_A2:
        # Either WMMA isn't applicable for this layer or M < tile size.
        # Should be rare given the WMMA branch above, but keep A.2 as fallback.
        out = _ext.fp8_w8a16_gemm_a2(
            x2d, weight_u8, scales, N, K, block_h, block_w)
        variant = "A.2"
    else:
        out = _ext.fp8_w8a16_gemm_a1(
            x2d, weight_u8, scales, N, K, block_h, block_w)
        variant = "A.1"

    # Variant counter for observability — periodic per-process summary.
    global _VARIANT_TOTAL
    _VARIANT_COUNTS[variant] = _VARIANT_COUNTS.get(variant, 0) + 1
    _VARIANT_TOTAL += 1
    if _VARIANT_TOTAL % _VARIANT_COUNTER_LOG_EVERY == 0:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            tot = _VARIANT_TOTAL
            parts = [f"{k}={v} ({100.0*v/tot:.0f}%)"
                     for k, v in _VARIANT_COUNTS.items() if v]
            print(f"[serve_fp8_v100 pid={os.getpid()}] kernel variant counts after "
                  f"{tot} calls: {', '.join(parts)}", flush=True)

    # Reshape back to whatever rank the input had ([B, S, N] → reshape to that).
    out = out.reshape(orig_shape[:-1] + (N,))

    if bias is not None:
        out = out + bias.to(out.dtype)

    # Snapshot pre-cast output so the apply-stats logger can detect a cast
    # that turns finite values into non-finite ones (FP16→BF16 narrowing,
    # for example).
    out_pre_cast = out

    # vLLM's caller expects the original input dtype back — match it.
    if out.dtype != x.dtype:
        out = out.to(x.dtype)

    _maybe_log_apply_stats(layer, x2d, out_pre_cast, out, variant)
    return out


def _our_process_weights_after_loading(self, layer):
    """Replacement for Fp8LinearMethod.process_weights_after_loading.

    On V100:
      - For block_quant: keep weight in original [N, K] FP8 layout.
                          Cast scale BF16/FP32 → FP16 once.
      - SKIP prepare_fp8_layer_for_marlin (uses sm_75+ instructions)
      - SKIP cutlass/DeepGEMM post-processing.

    Falls back to the original implementation if this method's preconditions
    aren't met (e.g. non-block quant, which Qwen3.5-4B-FP8 doesn't use).
    """
    global _PWAL_BANNERED
    if not _PWAL_BANNERED:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
        except Exception:
            rank = -1
        print(
            f"[V100-FP8-BANNER rank={rank} pid={os.getpid()}] "
            f"patched PWAL fired (block_quant={getattr(self, 'block_quant', '?')}, "
            f"layer.prefix={getattr(layer, 'prefix', '<no-prefix>')})",
            flush=True,
        )
        _PWAL_BANNERED = True

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        process_fp8_weight_block_strategy,
    )
    from vllm.model_executor.utils import replace_parameter

    if not self.block_quant:
        # Non-block FP8 (per-tensor / per-channel) not in our scope — fall back
        # to the original method. This won't engage our kernel; if the model
        # actually triggers this path, it'd fail elsewhere on V100.
        return _ORIG_process_weights_after_loading(self, layer)

    # Same first step as upstream — pads weight, normalizes for FNUZ (AMD).
    weight, weight_scale_inv = process_fp8_weight_block_strategy(
        layer.weight, layer.weight_scale_inv
    )

    # Cast scale to FP16 once (model usually stores BF16). Lossless in practice
    # for DeepSeek-style scales; saves a per-call cast.
    if weight_scale_inv.dtype != torch.float16:
        weight_scale_inv = weight_scale_inv.to(torch.float16)
    # Ensure contiguous so .view() and reinterpret in the kernel are safe.
    weight = weight.contiguous()
    weight_scale_inv = weight_scale_inv.contiguous()

    replace_parameter(layer, "weight", weight.data)
    replace_parameter(layer, "weight_scale_inv", weight_scale_inv.data)
    layer.input_scale = None
    # Note: NO prepare_fp8_layer_for_marlin call — that's the whole point.

    _maybe_log_block_shapes(self, layer, weight, weight_scale_inv)


def _maybe_log_block_shapes(self, layer, weight, scale):
    """Per-layer block-shape diagnostic for the TP>1 numerical bug.

    Prints one line per Linear with: prefix, sharded weight/scale shapes,
    config block_size, the *effective* block size derived from
    weight.shape / scale.shape, and a mismatch flag. The expectation is
    [128, 128] everywhere; any layer where effective block != 128 is a
    candidate source of the "!!!!!" corruption at TP=4.

    Controlled by VLLM_V100_FP8_DEBUG_SHAPES:
        unset / "0" / "off"  → no output (DEFAULT, post-bugfix)
        "mismatch"           → only print rows where effective != config
        "1" / "full" / "all" → print every layer
    """
    mode = os.environ.get("VLLM_V100_FP8_DEBUG_SHAPES", "off").lower() or "off"
    if mode not in ("0", "off", "false"):
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
        except Exception:
            rank = -1

        N = int(getattr(layer, "output_size_per_partition", weight.shape[0]))
        K = int(getattr(layer, "input_size_per_partition", weight.shape[1]))
        cfg_h, cfg_w = (int(layer.weight_block_size[0]),
                        int(layer.weight_block_size[1]))
        s0, s1 = int(scale.shape[0]), int(scale.shape[1])
        eff_h = N / s0 if s0 else float("nan")
        eff_w = K / s1 if s1 else float("nan")
        mismatch = (eff_h != cfg_h) or (eff_w != cfg_w)

        if not (mode == "mismatch" and not mismatch):
            prefix = getattr(layer, "prefix", "<unknown>")
            allow = bool(getattr(layer, "allow_fp8_block_shape_mismatch", False))
            logical_widths = getattr(layer, "logical_widths", None)
            print(
                f"[V100-FP8-PWAL rank={rank} pid={os.getpid()}] {prefix} "
                f"weight={tuple(weight.shape)} scale={tuple(scale.shape)} "
                f"N={N} K={K} cfg_block=[{cfg_h},{cfg_w}] "
                f"eff_block=[{eff_h:g},{eff_w:g}] "
                f"mismatch={mismatch} allow_mismatch={allow} "
                f"logical_widths={logical_widths}",
                flush=True,
            )

    _maybe_log_tensor_hashes(layer, weight, scale)


# Layer-prefix SUFFIXES to hash-dump at PWAL time.
# Use suffix matching because vLLM's runtime layer.prefix can prepend
# arbitrary parent-module paths (e.g. "language_model.model.layers.0.*")
# that don't appear in the checkpoint name. The offline emulator and
# the runtime instrumentation match on these suffixes alike.
_HASH_TARGET_PREFIXES = (
    "layers.3.self_attn.qkv_proj",
    "layers.3.self_attn.o_proj",
    "layers.0.linear_attn.in_proj_qkvz",
    "layers.0.linear_attn.out_proj",
    "layers.0.mlp.down_proj",
)

# Print a one-shot banner the first time each TP rank's patched PWAL runs,
# so we can verify the patch path is exercised even if no hash target matches.
_PWAL_BANNERED = False


def _maybe_log_tensor_hashes(layer, weight, scale):
    """Hash-dump weight + scale bytes for a small set of target layers so
    we can diff vLLM-loaded bytes against the offline emulator.

    Enabled by default; set VLLM_V100_FP8_HASH_LAYERS=off to disable.
    """
    if os.environ.get("VLLM_V100_FP8_HASH_LAYERS", "on").lower() in (
            "0", "off", "false"):
        return
    prefix = getattr(layer, "prefix", "")
    matched = None
    for t in _HASH_TARGET_PREFIXES:
        if prefix.endswith(t) or prefix == t:
            matched = t
            break
    if matched is None:
        return

    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
    except Exception:
        rank = -1

    import hashlib
    w_bytes = weight.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    s_bytes = scale.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    w_hash = hashlib.sha256(w_bytes).hexdigest()
    s_hash = hashlib.sha256(s_bytes).hexdigest()

    w_head = list(weight.detach().contiguous().view(torch.uint8)
                  .reshape(-1)[:4].cpu().tolist())
    w_tail = list(weight.detach().contiguous().view(torch.uint8)
                  .reshape(-1)[-4:].cpu().tolist())
    s_head = scale.detach().contiguous().reshape(-1)[:4].cpu().tolist()
    s_tail = scale.detach().contiguous().reshape(-1)[-4:].cpu().tolist()

    print(
        f"[V100-FP8-HASH rank={rank} pid={os.getpid()}] {matched}  "
        f"(layer.prefix={prefix})\n"
        f"  weight  shape={tuple(weight.shape)} dtype={weight.dtype} "
        f"stride={weight.stride()} contig={weight.is_contiguous()} "
        f"sha256={w_hash[:32]} head={w_head} tail={w_tail}\n"
        f"  scale   shape={tuple(scale.shape)} dtype={scale.dtype} "
        f"stride={scale.stride()} contig={scale.is_contiguous()} "
        f"sha256={s_hash[:32]} head={s_head} tail={s_tail}\n"
        f"  weight_block_size={list(layer.weight_block_size)}",
        flush=True,
    )


def _patch_vllm_for_v100():
    """Apply the four patches that make vLLM 0.18.0 work on V100 for block-FP8.
    Called once at module import time, BEFORE vllm.entrypoints.openai.api_server.

    Prints a banner that includes os.getpid() so we can prove the patches ran
    in every TP worker process (grep the serve logs for the marker)."""
    import os
    print(f"[serve_fp8_v100 pid={os.getpid()}] applying V100 FP8 patches "
          f"(volta={_is_volta()})", flush=True)
    from vllm.model_executor.layers.quantization.fp8 import (
        Fp8Config, Fp8LinearMethod,
    )

    # Patch 1: lower the min capability check to allow sm_70.
    original_min_cap = Fp8Config.get_min_capability
    @classmethod
    def patched_min_cap(cls):
        return 70
    Fp8Config.get_min_capability = patched_min_cap

    # Patch 2: in Fp8LinearMethod.__init__, force use_marlin = False on V100.
    # We wrap __init__ — call original first, then override the flag.
    original_init = Fp8LinearMethod.__init__
    def patched_init(self, quant_config):
        original_init(self, quant_config)
        if _is_volta():
            self.use_marlin = False
            self.use_deep_gemm = False
            # The block_quant case will use w8a8_block_fp8_linear (cutlass);
            # we override apply() below to redirect to our kernel instead.
    Fp8LinearMethod.__init__ = patched_init

    # Patch 3: replace process_weights_after_loading on V100 path.
    global _ORIG_process_weights_after_loading
    _ORIG_process_weights_after_loading = Fp8LinearMethod.process_weights_after_loading
    def patched_pwal(self, layer):
        if _is_volta():
            _our_process_weights_after_loading(self, layer)
        else:
            _ORIG_process_weights_after_loading(self, layer)
    Fp8LinearMethod.process_weights_after_loading = patched_pwal

    # Patch 4: replace apply() on V100 path. Fail-closed for non-block FP8:
    # the kernel only supports DeepSeek-style block quantization. Per-tensor
    # or per-channel FP8 on V100 would otherwise fall through to vllm's stock
    # code which assumes sm_75+ (Marlin) or sm_89+ (cutlass) — both of which
    # we've disabled. Better to error explicitly than hang at first inference.
    original_apply = Fp8LinearMethod.apply
    def patched_apply(self, layer, x, bias=None):
        if _is_volta():
            if not getattr(self, "block_quant", False):
                raise NotImplementedError(
                    "serve_fp8_v100: V100 FP8 path supports only block-quantized "
                    "weights (weight_block_size in config.json's quantization_config). "
                    "Per-tensor / per-channel FP8 on sm_70 is not implemented — "
                    "use a block-FP8 model (e.g. DeepSeek-style weight_scale_inv) "
                    "or run on sm_75+ for the stock Marlin path.")
            return _our_apply(self, layer, x, bias)
        return original_apply(self, layer, x, bias)
    Fp8LinearMethod.apply = patched_apply

    print(f"[serve_fp8_v100 pid={os.getpid()}] Patches applied: "
          "min_cap=70, use_marlin=False on V100, apply() routed to our kernel "
          "for block_quant, fail-closed for non-block FP8.", flush=True)


# Apply patches before vllm starts loading models.
_patch_vllm_for_v100()


# ─── Hand off to vLLM's CLI ─────────────────────────────────────────────────
if __name__ == "__main__":
    import runpy
    sys.argv[0] = "vllm-serve"  # cleaner banner in vLLM logs
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
