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
    4. Replace Fp8MoEMethod with a conservative Volta fallback for block-FP8
       routed experts. Stage 1 prioritizes load + correctness over speed.

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
import time
from collections import defaultdict
from pathlib import Path

import torch

from fp8_w8a16_sm70.ext_loader import load_kernel


# ─── Compile our FP8 W8A16 kernel ────────────────────────────────────────────
print("[serve_fp8_v100] Compiling FP8 W8A16 kernel for sm_70 ...", flush=True)
_ext = load_kernel(name="fp8_dequant_ext_vllm")
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

# Stage-1 FP8 MoE fallback. This is intentionally slow: it loops over local
# experts in Python and calls the existing dense FP8 W8A16 kernels twice per
# routed expert group. The point is to unblock load + correctness for FP8 MoE
# models on V100; a fused/batched MoE kernel is a later optimization.
_ENABLE_MOE_FALLBACK = os.environ.get(
    "VLLM_V100_FP8_MOE_FALLBACK", "1").lower() not in ("0", "off", "false")
_MOE_ACTIVE_LIST = os.environ.get(
    "VLLM_V100_FP8_MOE_ACTIVE_LIST", "1").lower() not in ("0", "off", "false")
_MOE_GROUPED_ROUTED_GEMM = os.environ.get(
    "VLLM_V100_FP8_MOE_GROUPED_ROUTED_GEMM", "1").lower() not in (
        "0", "off", "false", "")
_MOE_GROUPED_MAX_ROUTE_SLOTS = int(os.environ.get(
    "VLLM_V100_FP8_MOE_GROUPED_MAX_ROUTE_SLOTS", "32"))
_MOE_GROUPED_K_SPLIT = os.environ.get(
    "VLLM_V100_FP8_MOE_GROUPED_K_SPLIT", "auto").lower()
_MOE_GROUPED_LOG_ONCE = os.environ.get(
    "VLLM_V100_FP8_MOE_GROUPED_LOG_ONCE", "1").lower() not in (
        "0", "off", "false", "")
_MOE_GROUPED_LOGGED = False
_MOE_PROFILE = os.environ.get(
    "VLLM_V100_FP8_MOE_PROFILE", "0").lower() not in ("0", "off", "false", "")
_MOE_PROFILE_EVERY = int(os.environ.get("VLLM_V100_FP8_MOE_PROFILE_EVERY", "64"))
_MOE_PROFILE_STATS = {}
_MOE_PROFILE_TOTAL_CALLS = 0
_MOE_PROFILE_SECTIONS = (
    "index_select",
    "w13_gemm",
    "activation",
    "w2_gemm",
    "scatter",
)

# Coarse decode-time module breakdown. This is deliberately separate from the
# MoE micro-profile above: hooks are cheap and only record CUDA events; we sync
# once at the reporting boundary.
_DECODE_BREAKDOWN = os.environ.get(
    "VLLM_V100_FP8_DECODE_BREAKDOWN", "0").lower() not in (
        "0", "off", "false", "")
_DECODE_BREAKDOWN_EVERY = max(1, int(os.environ.get(
    "VLLM_V100_FP8_DECODE_BREAKDOWN_EVERY", "32")))
_BREAKDOWN_HOOK_CLASSES = {
    "Qwen3NextSparseMoeBlock": "Qwen3NextSparseMoeBlock",
    "Qwen3NextGatedDeltaNet": "Qwen3NextGatedDeltaNet",
    "Qwen3_5GatedDeltaNet": "Qwen3NextGatedDeltaNet",
    "Qwen3NextAttention": "Qwen3NextAttention",
    "LogitsProcessor": "LogitsProcessor",
}
_BREAKDOWN_SECTION_ORDER = (
    "Qwen3NextSparseMoeBlock",
    "Qwen3NextGatedDeltaNet",
    "Qwen3NextAttention",
    "LogitsProcessor",
)
_BREAKDOWN_PENDING_EVENTS = []
_BREAKDOWN_TOTALS = {
    "decode": defaultdict(float),
    "prefill": defaultdict(float),
}
_BREAKDOWN_COUNTS = {
    "decode": defaultdict(int),
    "prefill": defaultdict(int),
}
_BREAKDOWN_TOKEN_EVENTS = []
_BREAKDOWN_DECODE_TOKENS = 0
_BREAKDOWN_TOKEN_OPEN = False
_BREAKDOWN_TOKEN_START = None
_BREAKDOWN_ATTACHED_MODEL_IDS = set()


def _breakdown_rank():
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return -1


def _first_tensor_m(args, kwargs=None):
    values = list(args)
    if kwargs:
        # Prefer hidden_states when modules pass inputs by keyword. Qwen3-Next
        # attention blocks use output/positions kwargs too, and output has the
        # same M, but hidden_states is the clearest regime signal.
        if "hidden_states" in kwargs:
            values.insert(0, kwargs["hidden_states"])
        values.extend(v for k, v in kwargs.items() if k != "hidden_states")
    for arg in values:
        if isinstance(arg, torch.Tensor) and arg.ndim > 0:
            return int(arg.shape[0])
        if isinstance(arg, (tuple, list)):
            nested = _first_tensor_m(arg)
            if nested is not None:
                return nested
    return None


def _breakdown_pre_hook(module, args, kwargs):
    if not torch.cuda.is_available():
        return
    cls_name = type(module).__name__
    section = _BREAKDOWN_HOOK_CLASSES.get(cls_name, cls_name)
    m = _first_tensor_m(args, kwargs)
    regime = "decode" if m == 1 else "prefill"
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    stack = getattr(module, "_v100_breakdown_stack", None)
    if stack is None:
        stack = []
        setattr(module, "_v100_breakdown_stack", stack)
    stack.append((regime, start))

    global _BREAKDOWN_TOKEN_OPEN, _BREAKDOWN_TOKEN_START
    if (regime == "decode" and section != "LogitsProcessor"
            and not _BREAKDOWN_TOKEN_OPEN):
        _BREAKDOWN_TOKEN_START = torch.cuda.Event(enable_timing=True)
        _BREAKDOWN_TOKEN_START.record()
        _BREAKDOWN_TOKEN_OPEN = True


def _breakdown_post_hook(module, args, kwargs, output):
    if not torch.cuda.is_available():
        return
    stack = getattr(module, "_v100_breakdown_stack", None)
    if not stack:
        return
    regime, start = stack.pop()
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    cls_name = type(module).__name__
    section = _BREAKDOWN_HOOK_CLASSES.get(cls_name, cls_name)
    _BREAKDOWN_PENDING_EVENTS.append((regime, section, start, end))

    global _BREAKDOWN_DECODE_TOKENS, _BREAKDOWN_TOKEN_OPEN, _BREAKDOWN_TOKEN_START
    if regime == "decode" and section == "LogitsProcessor":
        if _BREAKDOWN_TOKEN_OPEN and _BREAKDOWN_TOKEN_START is not None:
            _BREAKDOWN_TOKEN_EVENTS.append((_BREAKDOWN_TOKEN_START, end))
        _BREAKDOWN_TOKEN_OPEN = False
        _BREAKDOWN_TOKEN_START = None
        _BREAKDOWN_DECODE_TOKENS += 1
        if (_BREAKDOWN_DECODE_TOKENS % _DECODE_BREAKDOWN_EVERY) == 0:
            _breakdown_flush()


def _breakdown_flush():
    if not _BREAKDOWN_PENDING_EVENTS and not _BREAKDOWN_TOKEN_EVENTS:
        return
    torch.cuda.synchronize()
    interval_totals = defaultdict(float)
    interval_counts = defaultdict(int)
    for regime, section, start, end in _BREAKDOWN_PENDING_EVENTS:
        elapsed = float(start.elapsed_time(end))
        _BREAKDOWN_TOTALS[regime][section] += elapsed
        _BREAKDOWN_COUNTS[regime][section] += 1
        if regime == "decode":
            interval_totals[section] += elapsed
            interval_counts[section] += 1
    _BREAKDOWN_PENDING_EVENTS.clear()

    total_ms = 0.0
    for start, end in _BREAKDOWN_TOKEN_EVENTS:
        total_ms += float(start.elapsed_time(end))
    _BREAKDOWN_TOKEN_EVENTS.clear()

    if _breakdown_rank() not in (0, -1):
        return
    tokens = max(1, _BREAKDOWN_DECODE_TOKENS)
    if total_ms <= 0.0:
        total_ms = sum(_BREAKDOWN_TOTALS["decode"].values())
    total_per_token = total_ms / max(1, _DECODE_BREAKDOWN_EVERY)
    section_sum = 0.0
    lines = []
    for section in _BREAKDOWN_SECTION_ORDER:
        ms = interval_totals.get(section, 0.0)
        if ms <= 0.0:
            continue
        per_token = ms / max(1, _DECODE_BREAKDOWN_EVERY)
        section_sum += per_token
        pct = (100.0 * per_token / total_per_token) if total_per_token else 0.0
        calls = interval_counts.get(section, 0)
        lines.append(
            f"  {section:<30} {per_token:8.3f} ms/token ({pct:5.1f}%) "
            f"calls={calls}")
    other = max(0.0, total_per_token - section_sum)
    other_pct = (100.0 * other / total_per_token) if total_per_token else 0.0
    print(
        f"[DECODE-BREAKDOWN rank={_breakdown_rank()} pid={os.getpid()} "
        f"tokens={_BREAKDOWN_DECODE_TOKENS} decode]",
        flush=True,
    )
    for line in lines:
        print(line, flush=True)
    print(
        f"  {'Other (residual)':<30} {other:8.3f} ms/token ({other_pct:5.1f}%)",
        flush=True,
    )
    print(
        f"  {'Total':<30} {total_per_token:8.3f} ms/token (100.0%)",
        flush=True,
    )


def _attach_decode_breakdown_hooks(model):
    if not _DECODE_BREAKDOWN or not torch.cuda.is_available():
        return
    model_id = id(model)
    if model_id in _BREAKDOWN_ATTACHED_MODEL_IDS:
        return
    _BREAKDOWN_ATTACHED_MODEL_IDS.add(model_id)

    attached = defaultdict(int)
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name not in _BREAKDOWN_HOOK_CLASSES:
            continue
        module.register_forward_pre_hook(_breakdown_pre_hook, with_kwargs=True)
        module.register_forward_hook(_breakdown_post_hook, with_kwargs=True)
        attached[_BREAKDOWN_HOOK_CLASSES[cls_name]] += 1
    if _breakdown_rank() in (0, -1):
        parts = ", ".join(f"{name}={count}" for name, count in sorted(attached.items()))
        print(
            f"[DECODE-BREAKDOWN rank={_breakdown_rank()} pid={os.getpid()}] "
            f"attached hooks: {parts or 'none'} every={_DECODE_BREAKDOWN_EVERY}",
            flush=True,
        )


def _v100_fp8_gemm(
    x2d: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    N: int,
    K: int,
    block_h: int,
    block_w: int,
):
    """Run the same FP8 W8A16 dispatch used by patched dense Linear.

    `weight` is a 2D FP8 tensor [N, K], `scale` is [ceil(N/bh), ceil(K/bw)].
    Returns a 2D FP16 output [M, N].
    """
    if x2d.dtype != torch.float16:
        x2d = x2d.to(torch.float16)
    x2d = x2d.contiguous()
    weight_u8 = weight.view(torch.uint8).reshape(-1).contiguous()
    scales = scale.reshape(-1).contiguous()
    M = x2d.size(0)

    def k_split_ok(k_split):
        return (K % (k_split * block_w)) == 0

    wmma_layer_ok = (
        _HAS_WMMA
        and (N % _WMMA_TILE_N) == 0
        and (K % _WMMA_TILE_K) == 0
        and block_h == 128 and block_w == 128
    )

    if M <= _DISPATCH_M_A3_K8 and k_split_ok(8):
        return _ext.fp8_w8a16_gemm_a3(
            x2d, weight_u8, scales, N, K, block_h, block_w, 8), "A.3 k=8"
    if M <= _DISPATCH_M_A3_K4 and k_split_ok(4):
        return _ext.fp8_w8a16_gemm_a3(
            x2d, weight_u8, scales, N, K, block_h, block_w, 4), "A.3 k=4"
    if wmma_layer_ok and M >= _WMMA_MIN_M:
        M_aligned = (M // _WMMA_TILE_M) * _WMMA_TILE_M
        M_tail = M - M_aligned
        x_main = x2d[:M_aligned].contiguous()
        out_main = _ext.fp8_w8a16_gemm_wmma_poc(
            x_main, weight_u8, scales, N, K, block_h, block_w)
        if M_tail > 0:
            x_tail = x2d[M_aligned:].contiguous()
            out_tail = _ext.fp8_w8a16_gemm_a2(
                x_tail, weight_u8, scales, N, K, block_h, block_w)
            return torch.cat([out_main, out_tail], dim=0), "WMMA+A.2(tail)"
        return out_main, "WMMA"
    if M >= _DISPATCH_M_A2:
        return _ext.fp8_w8a16_gemm_a2(
            x2d, weight_u8, scales, N, K, block_h, block_w), "A.2"
    return _ext.fp8_w8a16_gemm_a1(
        x2d, weight_u8, scales, N, K, block_h, block_w), "A.1"


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

    out, variant = _v100_fp8_gemm(
        x2d, layer.weight, layer.weight_scale_inv, N, K, block_h, block_w)

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


def _our_moe_init(self, quant_config, layer):
    """Volta replacement for Fp8MoEMethod.__init__.

    The stock constructor selects an FP8 MoE backend and raises on sm_70. For
    stage 1 we bypass backend selection and use our direct Python fallback in
    `apply()`.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    FusedMoEMethodBase.__init__(self, layer.moe_config)
    self.quant_config = quant_config
    self.weight_block_size = self.quant_config.weight_block_size
    self.block_quant = self.weight_block_size is not None
    self.weight_scale_name = (
        "weight_scale_inv" if self.block_quant else "weight_scale"
    )
    self.fp8_backend = None
    self.experts_cls = None
    self.moe_kernel = None
    if hasattr(self, "experts_cls"):
        delattr(self, "experts_cls")


def _our_moe_process_weights_after_loading(self, layer):
    """Volta FP8 MoE PWAL fallback.

    Keep the model-loaded [E, 2I, H] and [E, H, I] FP8 layouts intact, cast
    block scales to FP16, and skip all modular-kernel setup.
    """
    from vllm.model_executor.utils import replace_parameter

    if getattr(layer, "_already_called_process_weights_after_loading", False):
        return

    if not self.block_quant:
        raise NotImplementedError(
            "serve_fp8_v100: V100 FP8 MoE fallback supports only block-FP8 "
            "weights with weight_block_size."
        )

    w13 = layer.w13_weight.contiguous()
    w2 = layer.w2_weight.contiguous()
    w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
    w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
    if w13_scale.dtype != torch.float16:
        w13_scale = w13_scale.to(torch.float16)
    if w2_scale.dtype != torch.float16:
        w2_scale = w2_scale.to(torch.float16)
    w13_scale = w13_scale.contiguous()
    w2_scale = w2_scale.contiguous()

    replace_parameter(layer, "w13_weight", w13.data)
    replace_parameter(layer, "w2_weight", w2.data)
    replace_parameter(layer, f"w13_{self.weight_scale_name}", w13_scale.data)
    replace_parameter(layer, f"w2_{self.weight_scale_name}", w2_scale.data)

    layer.w13_input_scale = None
    layer.w2_input_scale = None
    self.moe_quant_config = None
    self.moe_kernel = None
    layer._already_called_process_weights_after_loading = True

    if os.environ.get("VLLM_V100_FP8_MOE_DEBUG", "0").lower() not in (
            "0", "off", "false", ""):
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
        except Exception:
            rank = -1
        print(
            f"[V100-FP8-MOE-PWAL rank={rank} pid={os.getpid()}] "
            f"{getattr(layer, 'prefix', '<unknown>')} "
            f"w13={tuple(w13.shape)} w13_scale={tuple(w13_scale.shape)} "
            f"w2={tuple(w2.shape)} w2_scale={tuple(w2_scale.shape)} "
            f"block={self.weight_block_size}",
            flush=True,
        )


def _moe_activation(layer, x):
    activation = getattr(layer, "activation", None)
    name = getattr(activation, "name", str(activation)).upper()
    if "SILU" in name or "SWIGLU" in name:
        return torch.nn.functional.silu(x)
    if "GELU" in name:
        return torch.nn.functional.gelu(x)
    raise NotImplementedError(
        f"serve_fp8_v100: unsupported FP8 MoE activation {activation!r}"
    )


def _moe_grouped_routed_k_splits(
    hidden_size,
    intermediate,
    block_h,
    block_w,
    route_slots,
):
    if not _MOE_GROUPED_ROUTED_GEMM:
        return None
    if not hasattr(_ext, "fp8_w8a16_grouped_routed_gemm_a3"):
        return None
    if block_h != 128 or block_w != 128:
        return None
    if route_slots > _MOE_GROUPED_MAX_ROUTE_SLOTS:
        return None

    def parse_forced_split():
        if _MOE_GROUPED_K_SPLIT in ("", "auto", "default"):
            return None
        try:
            value = int(_MOE_GROUPED_K_SPLIT)
        except ValueError as exc:
            raise ValueError(
                "VLLM_V100_FP8_MOE_GROUPED_K_SPLIT must be auto, 1, 2, 4, or 8"
            ) from exc
        if value not in (1, 2, 4, 8):
            raise ValueError(
                "VLLM_V100_FP8_MOE_GROUPED_K_SPLIT must be auto, 1, 2, 4, or 8"
            )
        return value

    def pick_split(K, max_split=8):
        for split in (8, 4, 2, 1):
            if split > max_split:
                continue
            if (K % (split * block_w)) == 0:
                return split
        return None

    pinned = parse_forced_split()
    max_split = 8 if pinned is None else pinned
    w13_k_split = pick_split(hidden_size, max_split)
    w2_k_split = pick_split(intermediate, max_split)
    if w13_k_split is None or w2_k_split is None:
        return None
    return w13_k_split, w2_k_split


def _our_moe_apply_grouped(
    self,
    layer,
    x,
    x_work,
    topk_weights,
    local_topk,
    hidden_size,
    intermediate,
    block_h,
    block_w,
    profile,
    call_stats,
    timed_cuda,
    w13_k_split,
    w2_k_split,
):
    """Grouped-routed MoE path: one w13 GEMM and one w2 GEMM per layer call."""
    global _MOE_GROUPED_LOGGED

    valid_mask = local_topk >= 0
    token_idx, route_idx = torch.nonzero(valid_mask, as_tuple=True)
    route_count = int(token_idx.numel())
    out = torch.zeros((x_work.size(0), hidden_size),
                      dtype=torch.float16, device=x.device)
    if route_count == 0:
        return out

    local_expert_ids = local_topk[token_idx, route_idx].to(torch.int64).contiguous()
    route_w = topk_weights[token_idx, route_idx].to(torch.float16)

    if profile:
        call_stats["active_experts"] += int(torch.unique(local_expert_ids).numel())
        call_stats["routed_items"] += route_count
        call_stats["skipped_experts"] += (
            int(layer.local_num_experts) - call_stats["active_experts"])

    if profile:
        route_x = timed_cuda(
            "index_select", lambda: x_work.index_select(0, token_idx))
    else:
        route_x = x_work.index_select(0, token_idx)

    apply_weight_on_input = bool(getattr(layer, "apply_router_weight_on_input", False))
    if apply_weight_on_input:
        route_x = route_x * route_w.unsqueeze(-1)
    route_x = route_x.contiguous()

    w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
    w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
    w13_weight_view = layer.w13_weight.view(torch.uint8)
    w2_weight_view = layer.w2_weight.view(torch.uint8)
    w13_weight = w13_weight_view.contiguous()
    w2_weight = w2_weight_view.contiguous()

    if _MOE_GROUPED_LOG_ONCE and not _MOE_GROUPED_LOGGED:
        _MOE_GROUPED_LOGGED = True
        try:
            rank = _moe_profile_rank()
        except Exception:
            rank = -1
        w13_zero_copy = (
            w13_weight.data_ptr() == w13_weight_view.data_ptr()
            and w13_weight.is_contiguous()
        )
        w2_zero_copy = (
            w2_weight.data_ptr() == w2_weight_view.data_ptr()
            and w2_weight.is_contiguous()
        )
        print(
            f"[V100-FP8-MOE-GROUPED rank={rank} pid={os.getpid()}] "
            f"enabled prefix={getattr(layer, 'prefix', '<unknown>')} "
            f"M={x_work.size(0)} route_slots={local_topk.numel()} "
            f"route_count={route_count} hidden={hidden_size} "
            f"intermediate={intermediate} block=({block_h},{block_w}) "
            f"w13_k_split={w13_k_split} w2_k_split={w2_k_split} "
            f"k_split_env={_MOE_GROUPED_K_SPLIT} "
            f"max_route_slots={_MOE_GROUPED_MAX_ROUTE_SLOTS} "
            f"w13_u8_zero_copy={w13_zero_copy} "
            f"w2_u8_zero_copy={w2_zero_copy}",
            flush=True,
        )

    def run_w13_grouped():
        return _ext.fp8_w8a16_grouped_routed_gemm_a3(
            route_x,
            local_expert_ids,
            w13_weight,
            w13_scale,
            2 * intermediate,
            hidden_size,
            block_h,
            block_w,
            w13_k_split,
        )

    if profile:
        w13 = timed_cuda("w13_gemm", run_w13_grouped)
    else:
        w13 = run_w13_grouped()
    gate = w13[:, :intermediate]
    up = w13[:, intermediate:]

    if profile:
        hidden = timed_cuda(
            "activation", lambda: _moe_activation(layer, gate) * up)
    else:
        hidden = _moe_activation(layer, gate) * up
    hidden = hidden.contiguous()

    def run_w2_grouped():
        return _ext.fp8_w8a16_grouped_routed_gemm_a3(
            hidden,
            local_expert_ids,
            w2_weight,
            w2_scale,
            hidden_size,
            intermediate,
            block_h,
            block_w,
            w2_k_split,
        )

    if profile:
        expert_out = timed_cuda("w2_gemm", run_w2_grouped)
    else:
        expert_out = run_w2_grouped()
    if not apply_weight_on_input:
        expert_out = expert_out * route_w.unsqueeze(-1)
    if profile:
        timed_cuda("scatter", lambda: out.index_add_(0, token_idx, expert_out))
    else:
        out.index_add_(0, token_idx, expert_out)
    return out


def _moe_profile_rank():
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return -1


def _new_moe_profile_stats():
    return {
        "calls": 0,
        "wall_ms": 0.0,
        "routing_wall_ms": 0.0,
        "mask_sync_wall_ms": 0.0,
        "active_experts": 0,
        "routed_items": 0,
        "empty_expert_iters": 0,
        "skipped_experts": 0,
        "active_hist": {},
        "sections": {name: 0.0 for name in _MOE_PROFILE_SECTIONS},
    }


def _moe_profile_update(prefix, call_stats):
    global _MOE_PROFILE_TOTAL_CALLS

    stats = _MOE_PROFILE_STATS.setdefault(prefix, _new_moe_profile_stats())
    stats["calls"] += 1
    stats["wall_ms"] += call_stats["wall_ms"]
    stats["routing_wall_ms"] += call_stats["routing_wall_ms"]
    stats["mask_sync_wall_ms"] += call_stats["mask_sync_wall_ms"]
    stats["active_experts"] += call_stats["active_experts"]
    stats["routed_items"] += call_stats["routed_items"]
    stats["empty_expert_iters"] += call_stats["empty_expert_iters"]
    stats["skipped_experts"] += call_stats["skipped_experts"]
    hist = stats["active_hist"]
    active = call_stats["active_experts"]
    hist[active] = hist.get(active, 0) + 1
    for name in _MOE_PROFILE_SECTIONS:
        stats["sections"][name] += call_stats["sections"].get(name, 0.0)

    _MOE_PROFILE_TOTAL_CALLS += 1
    if (_MOE_PROFILE_TOTAL_CALLS % _MOE_PROFILE_EVERY) != 0:
        return
    if _moe_profile_rank() not in (0, -1):
        return

    totals = _new_moe_profile_stats()
    for layer_stats in _MOE_PROFILE_STATS.values():
        for key in (
            "calls",
            "active_experts",
            "routed_items",
            "empty_expert_iters",
            "skipped_experts",
        ):
            totals[key] += layer_stats[key]
        for key in ("wall_ms", "routing_wall_ms", "mask_sync_wall_ms"):
            totals[key] += layer_stats[key]
        for name in _MOE_PROFILE_SECTIONS:
            totals["sections"][name] += layer_stats["sections"][name]

    calls = max(1, totals["calls"])
    section_bits = " ".join(
        f"{name}={totals['sections'][name] / calls:.3f}ms"
        for name in _MOE_PROFILE_SECTIONS
    )
    print(
        f"[V100-FP8-MOE-PROFILE rank={_moe_profile_rank()} pid={os.getpid()}] "
        f"calls={totals['calls']} avg_wall={totals['wall_ms'] / calls:.3f}ms "
        f"routing={totals['routing_wall_ms'] / calls:.3f}ms "
        f"mask_sync={totals['mask_sync_wall_ms'] / calls:.3f}ms "
        f"active_experts={totals['active_experts'] / calls:.2f}/call "
        f"routed_items={totals['routed_items'] / calls:.2f}/call "
        f"empty_iters={totals['empty_expert_iters'] / calls:.2f}/call "
        f"skipped_experts={totals['skipped_experts'] / calls:.2f}/call "
        f"{section_bits}",
        flush=True,
    )

    top_layers = sorted(
        _MOE_PROFILE_STATS.items(),
        key=lambda item: item[1]["wall_ms"],
        reverse=True,
    )[:3]
    for layer_prefix, layer_stats in top_layers:
        layer_calls = max(1, layer_stats["calls"])
        hist = ",".join(
            f"{active}:{count}"
            for active, count in sorted(layer_stats["active_hist"].items())
        )
        print(
            f"[V100-FP8-MOE-PROFILE rank={_moe_profile_rank()} pid={os.getpid()}] "
            f"top_layer={layer_prefix} calls={layer_stats['calls']} "
            f"avg_wall={layer_stats['wall_ms'] / layer_calls:.3f}ms "
            f"active_hist={hist}",
            flush=True,
        )


def _our_moe_apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
    """Slow Volta FP8 MoE forward.

    Expert-major loop:
      selected tokens -> w13 FP8 GEMM -> SwiGLU/GELU gate -> w2 FP8 GEMM
      -> route-weighted scatter-add.
    """
    del shared_experts_input  # Shared experts are handled by DefaultMoERunner.

    if not self.block_quant:
        raise NotImplementedError(
            "serve_fp8_v100: V100 FP8 MoE fallback supports only block-FP8."
        )

    if x.dtype != torch.float16:
        x_work = x.to(torch.float16)
    else:
        x_work = x
    x_work = x_work.contiguous()
    profile = _MOE_PROFILE and torch.cuda.is_available()
    if profile:
        call_t0 = time.perf_counter()
        call_stats = {
            "wall_ms": 0.0,
            "routing_wall_ms": 0.0,
            "mask_sync_wall_ms": 0.0,
            "active_experts": 0,
            "routed_items": 0,
            "empty_expert_iters": 0,
            "skipped_experts": 0,
            "sections": {name: 0.0 for name in _MOE_PROFILE_SECTIONS},
        }
        cuda_events = []

        def timed_cuda(name, fn):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            cuda_events.append((name, start, end))
            return result
    else:
        call_stats = None
        cuda_events = None

    M, hidden_size = x_work.shape
    intermediate = int(layer.intermediate_size_per_partition)
    block_h, block_w = layer.weight_block_size

    # Convert global expert ids to local ids. Non-local experts become -1 and
    # contribute zero on this rank; vLLM's later TP/EP combine handles the sum.
    routing_t0 = time.perf_counter() if profile else None
    local_topk = topk_ids.to(torch.long)
    expert_map = getattr(layer, "expert_map", None)
    if expert_map is not None:
        safe_ids = torch.clamp(local_topk, min=0)
        local_topk = expert_map.to(device=topk_ids.device)[safe_ids]
        local_topk = torch.where(topk_ids < 0, torch.full_like(local_topk, -1), local_topk)
    if profile:
        call_stats["routing_wall_ms"] += (time.perf_counter() - routing_t0) * 1000.0

    grouped_k_splits = _moe_grouped_routed_k_splits(
        hidden_size,
        intermediate,
        block_h,
        block_w,
        int(local_topk.numel()),
    )
    if grouped_k_splits is not None:
        out = _our_moe_apply_grouped(
            self,
            layer,
            x,
            x_work,
            topk_weights,
            local_topk,
            hidden_size,
            intermediate,
            block_h,
            block_w,
            profile,
            call_stats,
            timed_cuda if profile else None,
            grouped_k_splits[0],
            grouped_k_splits[1],
        )
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        if profile:
            torch.cuda.synchronize()
            for name, start, end in cuda_events:
                call_stats["sections"][name] += float(start.elapsed_time(end))
            call_stats["wall_ms"] = (time.perf_counter() - call_t0) * 1000.0
            _moe_profile_update(getattr(layer, "prefix", "<unknown>"), call_stats)
        return out

    w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
    w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
    out = torch.zeros((M, hidden_size), dtype=torch.float16, device=x.device)
    apply_weight_on_input = bool(getattr(layer, "apply_router_weight_on_input", False))

    local_num_experts = int(layer.local_num_experts)
    if _MOE_ACTIVE_LIST:
        if profile:
            mask_t0 = time.perf_counter()
        expert_iter = torch.unique(local_topk[local_topk >= 0]).tolist()
        if profile:
            call_stats["mask_sync_wall_ms"] += (
                time.perf_counter() - mask_t0) * 1000.0
            call_stats["skipped_experts"] += (
                local_num_experts - len(expert_iter))
    else:
        expert_iter = range(local_num_experts)

    for local_expert in expert_iter:
        mask = local_topk == local_expert
        if not _MOE_ACTIVE_LIST:
            if profile:
                mask_t0 = time.perf_counter()
                has_tokens = bool(mask.any().item())
                call_stats["mask_sync_wall_ms"] += (
                    time.perf_counter() - mask_t0) * 1000.0
            else:
                has_tokens = bool(mask.any().item())
            if not has_tokens:
                if profile:
                    call_stats["empty_expert_iters"] += 1
                continue
        token_idx, route_idx = torch.nonzero(mask, as_tuple=True)
        route_count = int(token_idx.numel())
        if profile:
            call_stats["active_experts"] += 1
            call_stats["routed_items"] += route_count
        if profile:
            route_x = timed_cuda(
                "index_select", lambda: x_work.index_select(0, token_idx))
        else:
            route_x = x_work.index_select(0, token_idx)
        route_w = topk_weights[token_idx, route_idx].to(torch.float16)
        if apply_weight_on_input:
            route_x = route_x * route_w.unsqueeze(-1)

        def run_w13():
            return _v100_fp8_gemm(
                route_x,
                layer.w13_weight[local_expert],
                w13_scale[local_expert],
                2 * intermediate,
                hidden_size,
                block_h,
                block_w,
            )

        if profile:
            w13, _ = timed_cuda("w13_gemm", run_w13)
        else:
            w13, _ = run_w13()
        gate = w13[:, :intermediate]
        up = w13[:, intermediate:]

        if profile:
            hidden = timed_cuda(
                "activation", lambda: _moe_activation(layer, gate) * up)
        else:
            hidden = _moe_activation(layer, gate) * up

        def run_w2():
            return _v100_fp8_gemm(
                hidden,
                layer.w2_weight[local_expert],
                w2_scale[local_expert],
                hidden_size,
                intermediate,
                block_h,
                block_w,
            )

        if profile:
            expert_out, _ = timed_cuda("w2_gemm", run_w2)
        else:
            expert_out, _ = run_w2()
        if not apply_weight_on_input:
            expert_out = expert_out * route_w.unsqueeze(-1)
        if profile:
            timed_cuda("scatter", lambda: out.index_add_(0, token_idx, expert_out))
        else:
            out.index_add_(0, token_idx, expert_out)

    if out.dtype != x.dtype:
        out = out.to(x.dtype)
    if profile:
        torch.cuda.synchronize()
        for name, start, end in cuda_events:
            call_stats["sections"][name] += float(start.elapsed_time(end))
        call_stats["wall_ms"] = (time.perf_counter() - call_t0) * 1000.0
        _moe_profile_update(getattr(layer, "prefix", "<unknown>"), call_stats)
    return out


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
        Fp8Config, Fp8LinearMethod, Fp8MoEMethod, Fp8OnlineMoEMethod,
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

    # Patch 5: bypass stock FP8 MoE backend selection on V100. Stock vLLM has
    # no sm_70 FP8 MoE backend, so Fp8MoEMethod.__init__ raises during model
    # construction for Qwen3.5-122B-A10B-FP8.
    original_moe_init = Fp8MoEMethod.__init__
    original_online_moe_init = Fp8OnlineMoEMethod.__init__

    def patched_moe_init(self, quant_config, layer):
        if _is_volta() and _ENABLE_MOE_FALLBACK:
            _our_moe_init(self, quant_config, layer)
        else:
            original_moe_init(self, quant_config, layer)

    def patched_online_moe_init(self, quant_config, layer):
        if _is_volta() and _ENABLE_MOE_FALLBACK:
            _our_moe_init(self, quant_config, layer)
        else:
            original_online_moe_init(self, quant_config, layer)

    Fp8MoEMethod.__init__ = patched_moe_init
    Fp8OnlineMoEMethod.__init__ = patched_online_moe_init

    # Patch 6: keep FP8 MoE weights in model-loaded layout and skip
    # make_fp8_moe_kernel().
    original_moe_pwal = Fp8MoEMethod.process_weights_after_loading
    original_online_moe_pwal = Fp8OnlineMoEMethod.process_weights_after_loading

    def patched_moe_pwal(self, layer):
        if _is_volta() and _ENABLE_MOE_FALLBACK:
            return _our_moe_process_weights_after_loading(self, layer)
        return original_moe_pwal(self, layer)

    def patched_online_moe_pwal(self, layer):
        if _is_volta() and _ENABLE_MOE_FALLBACK:
            return _our_moe_process_weights_after_loading(self, layer)
        return original_online_moe_pwal(self, layer)

    Fp8MoEMethod.process_weights_after_loading = patched_moe_pwal
    Fp8OnlineMoEMethod.process_weights_after_loading = patched_online_moe_pwal

    # Patch 7: direct routed-expert fallback.
    original_moe_apply = Fp8MoEMethod.apply
    original_online_moe_apply = Fp8OnlineMoEMethod.apply

    def patched_moe_apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
        if _is_volta() and _ENABLE_MOE_FALLBACK:
            return _our_moe_apply(
                self, layer, x, topk_weights, topk_ids, shared_experts_input)
        return original_moe_apply(
            self, layer, x, topk_weights, topk_ids, shared_experts_input)

    def patched_online_moe_apply(self, layer, x, topk_weights, topk_ids,
                                 shared_experts_input):
        if _is_volta() and _ENABLE_MOE_FALLBACK:
            return _our_moe_apply(
                self, layer, x, topk_weights, topk_ids, shared_experts_input)
        return original_online_moe_apply(
            self, layer, x, topk_weights, topk_ids, shared_experts_input)

    Fp8MoEMethod.apply = patched_moe_apply
    Fp8OnlineMoEMethod.apply = patched_online_moe_apply

    # Patch 8: vLLM may later ask the quant method to create modular-kernel
    # prepare/finalize state. The stock FP8 method raises here because it
    # expects an internal MK to already exist. Our fallback is intentionally
    # direct, so returning None keeps the runner on quant_method.apply().
    original_moe_maybe_pf = Fp8MoEMethod.maybe_make_prepare_finalize
    original_online_moe_maybe_pf = Fp8OnlineMoEMethod.maybe_make_prepare_finalize

    def patched_moe_maybe_pf(self, routing_tables=None):
        if _is_volta() and _ENABLE_MOE_FALLBACK:
            return None
        return original_moe_maybe_pf(self, routing_tables)

    def patched_online_moe_maybe_pf(self, routing_tables=None):
        if _is_volta() and _ENABLE_MOE_FALLBACK:
            return None
        return original_online_moe_maybe_pf(self, routing_tables)

    Fp8MoEMethod.maybe_make_prepare_finalize = patched_moe_maybe_pf
    Fp8OnlineMoEMethod.maybe_make_prepare_finalize = patched_online_moe_maybe_pf

    # Patch 9: optional coarse decode breakdown for Qwen3-Next/Qwen3.5.
    # Attach once after model construction so forward hooks cover all layers but
    # avoid parent decoder layers that would double-count child modules.
    if _DECODE_BREAKDOWN:
        try:
            from vllm.model_executor.models import qwen3_next, qwen3_5

            original_qwen3next_init = qwen3_next.Qwen3NextForCausalLM.__init__

            def patched_qwen3next_init(self, *args, **kwargs):
                original_qwen3next_init(self, *args, **kwargs)
                _attach_decode_breakdown_hooks(self)

            qwen3_next.Qwen3NextForCausalLM.__init__ = patched_qwen3next_init

            original_qwen35_base_init = qwen3_5.Qwen3_5ForCausalLMBase.__init__

            def patched_qwen35_base_init(self, *args, **kwargs):
                original_qwen35_base_init(self, *args, **kwargs)
                _attach_decode_breakdown_hooks(self)

            qwen3_5.Qwen3_5ForCausalLMBase.__init__ = patched_qwen35_base_init
        except Exception as exc:
            print(
                f"[DECODE-BREAKDOWN rank={_breakdown_rank()} pid={os.getpid()}] "
                f"failed to install hooks: {type(exc).__name__}: {exc}",
                flush=True,
            )

    print(f"[serve_fp8_v100 pid={os.getpid()}] Patches applied: "
          "min_cap=70, use_marlin=False on V100, apply() routed to our kernel "
          "for block_quant, FP8 MoE fallback="
          f"{'on' if _ENABLE_MOE_FALLBACK else 'off'}, "
          "fail-closed for non-block FP8.", flush=True)


# Apply patches before vllm starts loading models.
_patch_vllm_for_v100()


# ─── Hand off to vLLM's CLI ─────────────────────────────────────────────────
if __name__ == "__main__":
    import runpy
    sys.argv[0] = "vllm-serve"  # cleaner banner in vLLM logs
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
