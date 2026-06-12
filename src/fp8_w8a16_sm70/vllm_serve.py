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

# Dynamo / torch.compile cannot trace into pybind11 C extension functions. On
# the cu128 stack we forced --enforce-eager so dynamo never ran, hiding the
# problem; on py3.12 + cudagraph (vllm 0.18, FULL_AND_PIECEWISE) the dynamo
# pass tries to inline these calls. `allow_in_graph` alone is insufficient —
# dynamo still tries to execute the kernel against FakeTensors during shape
# inference and aborts with "tensor data is not allocated yet". The proper
# long-term fix is torch.library.custom_op with a fake/meta impl per entry
# point; for now we use the narrower graph-break tool: wrap each pybind entry
# point with torch._dynamo.disable so dynamo bails to eager for just that
# call. PIECEWISE cudagraph captures around each break. Expected overhead:
# ~2-5 μs per break × ~200 GEMMs/token = ~0.4-1.0 ms/token (5-15% of cudagraph
# win). Acceptable cost as a path-validating probe.
import torch._dynamo as _dynamo  # noqa: E402
for _name in (
    "fp8_w8a16_gemm_a1",
    "fp8_w8a16_gemm_a2",
    "fp8_w8a16_gemm_a3",
    "fp8_w8a16_gemm_wmma_poc",
    "fp8_w8a16_grouped_routed_gemm_a3",
):
    _fn = getattr(_ext, _name, None)
    if _fn is not None:
        setattr(_ext, _name, _dynamo.disable(_fn))
del _dynamo, _fn, _name


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
# Phase 4 Stage 2: route CHANNEL-scale (block_h=1) Linears through the WMMA
# (tensor-core) kernel at prefill M too, not just block_h=128. The WMMA kernel now
# applies per-output-row scale for block_h=1. This is the PREFILL lever for the CT
# resident dense Linears (qkv/o_proj, N%64==0), which otherwise fall to A.2
# (CUDA-core, ~1.75 TFLOP/s vs WMMA ~17). Decode (M<=8 -> A.3) is unaffected; only
# prefill M>=_WMMA_MIN_M uses WMMA. Kill switch: VLLM_V100_CT_CHANNEL_WMMA=0.
_CT_CHANNEL_WMMA = os.environ.get(
    "VLLM_V100_CT_CHANNEL_WMMA", "1").lower() not in ("0", "off", "false", "")
_HAS_COALESCED_GEMV = hasattr(_ext, "fp8_w8a16_gemv_coalesced")
_HAS_COALESCED_GEMV_M = hasattr(_ext, "fp8_w8a16_gemv_coalesced_m")
_COALESCED_GEMV = os.environ.get(
    "VLLM_V100_FP8_COALESCED_GEMV", "0").lower() not in (
        "0", "off", "false", "")
try:
    _COALESCED_GEMV_M_MAX = int(os.environ.get(
        "VLLM_V100_FP8_COALESCED_GEMV_M_MAX", "1"))
except ValueError:
    _COALESCED_GEMV_M_MAX = 1
_COALESCED_GEMV_M_MAX = max(1, min(8, _COALESCED_GEMV_M_MAX))
# Grouped coalesced MoE w13 decode GEMV for the BLOCK-FP8 (Qwen quant_method=fp8)
# path. Mirrors the CT Stage G1 wiring (VLLM_V100_CT_MOE_W13_COALESCED) but for
# the `_our_moe_apply_grouped` decode kernel, which the CT flag never touches.
# Safe: _our_moe_apply_grouped is decode-only (route_slots <= 32 gate) and only
# runs for block_h==block_w==128, exactly what the grouped coalesced kernel needs
# (it also requires K=hidden_size % 128 == 0). Default ON; kill switch = 0/off.
_HAS_GROUPED_COALESCED_GEMV = hasattr(_ext, "fp8_w8a16_grouped_gemv_coalesced")
_MOE_W13_COALESCED = os.environ.get(
    "VLLM_V100_FP8_MOE_W13_COALESCED", "1").lower() not in (
        "0", "off", "false", "")
_MOE_W13_COAL_ENGAGED = [False]

# Per-process counters for per-variant call rate observability. Each TP
# worker tracks its own counts; periodic summary printed every COUNTER_LOG_EVERY
# calls on rank 0.
_VARIANT_COUNTS = {"Coalesced GEMV": 0, "Coalesced GEMV-M": 0, "A.3 k=8": 0,
                   "A.3 k=4": 0, "A.1": 0, "A.2": 0, "WMMA": 0,
                   "WMMA+A.2(tail)": 0}
_VARIANT_COUNTER_LOG_EVERY = int(os.environ.get("FP8_WMMA_COUNTER_LOG_EVERY", "1000"))
_VARIANT_TOTAL = 0
_PREFIX_VARIANT_PROFILE = os.environ.get(
    "VLLM_V100_FP8_PREFIX_VARIANT_PROFILE", "0").lower() not in (
        "0", "off", "false", "")
_PREFIX_VARIANT_FILTERS = tuple(
    s.strip() for s in os.environ.get(
        "VLLM_V100_FP8_PREFIX_VARIANT_FILTER", "shared_expert").split(",")
    if s.strip())
_PREFIX_VARIANT_COUNTS = defaultdict(lambda: defaultdict(int))


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
# Stage 2C, Step A: per-rank warmup-skip + phase tagging. Each rank tracks
# its own MoE-call counter; the first N calls are discarded before stats
# recording begins, so late-decode signals are not polluted by warmup/profile-
# run or model prefill. Reporting is still rank-gated to rank 0 / -1.
_MOE_PROFILE_WARMUP_CALLS = int(os.environ.get(
    "VLLM_V100_FP8_MOE_PROFILE_WARMUP_CALLS", "200"))
# Phase tag is M-driven so we don't need to plumb max_num_seqs through the
# monkey-patch. A call counts as 'decode' iff M <= MOE_DECODE_M_MAX AND
# route_slots <= MOE_GROUPED_MAX_ROUTE_SLOTS. Anything else is 'prefill'.
# The phase label is best-effort; bucket keying below also carries M and
# route_slots so even an imperfect label keeps the shape bucket honest.
_MOE_DECODE_M_MAX = int(os.environ.get(
    "VLLM_V100_FP8_MOE_DECODE_M_MAX", "8"))
# Stage 2C, Step B: instrumentation-only torch.unique(local_expert_ids) is
# ~6% of the MoE call when enabled. Off by default; turn on only when
# debugging active-expert distribution.
_MOE_PROFILE_ACTIVE_STAT = os.environ.get(
    "VLLM_V100_FP8_MOE_PROFILE_ACTIVE_STAT", "0").lower() not in (
        "0", "off", "false", "")
# Stage 2C, Step C: reshape-based route-prep when expert_map is None.
# Replaces nonzero + local_expert_ids gather + route_w gather with
# topk_ids.reshape(-1) + a cached token_idx. The CUDA-event savings were
# real (~14 ms/token of stream work) but did NOT translate to production
# wall time on Qwen3.5-122B-A10B-FP8 TP=8 -- the CPU/GPU pipeline already
# overlapped that work, and the fast path measured ~1.3% slower in a
# warmed profile-off A/B (4 curls per leg). Default-off; keep code behind
# env flag for experimentation on different shapes / models.
_MOE_FAST_ROUTE_PREP = os.environ.get(
    "VLLM_V100_FP8_MOE_FAST_ROUTE_PREP", "0").lower() not in (
        "0", "off", "false", "")
# Process-local cache for the synthetic token_idx tensor. Keyed by
# (M, K, device-str). Allocated once on first need; reused for the rest
# of the process lifetime.
_MOE_TOKEN_IDX_CACHE = {}
_MOE_PROFILE_STATS = {}        # bucket_key: (phase, M, route_slots, grouped, fast) -> stats
_MOE_LAYER_STATS = {}          # prefix -> stats (kept across buckets for offender ranking)
_MOE_PROFILE_TOTAL_CALLS = 0   # per-rank
_MOE_PROFILE_RECORDED = 0      # per-rank: calls actually recorded into stats
# CUDA-event-timed sections (measure GPU elapsed time accurately):
_MOE_CUDA_EVENT_SECTIONS = (
    "nonzero",
    "local_expert_ids",
    "route_w_build",
    "index_select",
    "route_weight_apply",
    "hidden_contig",
    "w13_gemm",
    "activation",
    "w2_gemm",
    "scatter",
)
# Wall-timed sections (perf_counter; measures launch + Python overhead, not
# GPU elapsed; useful for residual hunting but must not be conflated with
# cuda_event_sections_ms in reporting):
_MOE_WALL_SECTIONS = (
    "out_zeros",
    "view_uint8_contig",
    "routing",
    "active_experts_stat",
    # Stage 2C, Step B: finer wall timers to attribute the unattributed
    # bucket. py_inner_loop covers the full _our_moe_apply_grouped body;
    # py_dispatch_w13/w2 cover the Python+C++ dispatch around the grouped
    # GEMM calls (CUDA-event w13_gemm/w2_gemm separately captures GPU time).
    "py_inner_loop",
    "py_dispatch_w13",
    "py_dispatch_w2",
)
_MOE_PROFILE_SECTIONS = _MOE_CUDA_EVENT_SECTIONS + _MOE_WALL_SECTIONS

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
    # Stage 3.5+ dense-FP8 diagnostic: Qwen3.6-27B-FP8 (dense+GDN) uses
    # `Qwen2MoeMLP` (via `from vllm...qwen2_moe import Qwen2MoeMLP as
    # Qwen3NextMLP` in vllm/model_executor/models/qwen3_next.py). MoE
    # variants of this model also use Qwen2MoeMLP for shared_expert,
    # which the SparseMoeBlock walk already tags as `moe_shared`; the
    # class-level attach loop must skip already-tagged instances so the
    # shared-expert bucket isn't clobbered. See `_attach_decode_breakdown_hooks`.
    "Qwen2MoeMLP": "Qwen2MoeMLP",
    "LogitsProcessor": "LogitsProcessor",
}
_BREAKDOWN_SECTION_ORDER = (
    "Qwen3NextSparseMoeBlock",
    "Qwen3NextGatedDeltaNet",
    "Qwen3NextAttention",
    "Qwen2MoeMLP",
    "LogitsProcessor",
)
# Stage 2D, Step 1: read-only attribution of the MoE-block wrapper.
# Sub-sections decompose Qwen3NextSparseMoeBlock without being double-counted
# into the per-token decode total (they live *inside* the parent section,
# whose hook captures the full block time end-to-end). We attach hooks on
# specific instance children of each Qwen3NextSparseMoeBlock and tag them
# via `module._v100_breakdown_section` (read first in the hooks), because
# the underlying torch classes (e.g. ReplicatedLinear, FusedMoE) are also
# used elsewhere and we don't want over-attaching to attention / other
# layers.
_BREAKDOWN_MOE_SUB_HOOK = os.environ.get(
    "VLLM_V100_FP8_DECODE_BREAKDOWN_MOE_SUBS", "1").lower() not in (
        "0", "off", "false", "")
# Stage 2D, Step 2A.2b: one-shot rank-0 dump of the shared_expert runtime
# structure (type, children, quant method, weight dtype/shape, whether
# child Linears go through our patched FP8 apply). Fires once on the
# first Qwen3NextSparseMoeBlock forward call so it reads AFTER PWAL has
# transformed weights to block-FP8 uint8 storage. Off by default; turn on
# to confirm shared_experts is on our FP8 path (Stage 2A.3 measurement).
_DEBUG_SHARED_EXPERTS = os.environ.get(
    "VLLM_V100_FP8_DEBUG_SHARED_EXPERTS", "0").lower() not in (
        "0", "off", "false", "")
_DEBUG_SHARED_EXPERTS_DONE = False
# Stage 2D, Step 2B.1: measurement-only sub-attribution of the moe_other
# residual (the work inside Qwen3NextSparseMoeBlock.forward that is NOT
# inside self.gate or self.experts). When enabled, monkey-patches the
# forward with CUDA-event timers bracketing:
#   moe_other_combine    -- the `(routed_out, shared_out)` -> sum step
#   moe_other_allreduce  -- maybe_all_reduce_tensor_model_parallel
#                           (or SP all-gather when use_sequence_parallel_moe)
# The remainder (Python view/reshape, attribute lookups, dispatch glue)
# falls into moe_other_residual computed in the report.
_MOE_OTHER_PROFILE = os.environ.get(
    "VLLM_V100_FP8_MOE_OTHER_PROFILE", "0").lower() not in (
        "0", "off", "false", "")
_BREAKDOWN_MOE_OTHER_SUB_ORDER = (
    "moe_other_combine",
    "moe_other_allreduce",
)
# Stage 2D, Step 2D.2: measurement-only sub-attribution of
# Qwen3NextGatedDeltaNet (GDN), the linear-attention block that
# accounts for ~30% of decode wall on Qwen3.5-122B-A10B-FP8. Source-read
# (Stage 2D Step 2D.1) showed:
#   - in_proj_qkvz, out_proj: on our patched FP8 path (heavy)
#   - in_proj_ba, conv1d: FP16 fallback (small, block-FP8 alignment mismatch)
#   - core attention: torch.ops.vllm.gdn_attention_core -> _forward_core,
#     which invokes FLA Triton kernels (causal_conv1d_update,
#     fused_sigmoid_gating_delta_rule_update) directly
# Decode call-graph: in_proj_qkvz, in_proj_ba, _forward_core, norm,
# out_proj are all direct siblings inside Qwen3NextGatedDeltaNet.forward.
# self.conv1d.forward is NEVER called at decode -- only its weight is read
# inside _forward_core -- so we do NOT hook conv1d (a hook would never
# fire). All gdn_* sub-sections are siblings; no nested accounting.
_BREAKDOWN_GDN_SUB_HOOK = os.environ.get(
    "VLLM_V100_FP8_DECODE_BREAKDOWN_GDN_SUBS", "1").lower() not in (
        "0", "off", "false", "")
_BREAKDOWN_GDN_SUB_SECTION_ORDER = (
    "gdn_in_qkvz",
    "gdn_in_ba",
    "gdn_core",
    "gdn_norm",
    "gdn_out_proj",
)
# (attr_name, section_label) tuples for the three GDN children whose
# Module.forward fires at decode. gdn_core is NOT included here because
# it is provided by a method-level monkey-patch on _forward_core, not by
# a child Module hook.
_BREAKDOWN_GDN_CHILDREN = (
    ("in_proj_qkvz", "gdn_in_qkvz"),
    ("in_proj_ba", "gdn_in_ba"),
    ("norm", "gdn_norm"),
    ("out_proj", "gdn_out_proj"),
)
# Stage 3.5+ dense-FP8 diagnostic: measurement-only sub-attribution of
# Qwen2MoeMLP (the dense MLP used by Qwen3.6-27B-FP8 etc.). The three
# direct sibling children in Qwen2MoeMLP.forward are:
#   self.gate_up_proj(x)   -> densemlp_gate_up  (MergedColumnParallelLinear; column-parallel, no AR)
#   self.act_fn(gate_up)   -> densemlp_act      (SiluAndMul; element-wise)
#   self.down_proj(out)    -> densemlp_down     (RowParallelLinear; includes tensor_model_parallel_all_reduce)
# Splitting gate_up vs down matters because down_proj's wall-time is
# GEMM + AR, while gate_up_proj is pure GEMM (modulo the FP8 dispatch
# wrapper). The cross-cutting `row_parallel_ar` bucket isolates the AR
# fraction inside down_proj; (densemlp_down - row_parallel_ar share) is
# the GEMM+dequant+Python-dispatch cost. densemlp_other (residual) is
# Python overhead, view/reshape, and dispatch glue.
_BREAKDOWN_DENSEMLP_SUB_HOOK = os.environ.get(
    "VLLM_V100_FP8_DECODE_BREAKDOWN_DENSEMLP_SUBS", "1").lower() not in (
        "0", "off", "false", "")
_BREAKDOWN_DENSEMLP_SUB_SECTION_ORDER = (
    "densemlp_gate_up",
    "densemlp_act",
    "densemlp_down",
)
_BREAKDOWN_DENSEMLP_CHILDREN = (
    ("gate_up_proj", "densemlp_gate_up"),
    ("act_fn",       "densemlp_act"),
    ("down_proj",    "densemlp_down"),
)
# Stage 2D, Step 2D.3: cross-cutting attribution of the all-reduce hidden
# inside every RowParallelLinear.forward at TP>1. Stage 2D Step 2D.2
# surprise found that gdn_out_proj (RowParallelLinear) was 47.9% of GDN,
# of which only ~10% was the FP8 GEMM and ~90% the post-GEMM
# tensor_model_parallel_all_reduce. Attention's out_proj has the same
# shape. This profile mode times ONLY the AR call inside
# RowParallelLinear.forward (NOT the GEMM, NOT the input split, NOT the
# bias post-processing), aggregated across all RowParallelLinear
# instances in the model. Reported as a CROSS-CUTTING bucket
# `row_parallel_ar`, NOT a sibling of the module sub-sections (its time
# is already counted inside gdn_out_proj, the attention bucket, etc.).
# See _breakdown_flush rendering for the explicit non-sibling annotation.
_ROW_PARALLEL_AR_PROFILE = os.environ.get(
    "VLLM_V100_FP8_ROW_PARALLEL_AR_PROFILE", "0").lower() not in (
        "0", "off", "false", "")
_BREAKDOWN_SUB_SECTION_ORDER = (
    "moe_router",
    "moe_experts",
    "moe_shared",
    # Stage 2D Step 2D.2: GDN sibling sub-sections (all direct calls in
    # Qwen3NextGatedDeltaNet.forward; no nesting).
    "gdn_in_qkvz",
    "gdn_in_ba",
    "gdn_core",
    "gdn_norm",
    "gdn_out_proj",
    # Stage 3.5+ dense-FP8 diagnostic: Qwen2MoeMLP sibling sub-sections.
    "densemlp_gate_up",
    "densemlp_act",
    "densemlp_down",
)
# Residual / nesting map for the breakdown report. Each key is a section
# whose ms/token will be decomposed by subtracting the listed sub-sections.
# CRUCIAL: only list sub-sections that are *direct sibling* calls inside
# the parent's forward body. Nested calls (a sub-section invoked from
# inside another sub-section) live under _BREAKDOWN_NESTED_OF below.
#
# For Qwen3NextSparseMoeBlock the direct siblings are:
#   self.gate(...)        -> moe_router
#   self.experts(...)     -> moe_experts (a SharedFusedMoE that internally
#                                          invokes self._shared_experts(...))
# moe_shared is therefore NESTED inside moe_experts, not a sibling of it.
# Subtracting moe_shared at the parent level would double-count (since
# moe_experts already includes it), inflating the apparent moe_other
# residual by exactly the moe_shared time. See Stage 2D Step 2A source
# read in docs/SESSION_LOG.md for the call-graph trace.
#
# For Qwen3NextGatedDeltaNet the direct siblings are in_proj_qkvz,
# in_proj_ba, _forward_core (tagged gdn_core via class-method monkey-patch,
# not a child Module forward hook), norm, out_proj. None of these are
# nested. self.conv1d.forward is NEVER called at decode (only its weight
# is read inside _forward_core), so it is intentionally absent.
_BREAKDOWN_RESIDUAL_OF = {
    "Qwen3NextSparseMoeBlock": ("moe_router", "moe_experts"),
    "Qwen3NextGatedDeltaNet": _BREAKDOWN_GDN_SUB_SECTION_ORDER,
    "Qwen2MoeMLP": _BREAKDOWN_DENSEMLP_SUB_SECTION_ORDER,
}
_BREAKDOWN_NESTED_OF = {
    # parent_section -> tuple of sub-sections invoked from INSIDE the
    # parent's measurement window. Reporter renders these indented under
    # the parent and reports their share-of-parent, NOT share-of-block.
    "moe_experts": ("moe_shared",),
}
# Per-section label for the residual bucket printed under each parent's
# breakdown (the time inside the parent's window that is NOT covered by
# any sibling sub-section). Defaults to "<section>_other" if unspecified.
_BREAKDOWN_RESIDUAL_LABEL = {
    "Qwen3NextSparseMoeBlock": "moe_other (wrapper/topk/combine)",
    "Qwen3NextGatedDeltaNet":  "gdn_other (rearrange/cat/residual)",
    "Qwen2MoeMLP":             "densemlp_other (Python/dispatch)",
}
# Which named-child attributes of a Qwen3NextSparseMoeBlock to hook, and
# what section label to assign. The first present name wins. Qwen3-Next
# uses singular `shared_expert`; we keep the plural `shared_experts`
# fallback in case future architecture variants change the name.
_BREAKDOWN_SPARSE_MOE_CHILDREN = (
    ("gate", "moe_router"),
    ("experts", "moe_experts"),
    ("shared_expert", "moe_shared"),
    ("shared_experts", "moe_shared"),
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
_BREAKDOWN_RUNTIME_DISABLED = False


def _breakdown_rank():
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return -1


def _breakdown_disable_runtime(reason):
    """Fail closed if CUDA-event profiling is unsafe for this runtime mode."""
    global _BREAKDOWN_RUNTIME_DISABLED, _BREAKDOWN_TOKEN_OPEN, _BREAKDOWN_TOKEN_START
    if not _BREAKDOWN_RUNTIME_DISABLED and _breakdown_rank() in (0, -1):
        print(
            f"[DECODE-BREAKDOWN rank={_breakdown_rank()} pid={os.getpid()}] "
            f"disabled at runtime: {reason}",
            flush=True,
        )
    _BREAKDOWN_RUNTIME_DISABLED = True
    _BREAKDOWN_PENDING_EVENTS.clear()
    _BREAKDOWN_TOKEN_EVENTS.clear()
    _BREAKDOWN_TOKEN_OPEN = False
    _BREAKDOWN_TOKEN_START = None


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


def _breakdown_section_for(module):
    """Resolve the section label for a hooked module.

    Stage 2D, Step 1: prefer the instance-tagged `_v100_breakdown_section`
    attribute so we can attach sub-section hooks to specific submodule
    instances (e.g. `Qwen3NextSparseMoeBlock.gate` -> 'moe_router') without
    over-attaching every same-class module in the model.
    """
    section = getattr(module, "_v100_breakdown_section", None)
    if section is not None:
        return section
    cls_name = type(module).__name__
    return _BREAKDOWN_HOOK_CLASSES.get(cls_name, cls_name)


def _breakdown_pre_hook(module, args, kwargs):
    if _BREAKDOWN_RUNTIME_DISABLED or not torch.cuda.is_available():
        return
    # CUDA graph capture forbids cudaEventRecord on the captured stream
    # (and `.elapsed_time()` in the post-hook would also be illegal).
    # Skip silently during capture so the same binary works under both
    # eager and cudagraph; data is only collected from non-captured
    # forwards (warmup, profile_run, fallback shapes, --enforce-eager).
    if torch.cuda.is_current_stream_capturing():
        return
    section = _breakdown_section_for(module)
    m = _first_tensor_m(args, kwargs)
    regime = "decode" if m == 1 else "prefill"
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    stack = getattr(module, "_v100_breakdown_stack", None)
    if stack is None:
        stack = []
        setattr(module, "_v100_breakdown_stack", stack)
    stack.append((regime, start))

    # Sub-section hooks fire *inside* a parent section's measurement window;
    # they must not also re-open the per-token wall (which is opened by the
    # first non-LogitsProcessor parent section per token).
    global _BREAKDOWN_TOKEN_OPEN, _BREAKDOWN_TOKEN_START
    is_sub = section in _BREAKDOWN_SUB_SECTION_ORDER
    if (regime == "decode" and section != "LogitsProcessor" and not is_sub
            and not _BREAKDOWN_TOKEN_OPEN):
        _BREAKDOWN_TOKEN_START = torch.cuda.Event(enable_timing=True)
        _BREAKDOWN_TOKEN_START.record()
        _BREAKDOWN_TOKEN_OPEN = True


def _breakdown_post_hook(module, args, kwargs, output):
    if _BREAKDOWN_RUNTIME_DISABLED or not torch.cuda.is_available():
        return
    # Symmetric guard to _breakdown_pre_hook: under capture the pre-hook
    # returned without pushing a stack frame, so `stack` is empty and we
    # would no-op anyway, but check explicitly to avoid any `.record()`
    # call on a captured stream if hook ordering ever changes.
    if torch.cuda.is_current_stream_capturing():
        return
    stack = getattr(module, "_v100_breakdown_stack", None)
    if not stack:
        return
    regime, start = stack.pop()
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    section = _breakdown_section_for(module)
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
    interval_totals = defaultdict(float)
    interval_counts = defaultdict(int)
    try:
        torch.cuda.synchronize()
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
    except Exception as exc:
        _breakdown_disable_runtime(
            f"{type(exc).__name__} during CUDA event timing; "
            "profiling is eager-only under cudagraph/replay")
        return

    if _breakdown_rank() not in (0, -1):
        return
    tokens = max(1, _BREAKDOWN_DECODE_TOKENS)
    if total_ms <= 0.0:
        # Token-wall events weren't captured (the first token's worth of
        # data was thrown away, or no LogitsProcessor fired). Fall back to
        # the sum of *parent* section times -- never sum in sub-sections,
        # which would double-count their parent.
        total_ms = sum(
            interval_totals.get(name, 0.0)
            for name in _BREAKDOWN_SECTION_ORDER
        )
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

        # Stage 2D, Step 1: indent sub-section breakdown under its parent.
        # The sub-section times live *inside* the parent's measurement
        # window; report them as a breakdown of the parent, NOT as new
        # entries in the per-token total.
        # Stage 2D, Step 2A: also indent NESTED sub-sections under their
        # immediate parent sub-section (e.g. moe_shared lives inside
        # moe_experts because SharedFusedMoE.forward invokes
        # self._shared_experts(...) directly).
        sub_names = _BREAKDOWN_RESIDUAL_OF.get(section)
        if not sub_names:
            continue
        sub_sum_ms = 0.0
        sub_lines = []
        for sub in sub_names:
            sub_ms = interval_totals.get(sub, 0.0)
            if sub_ms <= 0.0:
                continue
            sub_per_token = sub_ms / max(1, _DECODE_BREAKDOWN_EVERY)
            sub_sum_ms += sub_per_token
            # share-of-parent (not share-of-total), so the breakdown sums
            # to ~100% within the MoE block alone.
            sub_pct = (100.0 * sub_per_token / per_token) if per_token else 0.0
            sub_calls = interval_counts.get(sub, 0)
            sub_lines.append(
                f"    + {sub:<26} {sub_per_token:8.3f} ms/token "
                f"({sub_pct:5.1f}% of {section}) calls={sub_calls}")

            # Render any nested sub-sections (e.g. moe_shared inside
            # moe_experts) at one more indent level. Their time is
            # already inside `sub`'s window, so we subtract their sum
            # from `sub` to compute the routed-only residual (e.g. our
            # _our_moe_apply path inside SharedFusedMoE).
            nested_names = _BREAKDOWN_NESTED_OF.get(sub)
            if not nested_names:
                continue
            nested_sum_ms = 0.0
            for nested in nested_names:
                nested_ms = interval_totals.get(nested, 0.0)
                if nested_ms <= 0.0:
                    continue
                nested_per_token = nested_ms / max(1, _DECODE_BREAKDOWN_EVERY)
                nested_sum_ms += nested_per_token
                nested_pct = (
                    100.0 * nested_per_token / sub_per_token
                ) if sub_per_token else 0.0
                nested_calls = interval_counts.get(nested, 0)
                sub_lines.append(
                    f"        +-- {nested:<22} {nested_per_token:8.3f} ms/token "
                    f"({nested_pct:5.1f}% of {sub}) calls={nested_calls}")
            sub_routed_only = max(0.0, sub_per_token - nested_sum_ms)
            sub_routed_pct = (
                100.0 * sub_routed_only / sub_per_token) if sub_per_token else 0.0
            sub_lines.append(
                f"        +-- {sub + ' (excl. nested)':<22} "
                f"{sub_routed_only:8.3f} ms/token "
                f"({sub_routed_pct:5.1f}% of {sub})")
        other_in_parent = max(0.0, per_token - sub_sum_ms)
        other_pct_parent = (
            100.0 * other_in_parent / per_token) if per_token else 0.0
        # Stage 2D Step 2D.2: per-section residual label so each parent
        # gets a clean "*_other" line (moe_other for MoE, gdn_other for
        # GDN, etc.). Falls back to a generic name if unmapped.
        residual_label = _BREAKDOWN_RESIDUAL_LABEL.get(
            section, f"{section}_other (residual)")
        sub_lines.append(
            f"    + {residual_label:<26} "
            f"{other_in_parent:8.3f} ms/token "
            f"({other_pct_parent:5.1f}% of {section})")
        # Stage 2D Step 2B.1: indent measurement-only sub-attribution of
        # the moe_other residual when the patched Qwen3NextSparseMoeBlock
        # forward is installed. moe_other_residual = moe_other - subs;
        # represents Python wrapper + view/reshape + dispatch glue we
        # didn't bracket explicitly. Only applies to Qwen3NextSparseMoeBlock;
        # GDN's residual is left as a single bucket since 2D.2 source-read
        # showed gdn_other should be small (rearrange/cat/Python only).
        if section == "Qwen3NextSparseMoeBlock":
            moe_other_sub_total = 0.0
            moe_other_sub_lines = []
            for sub in _BREAKDOWN_MOE_OTHER_SUB_ORDER:
                sub_ms = interval_totals.get(sub, 0.0)
                if sub_ms <= 0.0:
                    continue
                sub_per_token = sub_ms / max(1, _DECODE_BREAKDOWN_EVERY)
                moe_other_sub_total += sub_per_token
                sub_pct_other = (
                    100.0 * sub_per_token / other_in_parent
                ) if other_in_parent else 0.0
                sub_calls = interval_counts.get(sub, 0)
                moe_other_sub_lines.append(
                    f"        +-- {sub:<22} {sub_per_token:8.3f} ms/token "
                    f"({sub_pct_other:5.1f}% of moe_other) calls={sub_calls}")
            if moe_other_sub_total > 0:
                moe_other_residual = max(0.0, other_in_parent - moe_other_sub_total)
                moe_other_residual_pct = (
                    100.0 * moe_other_residual / other_in_parent
                ) if other_in_parent else 0.0
                moe_other_sub_lines.append(
                    f"        +-- moe_other_residual    "
                    f"{moe_other_residual:8.3f} ms/token "
                    f"({moe_other_residual_pct:5.1f}% of moe_other)")
                sub_lines.extend(moe_other_sub_lines)
        lines.extend(sub_lines)
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

    # Stage 2D Step 2D.3: cross-cutting attribution buckets. These are
    # AGGREGATE timings across every RowParallelLinear instance in the
    # model -- they do NOT correspond to a single named module, and the
    # time they measure is ALREADY counted inside the gdn_out_proj,
    # Qwen3NextAttention, and (where applicable) other module-level
    # buckets above. They are NOT summed into the per-token total; they
    # are printed below the breakdown with an explicit "(cross-cutting,
    # already counted)" annotation so future readers don't double-count.
    ar_ms = interval_totals.get("row_parallel_ar", 0.0)
    if ar_ms > 0.0:
        ar_per_token = ar_ms / max(1, _DECODE_BREAKDOWN_EVERY)
        ar_calls = interval_counts.get("row_parallel_ar", 0)
        ar_pct_total = (
            100.0 * ar_per_token / total_per_token
        ) if total_per_token else 0.0
        print(
            f"  [cross-cutting attribution; already counted in module "
            f"buckets above]",
            flush=True,
        )
        print(
            f"  {'row_parallel_ar':<30} {ar_per_token:8.3f} ms/token "
            f"({ar_pct_total:5.1f}% of total) calls={ar_calls}",
            flush=True,
        )


def _attach_shared_experts_debug_hook(model):
    """Stage 2D, Step 2A.2b: register a one-shot rank-0 hook on the first
    Qwen3NextSparseMoeBlock found in `model`. Fires once on first forward
    (post-PWAL), dumps the shared-expert structure, then short-circuits.

    Independent of `_DECODE_BREAKDOWN`: this attach runs even if the
    decode breakdown is disabled, so the debug dump is cheap to enable
    on its own (single profile-off serve, single curl).
    """
    if not _DEBUG_SHARED_EXPERTS:
        return
    model_id = id(model)
    # Reuse the decode-breakdown attached-set: we attach to the same
    # model anchor, but the debug-only hook is a different callable so
    # double-attach is harmless. Tracking just to avoid double work.
    target = None
    target_prefix = "<unknown>"
    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3NextSparseMoeBlock":
            target = module
            target_prefix = name or "<unknown>"
            break
    if target is None:
        return

    state = {"fired": False}

    def _one_shot(mod, args, kwargs):
        if state["fired"]:
            return
        state["fired"] = True
        try:
            _dump_shared_experts_structure(mod, target_prefix)
        except Exception as exc:
            print(
                f"[V100-FP8-DBG-SHARED rank={_breakdown_rank()} "
                f"pid={os.getpid()}] dump_failed={type(exc).__name__}: {exc}",
                flush=True,
            )

    target.register_forward_pre_hook(_one_shot, with_kwargs=True)


def _dump_shared_experts_structure(layer_module, layer_prefix):
    """Stage 2D, Step 2A.2b: one-shot rank-0 dump of the shared_expert
    runtime structure. Fires once per process; subsequent calls are skipped.

    Purpose: definitively classify the shared-expert path as one of
      (a) already-FP8 (child Linears use Fp8LinearMethod + our patched apply)
      (b) bypass (child Linears use a non-FP8 quant method, e.g. FP16)
      (c) hybrid (some children FP8, some not)

    The decision drives Stage 2D Step 2B: if (a), shared experts are
    structural M=1 memory-bound work and we pivot Step 2B to GDN or the
    moe_other wrapper bucket; if (b) or (c), there's an optimization
    target we hadn't measured yet.
    """
    global _DEBUG_SHARED_EXPERTS_DONE
    if not _DEBUG_SHARED_EXPERTS or _DEBUG_SHARED_EXPERTS_DONE:
        return
    if _breakdown_rank() not in (0, -1):
        # Mark done on this rank too so we don't keep firing.
        _DEBUG_SHARED_EXPERTS_DONE = True
        return
    _DEBUG_SHARED_EXPERTS_DONE = True

    rank = _breakdown_rank()
    pid = os.getpid()
    tag = f"[V100-FP8-DBG-SHARED rank={rank} pid={pid}]"

    shared = getattr(layer_module, "shared_expert", None)
    if shared is None:
        shared = getattr(layer_module, "shared_experts", None)
    if shared is None:
        print(
            f"{tag} layer={layer_prefix} shared_expert=None "
            f"(model has no shared experts)",
            flush=True,
        )
        return

    # Resolve whether Fp8LinearMethod.apply has been swapped to our patch.
    apply_is_patched = "?"
    fp8_method_cls = None
    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
        fp8_method_cls = Fp8LinearMethod
        apply_qual = getattr(
            Fp8LinearMethod.apply, "__qualname__", "")
        apply_is_patched = "patched_apply" in apply_qual
    except Exception as exc:
        print(f"{tag} fp8_introspect_failed={type(exc).__name__}: {exc}",
              flush=True)

    print(f"{tag} layer={layer_prefix}", flush=True)
    print(f"{tag}   type(shared_expert)={type(shared).__name__} "
          f"module={type(shared).__module__}", flush=True)
    print(f"{tag}   Fp8LinearMethod.apply patched_by_v100={apply_is_patched}",
          flush=True)

    for child_name, child in shared.named_children():
        child_cls = type(child).__name__
        print(f"{tag}   child={child_name} type={child_cls}", flush=True)
        qm = getattr(child, "quant_method", None)
        if qm is not None:
            qm_cls = type(qm).__name__
            is_fp8 = (fp8_method_cls is not None
                      and isinstance(qm, fp8_method_cls))
            block_quant = getattr(qm, "block_quant", None)
            block_size = getattr(qm, "weight_block_size", None)
            print(
                f"{tag}     quant_method={qm_cls} "
                f"is_fp8={is_fp8} block_quant={block_quant} "
                f"block_size={block_size}",
                flush=True,
            )
        for attr in ("weight", "weight_scale_inv", "weight_scale",
                     "input_scale"):
            t = getattr(child, attr, None)
            if t is None or not isinstance(t, torch.Tensor):
                continue
            print(
                f"{tag}     {attr}: dtype={t.dtype} shape={tuple(t.shape)} "
                f"contiguous={t.is_contiguous()}",
                flush=True,
            )


def _attach_decode_breakdown_hooks(model):
    if not _DECODE_BREAKDOWN or not torch.cuda.is_available():
        return
    model_id = id(model)
    if model_id in _BREAKDOWN_ATTACHED_MODEL_IDS:
        return
    _BREAKDOWN_ATTACHED_MODEL_IDS.add(model_id)

    def _attach(module, section_label):
        # Tag the instance so the hook can read the role-specific section
        # without depending on the module's class name (which may be shared
        # with unrelated modules elsewhere in the model).
        module._v100_breakdown_section = section_label
        module.register_forward_pre_hook(_breakdown_pre_hook, with_kwargs=True)
        module.register_forward_hook(_breakdown_post_hook, with_kwargs=True)

    attached = defaultdict(int)
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name not in _BREAKDOWN_HOOK_CLASSES:
            continue
        # Skip if a parent walk already tagged this instance with a more
        # specific role. Critical for Qwen3.5/3.6 MoE variants where the
        # SparseMoeBlock walk tags its shared_expert (a Qwen2MoeMLP) as
        # `moe_shared`; without this guard the class-level attach would
        # double-register hooks AND clobber the tag back to
        # `Qwen2MoeMLP`, attributing shared-expert work to the dense
        # bucket. Module preorder DFS guarantees the parent block is
        # visited before its children.
        if getattr(module, "_v100_breakdown_section", None) is not None:
            continue
        section = _BREAKDOWN_HOOK_CLASSES[cls_name]
        _attach(module, section)
        attached[section] += 1

        # Stage 2D, Step 1: read-only sub-attribution of the MoE block.
        # For each Qwen3NextSparseMoeBlock instance, hook specific child
        # attributes (gate, experts, shared_experts) with role-named
        # sub-sections. moe_other is computed at report time as the residual
        # of the parent block minus these children.
        if cls_name == "Qwen3NextSparseMoeBlock" and _BREAKDOWN_MOE_SUB_HOOK:
            for attr_name, sub_section in _BREAKDOWN_SPARSE_MOE_CHILDREN:
                child = getattr(module, attr_name, None)
                if child is None or not isinstance(child, torch.nn.Module):
                    continue
                # Avoid double-attach if a child is shared across blocks.
                if getattr(child, "_v100_breakdown_section", None) is not None:
                    continue
                _attach(child, sub_section)
                attached[sub_section] += 1
            continue

        # Stage 2D, Step 2D.2: read-only sub-attribution of GDN. Hook the
        # 4 child Modules whose forward fires per call; gdn_core is
        # handled by a separate class-method monkey-patch on _forward_core
        # (see _patch_vllm_for_v100 below) so it appears alongside these.
        if cls_name in ("Qwen3NextGatedDeltaNet", "Qwen3_5GatedDeltaNet") \
                and _BREAKDOWN_GDN_SUB_HOOK:
            for attr_name, sub_section in _BREAKDOWN_GDN_CHILDREN:
                child = getattr(module, attr_name, None)
                if child is None or not isinstance(child, torch.nn.Module):
                    continue
                if getattr(child, "_v100_breakdown_section", None) is not None:
                    continue
                _attach(child, sub_section)
                attached[sub_section] += 1
            continue

        # Stage 3.5+ dense-FP8 diagnostic: read-only sub-attribution of
        # Qwen2MoeMLP (dense MLP). Splits gate_up_proj (column-parallel,
        # no AR) vs act_fn vs down_proj (row-parallel, includes AR).
        # The class-level skip above guarantees we don't re-walk a
        # shared_expert Qwen2MoeMLP (already tagged moe_shared).
        if cls_name == "Qwen2MoeMLP" and _BREAKDOWN_DENSEMLP_SUB_HOOK:
            for attr_name, sub_section in _BREAKDOWN_DENSEMLP_CHILDREN:
                child = getattr(module, attr_name, None)
                if child is None or not isinstance(child, torch.nn.Module):
                    continue
                if getattr(child, "_v100_breakdown_section", None) is not None:
                    continue
                _attach(child, sub_section)
                attached[sub_section] += 1
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
        and block_w == 128
        and (block_h == 128 or (block_h == 1 and _CT_CHANNEL_WMMA))
    )
    coalesced_gemv_ok = (
        _COALESCED_GEMV
        and _HAS_COALESCED_GEMV
        and block_w == 128
        and block_h in (1, 128)
        and K % 128 == 0
        and M <= _COALESCED_GEMV_M_MAX
    )

    if coalesced_gemv_ok and M == 1:
        return _ext.fp8_w8a16_gemv_coalesced(
            x2d, weight_u8, scales, N, K, block_h, block_w), "Coalesced GEMV"
    if coalesced_gemv_ok and M <= 8 and _HAS_COALESCED_GEMV_M:
        return _ext.fp8_w8a16_gemv_coalesced_m(
            x2d, weight_u8, scales, N, K, block_h, block_w), "Coalesced GEMV-M"
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
    prefix = getattr(layer, "prefix", "<?>")
    if _PREFIX_VARIANT_PROFILE:
        _PREFIX_VARIANT_COUNTS[prefix][variant] += 1
    if _VARIANT_TOTAL % _VARIANT_COUNTER_LOG_EVERY == 0:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            tot = _VARIANT_TOTAL
            parts = [f"{k}={v} ({100.0*v/tot:.0f}%)"
                     for k, v in _VARIANT_COUNTS.items() if v]
            print(f"[serve_fp8_v100 pid={os.getpid()}] kernel variant counts after "
                  f"{tot} calls: {', '.join(parts)}", flush=True)
            if _PREFIX_VARIANT_PROFILE:
                for pfx, counts in sorted(_PREFIX_VARIANT_COUNTS.items()):
                    if (_PREFIX_VARIANT_FILTERS and
                            not any(f in pfx for f in _PREFIX_VARIANT_FILTERS)):
                        continue
                    pfx_tot = sum(counts.values())
                    pfx_parts = [f"{k}={v}" for k, v in sorted(counts.items())]
                    print(
                        f"[serve_fp8_v100 pid={os.getpid()}] prefix variants "
                        f"{pfx}: total={pfx_tot} {', '.join(pfx_parts)}",
                        flush=True,
                    )

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


def _get_token_idx_cached(M, topk, device):
    """Stage 2C, Step C: return the synthetic token-row index for the
    grouped fast path. Length is M*topk. For M==1 this is all zeros;
    for M>1 it is [0,0,...,0,1,1,...,1,M-1,M-1,...] (each token id
    repeated topk times).

    Cached per (M, topk, device-str) so we never allocate at decode.
    """
    key = (M, topk, str(device))
    t = _MOE_TOKEN_IDX_CACHE.get(key)
    if t is not None:
        return t
    if M == 1:
        t = torch.zeros(topk, dtype=torch.int64, device=device)
    else:
        t = (torch.arange(M, dtype=torch.int64, device=device)
             .repeat_interleave(topk))
    _MOE_TOKEN_IDX_CACHE[key] = t
    return t


def _get_layer_uint8_weights(layer):
    """Stage 2C, Step D: cache the uint8 contiguous view of w13/w2 weights
    on the layer. Today these are zero-copy aliases (confirmed by the
    grouped log line); the cost we eliminate is the Python `view` +
    `.contiguous()` call chain on every MoE invocation.

    Stored on the layer object itself (which lives for the serve lifetime).
    """
    w13 = getattr(layer, "_v100_w13_u8", None)
    if w13 is None:
        w13 = layer.w13_weight.view(torch.uint8).contiguous()
        layer._v100_w13_u8 = w13
    w2 = getattr(layer, "_v100_w2_u8", None)
    if w2 is None:
        w2 = layer.w2_weight.view(torch.uint8).contiguous()
        layer._v100_w2_u8 = w2
    return w13, w2


def _get_layer_expert_map_dev(layer, device):
    """Stage 2C, Step D: cache `expert_map.to(device=...)` on the layer.

    Returns the cached device-resident expert_map, or None if the layer
    has no expert_map (TP-replicated experts; current Qwen3.5-122B-A10B-FP8
    behavior). Cache is invalidated only if the device of the cached
    tensor differs from the request, which should never happen in steady
    state.
    """
    cached = getattr(layer, "_v100_expert_map_dev", None)
    em = getattr(layer, "expert_map", None)
    if em is None:
        return None
    if cached is not None and cached.device == device:
        return cached
    cached = em.to(device=device)
    layer._v100_expert_map_dev = cached
    return cached


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
    timed_wall,
    w13_k_split,
    w2_k_split,
):
    """Grouped-routed MoE path: one w13 GEMM and one w2 GEMM per layer call.

    Two route-prep paths:
      - Fast path (`_MOE_FAST_ROUTE_PREP=1` and `expert_map is None`):
        reshape topk_ids/topk_weights directly. No `nonzero`, no
        `valid_mask`, no per-call `token_idx` allocation. Dense top-k is
        the invariant.
      - Fallback: existing `valid_mask -> nonzero -> gather` path, used
        when expert_map filters routes (EP-partitioned models) or when
        FAST_ROUTE_PREP is explicitly disabled.
    """
    global _MOE_GROUPED_LOGGED

    if profile:
        inner_t0 = time.perf_counter()

    M = x_work.size(0)
    expert_map_present = getattr(layer, "expert_map", None) is not None
    use_fast = (_MOE_FAST_ROUTE_PREP and not expert_map_present)
    if call_stats is not None:
        call_stats["used_fast_path"] = use_fast

    if use_fast:
        # local_topk == topk_ids.to(long) (caller skipped expert_map remap).
        if local_topk.dim() >= 2:
            topk = local_topk.size(-1)
        else:
            topk = 1
        route_count = M * topk
        if profile:
            local_expert_ids = timed_cuda(
                "local_expert_ids",
                lambda: local_topk.reshape(-1).to(torch.int64).contiguous())
            route_w = timed_cuda(
                "route_w_build",
                lambda: topk_weights.reshape(-1).to(torch.float16))
        else:
            local_expert_ids = (
                local_topk.reshape(-1).to(torch.int64).contiguous())
            route_w = topk_weights.reshape(-1).to(torch.float16)
        token_idx = _get_token_idx_cached(M, topk, x_work.device)
    else:
        valid_mask = local_topk >= 0
        if profile:
            token_idx, route_idx = timed_cuda(
                "nonzero", lambda: torch.nonzero(valid_mask, as_tuple=True))
        else:
            token_idx, route_idx = torch.nonzero(valid_mask, as_tuple=True)
        route_count = int(token_idx.numel())
        if route_count == 0:
            if profile:
                out = timed_wall(
                    "out_zeros",
                    lambda: torch.zeros(
                        (M, hidden_size),
                        dtype=torch.float16, device=x.device))
                call_stats["sections"]["py_inner_loop"] += (
                    time.perf_counter() - inner_t0) * 1000.0
                return out
            return torch.zeros((M, hidden_size),
                               dtype=torch.float16, device=x.device)
        if profile:
            local_expert_ids = timed_cuda(
                "local_expert_ids",
                lambda: local_topk[token_idx, route_idx]
                    .to(torch.int64).contiguous())
            route_w = timed_cuda(
                "route_w_build",
                lambda: topk_weights[token_idx, route_idx].to(torch.float16))
        else:
            local_expert_ids = (
                local_topk[token_idx, route_idx]
                .to(torch.int64).contiguous())
            route_w = topk_weights[token_idx, route_idx].to(torch.float16)

    if profile:
        out = timed_wall(
            "out_zeros",
            lambda: torch.zeros(
                (M, hidden_size), dtype=torch.float16, device=x.device))
    else:
        out = torch.zeros(
            (M, hidden_size), dtype=torch.float16, device=x.device)

    if profile and _MOE_PROFILE_ACTIVE_STAT:
        # Instrumentation-only: torch.unique implies a CUDA sync. Gated
        # off by default in Stage 2C, Step B.
        t0 = time.perf_counter()
        call_stats["active_experts"] += int(
            torch.unique(local_expert_ids).numel())
        call_stats["sections"]["active_experts_stat"] += (
            time.perf_counter() - t0) * 1000.0
        call_stats["skipped_experts"] += (
            int(layer.local_num_experts) - call_stats["active_experts"])
    if profile:
        call_stats["routed_items"] += route_count

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
    if profile:
        w13_weight, w2_weight = timed_wall(
            "view_uint8_contig", lambda: _get_layer_uint8_weights(layer))
    else:
        w13_weight, w2_weight = _get_layer_uint8_weights(layer)

    if _MOE_GROUPED_LOG_ONCE and not _MOE_GROUPED_LOGGED:
        _MOE_GROUPED_LOGGED = True
        try:
            rank = _moe_profile_rank()
        except Exception:
            rank = -1
        # The view_uint8 cache returns the same tensor every call after
        # first; we still verify zero-copy invariant against the original.
        w13_zero_copy = (
            w13_weight.data_ptr() == layer.w13_weight.data_ptr()
            and w13_weight.is_contiguous()
        )
        w2_zero_copy = (
            w2_weight.data_ptr() == layer.w2_weight.data_ptr()
            and w2_weight.is_contiguous()
        )
        print(
            f"[V100-FP8-MOE-GROUPED rank={rank} pid={os.getpid()}] "
            f"enabled prefix={getattr(layer, 'prefix', '<unknown>')} "
            f"M={M} route_slots={local_topk.numel()} "
            f"route_count={route_count} hidden={hidden_size} "
            f"intermediate={intermediate} block=({block_h},{block_w}) "
            f"w13_k_split={w13_k_split} w2_k_split={w2_k_split} "
            f"k_split_env={_MOE_GROUPED_K_SPLIT} "
            f"max_route_slots={_MOE_GROUPED_MAX_ROUTE_SLOTS} "
            f"w13_u8_zero_copy={w13_zero_copy} "
            f"w2_u8_zero_copy={w2_zero_copy} "
            f"fast_route_prep={use_fast} "
            f"expert_map={'present' if expert_map_present else 'none'}",
            flush=True,
        )

    def run_w13_grouped():
        # Decode w13: the grouped coalesced GEMV (warp owns (routed-row, col),
        # lanes stride consecutive K, warp-reduce) replaces the N-strided a3
        # kernel. This path is decode-only (route_slots <= 32) so gridDim.y=R is
        # tiny — no 65535 grid-cap or per-row-GEMV-prefill concern. K=hidden_size
        # must be %128 (kernel constraint); block is already (128,128) here.
        if (_MOE_W13_COALESCED and _HAS_GROUPED_COALESCED_GEMV
                and hidden_size % 128 == 0):
            if not _MOE_W13_COAL_ENGAGED[0] and _moe_profile_rank() in (0, -1):
                _MOE_W13_COAL_ENGAGED[0] = True
                print("[V100-FP8-MOE-GROUPED] decode w13 = grouped COALESCED "
                      "GEMV (warp->col, lanes->K); first call", flush=True)
            return _ext.fp8_w8a16_grouped_gemv_coalesced(
                route_x,
                local_expert_ids,
                w13_weight,
                w13_scale,
                2 * intermediate,
                hidden_size,
                block_h,
                block_w,
            )
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
        w13 = timed_wall(
            "py_dispatch_w13",
            lambda: timed_cuda("w13_gemm", run_w13_grouped))
    else:
        w13 = run_w13_grouped()
    gate = w13[:, :intermediate]
    up = w13[:, intermediate:]

    if profile:
        hidden = timed_cuda(
            "activation", lambda: _moe_activation(layer, gate) * up)
    else:
        hidden = _moe_activation(layer, gate) * up
    if profile:
        hidden = timed_cuda("hidden_contig", lambda: hidden.contiguous())
    else:
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
        expert_out = timed_wall(
            "py_dispatch_w2",
            lambda: timed_cuda("w2_gemm", run_w2_grouped))
    else:
        expert_out = run_w2_grouped()
    if not apply_weight_on_input:
        if profile:
            expert_out = timed_cuda(
                "route_weight_apply",
                lambda: expert_out * route_w.unsqueeze(-1))
        else:
            expert_out = expert_out * route_w.unsqueeze(-1)
    if profile:
        timed_cuda("scatter", lambda: out.index_add_(0, token_idx, expert_out))
    else:
        out.index_add_(0, token_idx, expert_out)

    if profile:
        call_stats["sections"]["py_inner_loop"] += (
            time.perf_counter() - inner_t0) * 1000.0
    return out


def _moe_profile_rank():
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return -1


def _new_moe_call_stats():
    """Per-call stats accumulator. Reset at the top of each _our_moe_apply call."""
    return {
        "wall_ms": 0.0,
        "active_experts": 0,
        "routed_items": 0,
        "empty_expert_iters": 0,
        "skipped_experts": 0,
        "sections": {name: 0.0 for name in _MOE_PROFILE_SECTIONS},
    }


def _new_moe_profile_stats():
    """Aggregate stats bucket. Used for both per-bucket and per-layer rollups."""
    return {
        "calls": 0,
        "wall_ms": 0.0,
        "active_experts": 0,
        "routed_items": 0,
        "empty_expert_iters": 0,
        "skipped_experts": 0,
        "active_hist": {},
        "sections": {name: 0.0 for name in _MOE_PROFILE_SECTIONS},
    }


def _moe_phase_tag(M, route_slots):
    """Best-effort phase categorization. See STAGE_2C_PLAN.md, Step A."""
    if M <= _MOE_DECODE_M_MAX and route_slots <= _MOE_GROUPED_MAX_ROUTE_SLOTS:
        return "decode"
    return "prefill"


def _moe_profile_update(prefix, bucket_key, call_stats):
    """Stage 2C, Step A: per-rank warmup skip + per-bucket + per-layer rollup.

    bucket_key is (phase, M, route_slots, grouped, fast). Reports also include
    per-layer top-3 offenders so we keep visibility into hot layers regardless
    of which shape bucket they fall into.
    """
    global _MOE_PROFILE_TOTAL_CALLS, _MOE_PROFILE_RECORDED

    _MOE_PROFILE_TOTAL_CALLS += 1
    if _MOE_PROFILE_TOTAL_CALLS <= _MOE_PROFILE_WARMUP_CALLS:
        return  # Warmup window: do not pollute decode-only stats.

    _MOE_PROFILE_RECORDED += 1

    bucket_stats = _MOE_PROFILE_STATS.setdefault(
        bucket_key, _new_moe_profile_stats())
    layer_stats = _MOE_LAYER_STATS.setdefault(
        prefix, _new_moe_profile_stats())

    for stats in (bucket_stats, layer_stats):
        stats["calls"] += 1
        stats["wall_ms"] += call_stats["wall_ms"]
        stats["active_experts"] += call_stats["active_experts"]
        stats["routed_items"] += call_stats["routed_items"]
        stats["empty_expert_iters"] += call_stats["empty_expert_iters"]
        stats["skipped_experts"] += call_stats["skipped_experts"]
        hist = stats["active_hist"]
        active = call_stats["active_experts"]
        hist[active] = hist.get(active, 0) + 1
        for name in _MOE_PROFILE_SECTIONS:
            stats["sections"][name] += call_stats["sections"].get(name, 0.0)

    if (_MOE_PROFILE_RECORDED % _MOE_PROFILE_EVERY) != 0:
        return
    if _moe_profile_rank() not in (0, -1):
        return

    rank = _moe_profile_rank()
    pid = os.getpid()
    print(
        f"[V100-FP8-MOE-PROFILE rank={rank} pid={pid}] "
        f"warmup_skip={_MOE_PROFILE_WARMUP_CALLS} "
        f"recorded_calls={_MOE_PROFILE_RECORDED} "
        f"total_calls={_MOE_PROFILE_TOTAL_CALLS} "
        f"decode_m_max={_MOE_DECODE_M_MAX}",
        flush=True,
    )

    # Per-bucket: print the top-K buckets by wall_ms so we don't drown the
    # log when many shape combinations appear during a long run.
    top_buckets = sorted(
        _MOE_PROFILE_STATS.items(),
        key=lambda item: item[1]["wall_ms"],
        reverse=True,
    )[:6]
    for key, stats in top_buckets:
        _moe_profile_print_bucket(rank, pid, key, stats)

    top_layers = sorted(
        _MOE_LAYER_STATS.items(),
        key=lambda item: item[1]["wall_ms"],
        reverse=True,
    )[:3]
    for layer_prefix, lstats in top_layers:
        lcalls = max(1, lstats["calls"])
        hist = ",".join(
            f"{active}:{count}"
            for active, count in sorted(lstats["active_hist"].items())
        )
        print(
            f"[V100-FP8-MOE-PROFILE rank={rank} pid={pid}] "
            f"top_layer={layer_prefix} calls={lstats['calls']} "
            f"avg_wall={lstats['wall_ms'] / lcalls:.3f}ms "
            f"active_hist={hist}",
            flush=True,
        )


def _moe_profile_print_bucket(rank, pid, key, stats):
    """Pretty-print one shape-bucket aggregate.

    Important: distinguish cuda_event_sections (GPU-elapsed time, accurate)
    from wall_sections (perf_counter; launch + Python overhead, NOT GPU
    elapsed). The unattributed residual is what's left over and is the
    target signal for Step B optimization decisions.
    """
    phase, M, route_slots, grouped, fast = key
    calls = max(1, stats["calls"])
    sections = stats["sections"]
    cuda_ms = sum(sections[name] for name in _MOE_CUDA_EVENT_SECTIONS)
    wall_ms = sum(sections[name] for name in _MOE_WALL_SECTIONS)
    total_wall = stats["wall_ms"]
    unattrib_ms = total_wall - cuda_ms - wall_ms

    cuda_bits = " ".join(
        f"{name}={sections[name] / calls:.3f}"
        for name in _MOE_CUDA_EVENT_SECTIONS
        if sections[name] > 0.0
    )
    wall_bits = " ".join(
        f"{name}={sections[name] / calls:.3f}"
        for name in _MOE_WALL_SECTIONS
        if sections[name] > 0.0
    )
    print(
        f"[V100-FP8-MOE-PROFILE rank={rank} pid={pid}] "
        f"bucket=(phase={phase},M={M},route_slots={route_slots},grouped={grouped},fast={fast}) "
        f"calls={stats['calls']} "
        f"avg_wall={total_wall / calls:.3f}ms "
        f"cuda_sections={cuda_ms / calls:.3f}ms "
        f"wall_sections={wall_ms / calls:.3f}ms "
        f"unattributed={unattrib_ms / calls:.3f}ms "
        f"active_experts={stats['active_experts'] / calls:.2f}/call "
        f"routed_items={stats['routed_items'] / calls:.2f}/call",
        flush=True,
    )
    if cuda_bits:
        print(
            f"[V100-FP8-MOE-PROFILE rank={rank} pid={pid}] "
            f"  cuda_ms_per_call: {cuda_bits}",
            flush=True,
        )
    if wall_bits:
        print(
            f"[V100-FP8-MOE-PROFILE rank={rank} pid={pid}] "
            f"  wall_ms_per_call: {wall_bits}",
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
        call_stats = _new_moe_call_stats()
        cuda_events = []

        def timed_cuda(name, fn):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            cuda_events.append((name, start, end))
            return result

        def timed_wall(name, fn):
            # Wall (perf_counter) timer for sections dominated by Python/
            # launch overhead rather than GPU elapsed time. Does NOT sync,
            # so this measures the launch/allocator path, not GPU work.
            t0 = time.perf_counter()
            result = fn()
            call_stats["sections"][name] += (time.perf_counter() - t0) * 1000.0
            return result
    else:
        call_stats = None
        cuda_events = None
        timed_cuda = None
        timed_wall = None

    M, hidden_size = x_work.shape
    intermediate = int(layer.intermediate_size_per_partition)
    block_h, block_w = layer.weight_block_size

    # Convert global expert ids to local ids. Non-local experts become -1 and
    # contribute zero on this rank; vLLM's later TP/EP combine handles the sum.
    # When expert_map is None, the grouped fast path (Step C) skips this
    # entire block; we still run local_topk = topk_ids.to(long) for the
    # legacy/fallback paths.
    routing_t0 = time.perf_counter() if profile else None
    local_topk = topk_ids.to(torch.long)
    expert_map_dev = _get_layer_expert_map_dev(layer, topk_ids.device)
    if expert_map_dev is not None:
        safe_ids = torch.clamp(local_topk, min=0)
        local_topk = expert_map_dev[safe_ids]
        local_topk = torch.where(
            topk_ids < 0, torch.full_like(local_topk, -1), local_topk)
    if profile:
        call_stats["sections"]["routing"] += (
            time.perf_counter() - routing_t0) * 1000.0

    route_slots = int(local_topk.numel())

    grouped_k_splits = _moe_grouped_routed_k_splits(
        hidden_size,
        intermediate,
        block_h,
        block_w,
        route_slots,
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
            timed_cuda,
            timed_wall,
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
            fast_flag = 1 if call_stats.get("used_fast_path") else 0
            bucket_key = (
                _moe_phase_tag(M, route_slots), M, route_slots, 1, fast_flag)
            _moe_profile_update(
                getattr(layer, "prefix", "<unknown>"), bucket_key, call_stats)
        return out

    w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
    w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
    if profile:
        out = timed_wall(
            "out_zeros",
            lambda: torch.zeros(
                (M, hidden_size), dtype=torch.float16, device=x.device))
    else:
        out = torch.zeros((M, hidden_size), dtype=torch.float16, device=x.device)
    apply_weight_on_input = bool(getattr(layer, "apply_router_weight_on_input", False))

    local_num_experts = int(layer.local_num_experts)
    if _MOE_ACTIVE_LIST:
        if profile:
            mask_t0 = time.perf_counter()
        expert_iter = torch.unique(local_topk[local_topk >= 0]).tolist()
        if profile:
            # In active-list mode this is the same instrumentation-style
            # sync as grouped mode's torch.unique() stat; carry it under the
            # same bucket so the two paths are directly comparable.
            call_stats["sections"]["active_experts_stat"] += (
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
                call_stats["sections"]["active_experts_stat"] += (
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
        if profile:
            route_w = timed_cuda(
                "route_w_build",
                lambda: topk_weights[token_idx, route_idx].to(torch.float16))
        else:
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
            if profile:
                expert_out = timed_cuda(
                    "route_weight_apply",
                    lambda: expert_out * route_w.unsqueeze(-1))
            else:
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
        bucket_key = (_moe_phase_tag(M, route_slots), M, route_slots, 0, 0)
        _moe_profile_update(
            getattr(layer, "prefix", "<unknown>"), bucket_key, call_stats)
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


def _patch_volta_moe_default_config():
    """Volta-fit fused-MoE default config (fp16/bf16 unquantized MoE only).

    Replaces only the small-M branch of fused_moe.get_default_config on sm_70:
    stock picks BLOCK_SIZE_K=128 (and num_stages=4) for M<=64, which
    register-spills Triton's Volta codegen; spill traffic contends on HBM and
    the cost grows ~linearly with M. Measured (Qwen3.6-35B-A3B / gemma-4-26B
    shapes, results/moe_decode_msweep_*): BLOCK_K=64 tiles are 2.3x faster at
    M=1 and up to 9.6x at M=16; e2e 15.57 -> 65.87 tok/s (4.23x, bit-identical
    output) single-stream. M<=4 wants 16/32/64 w4; M=8..64 wants 16/128/64 w8.
    The config-file lookup (in-tree JSON or VLLM_TUNED_CONFIG_FOLDER) runs
    BEFORE this fallback and still wins if present.  Gate: set
    VLLM_V100_MOE_FP16_TUNED=0 to restore stock behavior (e.g. for A/B runs).
    """
    if os.environ.get("VLLM_V100_MOE_FP16_TUNED", "1") != "1" or not _is_volta():
        return
    from vllm.model_executor.layers.fused_moe import fused_moe as _fm

    _orig_get_default_config = _fm.get_default_config

    def volta_get_default_config(M, E, N, K, topk, dtype, block_shape=None):
        if dtype is None and M <= 64:
            if M <= 4:
                return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32,
                        "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
                        "num_warps": 4, "num_stages": 2}
            return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
                    "num_warps": 8, "num_stages": 2}
        return _orig_get_default_config(M, E, N, K, topk, dtype, block_shape)

    _fm.get_default_config = volta_get_default_config
    print(f"[serve_fp8_v100 pid={os.getpid()}] volta moe default-config patch "
          "ACTIVE (fp16/bf16 fused-MoE small-M -> BLOCK_K=64 tiles; "
          "VLLM_V100_MOE_FP16_TUNED=0 to disable)", flush=True)


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
                _attach_shared_experts_debug_hook(self)

            qwen3_next.Qwen3NextForCausalLM.__init__ = patched_qwen3next_init

            original_qwen35_base_init = qwen3_5.Qwen3_5ForCausalLMBase.__init__

            def patched_qwen35_base_init(self, *args, **kwargs):
                original_qwen35_base_init(self, *args, **kwargs)
                _attach_decode_breakdown_hooks(self)
                _attach_shared_experts_debug_hook(self)

            qwen3_5.Qwen3_5ForCausalLMBase.__init__ = patched_qwen35_base_init

            # Stage 2D, Step 2B.1: measurement-only wrap of
            # Qwen3NextSparseMoeBlock.forward so the moe_other residual
            # can be decomposed into combine + all-reduce + python.
            # Identical semantics to the upstream forward; the only
            # difference is CUDA events bracketing two narrow sections,
            # appended to the existing _BREAKDOWN_PENDING_EVENTS list.
            if _MOE_OTHER_PROFILE and torch.cuda.is_available():
                _orig_sparsemoe_forward = (
                    qwen3_next.Qwen3NextSparseMoeBlock.forward)
                # Imports the wrapper needs. Resolved once at install time
                # so the per-call hot path is just attribute lookups.
                from vllm.distributed import (
                    tensor_model_parallel_all_gather as _tp_all_gather)
                from vllm.model_executor.models.utils import (
                    sequence_parallel_chunk as _sp_chunk)

                def _timed_sparsemoe_forward(self, hidden_states):
                    orig_shape = hidden_states.shape
                    num_tokens, hidden_dim = hidden_states.shape
                    regime = "decode" if num_tokens == 1 else "prefill"
                    hidden_states = hidden_states.view(-1, hidden_dim)

                    if self.is_sequence_parallel:
                        hidden_states = _sp_chunk(hidden_states)

                    if self.experts.is_internal_router:
                        final_hidden_states = self.experts(
                            hidden_states=hidden_states,
                            router_logits=hidden_states)
                    else:
                        router_logits, _ = self.gate(hidden_states)
                        final_hidden_states = self.experts(
                            hidden_states=hidden_states,
                            router_logits=router_logits)

                    if self.shared_expert is not None:
                        do_timing = (
                            not _BREAKDOWN_RUNTIME_DISABLED
                            and not torch.cuda.is_current_stream_capturing())
                        if do_timing:
                            combine_start = torch.cuda.Event(enable_timing=True)
                            combine_end = torch.cuda.Event(enable_timing=True)
                            combine_start.record()
                        final_hidden_states = (
                            final_hidden_states[0] + final_hidden_states[1])
                        if do_timing:
                            combine_end.record()
                            _BREAKDOWN_PENDING_EVENTS.append(
                                (regime, "moe_other_combine",
                                 combine_start, combine_end))

                    if self.is_sequence_parallel:
                        do_timing = (
                            not _BREAKDOWN_RUNTIME_DISABLED
                            and not torch.cuda.is_current_stream_capturing())
                        if do_timing:
                            ag_start = torch.cuda.Event(enable_timing=True)
                            ag_end = torch.cuda.Event(enable_timing=True)
                            ag_start.record()
                        final_hidden_states = _tp_all_gather(
                            final_hidden_states, 0)
                        final_hidden_states = final_hidden_states[:num_tokens]
                        if do_timing:
                            ag_end.record()
                            _BREAKDOWN_PENDING_EVENTS.append(
                                (regime, "moe_other_allreduce",
                                 ag_start, ag_end))
                    elif self.tp_size > 1:
                        do_timing = (
                            not _BREAKDOWN_RUNTIME_DISABLED
                            and not torch.cuda.is_current_stream_capturing())
                        if do_timing:
                            ar_start = torch.cuda.Event(enable_timing=True)
                            ar_end = torch.cuda.Event(enable_timing=True)
                            ar_start.record()
                        final_hidden_states = (
                            self.experts.maybe_all_reduce_tensor_model_parallel(
                                final_hidden_states))
                        if do_timing:
                            ar_end.record()
                            _BREAKDOWN_PENDING_EVENTS.append(
                                (regime, "moe_other_allreduce",
                                 ar_start, ar_end))

                    return final_hidden_states.view(orig_shape)

                qwen3_next.Qwen3NextSparseMoeBlock.forward = (
                    _timed_sparsemoe_forward)
                if _breakdown_rank() in (0, -1):
                    print(
                        f"[V100-FP8 pid={os.getpid()}] Stage 2D Step 2B.1 "
                        f"moe_other sub-attribution patch installed "
                        f"(combine + allreduce timing).",
                        flush=True,
                    )

            # Stage 2D, Step 2D.2: measurement-only wrap of
            # Qwen3NextGatedDeltaNet._forward_core so the FLA Triton
            # core-attention path can be timed as a single `gdn_core`
            # bucket. Semantics identical to upstream; only adds CUDA
            # events bracketing the method body and a single
            # _BREAKDOWN_PENDING_EVENTS append per call. Sibling subs
            # (in_proj_qkvz, in_proj_ba, norm, out_proj) are hooked at
            # the Module level via _attach_decode_breakdown_hooks; no
            # nested accounting needed.
            if _BREAKDOWN_GDN_SUB_HOOK and torch.cuda.is_available():
                _orig_gdn_forward_core = (
                    qwen3_next.Qwen3NextGatedDeltaNet._forward_core)

                def _timed_gdn_forward_core(self, mixed_qkv, b, a, core_attn_out):
                    if (_BREAKDOWN_RUNTIME_DISABLED
                            or torch.cuda.is_current_stream_capturing()):
                        return _orig_gdn_forward_core(
                            self, mixed_qkv, b, a, core_attn_out)
                    regime = "decode" if mixed_qkv.size(0) == 1 else "prefill"
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    result = _orig_gdn_forward_core(
                        self, mixed_qkv, b, a, core_attn_out)
                    end.record()
                    _BREAKDOWN_PENDING_EVENTS.append(
                        (regime, "gdn_core", start, end))
                    return result

                qwen3_next.Qwen3NextGatedDeltaNet._forward_core = (
                    _timed_gdn_forward_core)
                if _breakdown_rank() in (0, -1):
                    print(
                        f"[V100-FP8 pid={os.getpid()}] Stage 2D Step 2D.2 "
                        f"gdn_core sub-attribution patch installed "
                        f"(_forward_core timing).",
                        flush=True,
                    )

            # Stage 2D, Step 2D.3: cross-cutting timer for the all-reduce
            # hidden inside RowParallelLinear.forward at TP>1. Replicates
            # upstream semantics exactly (input split, GEMM, AR, bias) --
            # only wraps the AR call with CUDA events. The patched forward
            # is a verbatim mirror of upstream linear.py to keep behavior
            # identical; only the two .record() calls and the
            # _BREAKDOWN_PENDING_EVENTS.append are additive.
            if _ROW_PARALLEL_AR_PROFILE and torch.cuda.is_available():
                from vllm.model_executor.layers.linear import (
                    RowParallelLinear as _RowParallelLinear,
                )
                from vllm.distributed import (
                    split_tensor_along_last_dim as _split_tensor_along_last_dim,
                    tensor_model_parallel_all_reduce as _tp_ar,
                )

                _orig_row_parallel_forward = _RowParallelLinear.forward

                def _timed_row_parallel_forward(self, input_):
                    if self.input_is_parallel:
                        input_parallel = input_
                    else:
                        split_input = _split_tensor_along_last_dim(
                            input_, num_partitions=self.tp_size)
                        input_parallel = split_input[self.tp_rank].contiguous()

                    assert self.quant_method is not None
                    bias_ = (None
                             if (self.tp_rank > 0 or self.skip_bias_add)
                             else self.bias)
                    output_parallel = self.quant_method.apply(
                        self, input_parallel, bias_)

                    if self.reduce_results and self.tp_size > 1:
                        num_tokens = output_parallel.size(0)
                        regime = "decode" if num_tokens == 1 else "prefill"
                        do_timing = (
                            not _BREAKDOWN_RUNTIME_DISABLED
                            and not torch.cuda.is_current_stream_capturing())
                        if do_timing:
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                        output = _tp_ar(output_parallel)
                        if do_timing:
                            end.record()
                            _BREAKDOWN_PENDING_EVENTS.append(
                                (regime, "row_parallel_ar", start, end))
                    else:
                        output = output_parallel

                    if not self.return_bias:
                        return output
                    output_bias = (self.bias
                                   if self.skip_bias_add else None)
                    return output, output_bias

                _RowParallelLinear.forward = _timed_row_parallel_forward
                if _breakdown_rank() in (0, -1):
                    print(
                        f"[V100-FP8 pid={os.getpid()}] Stage 2D Step 2D.3 "
                        f"row_parallel_ar cross-cutting timer installed "
                        f"(times only the AR call inside "
                        f"RowParallelLinear.forward).",
                        flush=True,
                    )
        except Exception as exc:
            print(
                f"[DECODE-BREAKDOWN rank={_breakdown_rank()} pid={os.getpid()}] "
                f"failed to install hooks: {type(exc).__name__}: {exc}",
                flush=True,
            )

    # Additive, feature-detected: enable compressed-tensors W8A16-FP8 (RedHatAI
    # checkpoints) on sm_70. No-ops off-Volta or if the class is absent. Does NOT
    # touch the `fp8` (Fp8LinearMethod) hooks above — separate entry point.
    try:
        from fp8_w8a16_sm70.compressed_tensors_v100 import (
            patch_compressed_tensors_for_v100,
            patch_compressed_tensors_moe_for_v100,
        )
        patch_compressed_tensors_for_v100()
        patch_compressed_tensors_moe_for_v100()
    except Exception as exc:
        print(f"[serve_fp8_v100 pid={os.getpid()}] compressed-tensors patch "
              f"skipped: {type(exc).__name__}: {exc}", flush=True)

    # Additive, env-gated (VLLM_V100_FLASH_ATTN=1): route TRITON_ATTN prefill
    # batches to the ai-bond flash-attention-v100 kernel (8.4x at 26k, audit
    # docs/FA_V100_AUDIT.md Turn 8). Decode keeps the validated Triton+cudagraph
    # path by construction. No-op if the extension or the flag is absent.
    # Requires --block-size 256 (ai-bond paged constraint).
    try:
        from fp8_w8a16_sm70.fa_v100_prefill import patch_triton_prefill_for_v100
        patch_triton_prefill_for_v100()
    except Exception as exc:
        print(f"[serve_fp8_v100 pid={os.getpid()}] fa_v100 prefill patch "
              f"skipped: {type(exc).__name__}: {exc}", flush=True)

    # Additive, env-gated (VLLM_V100_MOE_FP16_TUNED=1, DEFAULT ON): Volta-fit
    # fused-MoE default config for unquantized (fp16/bf16) MoE. Stock
    # get_default_config picks BLOCK_K=128/num_stages=4 for decode-sized M,
    # which register-spills Triton's sm_70 codegen — measured 2.3-9.6x slower
    # than BLOCK_K=64 tiles (results/moe_decode_msweep_*, e2e 4.2x single-stream
    # validated 2026-06-12). Only the config-MISS fallback is patched: a tuned
    # JSON (in-tree or VLLM_TUNED_CONFIG_FOLDER) still takes priority, and
    # quantized dtypes (our FP8 path included) are untouched.
    try:
        _patch_volta_moe_default_config()
    except Exception as exc:
        print(f"[serve_fp8_v100 pid={os.getpid()}] volta moe default-config "
              f"patch skipped: {type(exc).__name__}: {exc}", flush=True)

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
