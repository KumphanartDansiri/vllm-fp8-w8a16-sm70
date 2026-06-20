"""V100 (sm_70) enablement for compressed-tensors FP8 models (e.g. RedHatAI).

WHY a separate hook from the main fp8 monkey-patch:
  Our W8A16 kernel patch in `vllm_serve` hooks `Fp8LinearMethod` (the `fp8`
  quant_method, used by the Qwen block-FP8 checkpoints). RedHatAI checkpoints use
  the `compressed-tensors` quant_method instead, which routes through entirely
  separate scheme classes. So this is a DIFFERENT entry point — added additively,
  feature-detected, so it no-ops on builds/versions that lack the classes and
  never disturbs the existing Fp8 hooks.

WHAT vLLM already does for us (verified in compressed_tensors.py:663-680):
  A compressed-tensors W8A8-FP8 model (weight FP8 + quantized activation, e.g.
  RedHatAI "FP8-Dynamic": per-channel weight + per-token dynamic act) is selected
  as `CompressedTensorsW8A8Fp8` ONLY if the GPU supports it (min cap 89). On a
  GPU below 89 — V100 included — vLLM FALLS BACK to `CompressedTensorsW8A16Fp8`
  (weight-only FP8, FP16 activation; the activation quant is ignored). So we do
  NOT touch W8A8 (leaving its 89 gate lets the fallback fire); we only enable the
  W8A16 fallback target on sm_70.

WHAT THIS PATCH DOES (first cut — dequant-to-FP16):
  `CompressedTensorsW8A16Fp8` normally delegates to a Marlin W8A16 kernel
  (init_wfp8_a16_linear_kernel) gated at sm_75. On sm_70 we:
    1. lower get_min_capability -> 70 (so the final get_scheme() check at
       compressed_tensors.py:792 passes), and
    2. replace process_weights_after_loading + apply_weights to BYPASS Marlin:
       dequantize the FP8 weight to FP16 once at load (per-channel / per-tensor /
       per-block scale), then serve with a plain cuBLAS F.linear.
  This is correct and fast (native FP16 tensor-core GEMM), but spends FP16 weight
  memory — so it only helps models that FIT in FP16. The FP8-resident path (keep
  weights FP8 + route to our W8A16 CUDA kernel, for models too big for FP16) is a
  planned SECOND cut; for block-strategy weights that can reuse the existing
  kernel directly.

Only sm_70 is touched; sm_75+ keeps stock Marlin.
"""
import os
import time as _time

import torch
import torch.nn.functional as F


# ─── Phase 4 prefill profiler (VLLM_V100_CT_PROFILE=1) ───────────────────────
# Per-section GPU time (CUDA events) AND wall time (perf_counter, NO sync) for the
# PREFILL forward, so we can see the real 64s breakdown instead of estimating. The
# GPU-vs-wall gap is the launch/Python-overhead signature: "GPU math small, wall
# huge" => the per-expert loop's ~5760 sequential launches dominate (Codex). Run
# under MODE=eager (CUDA-event timing inside cudagraph replay is unreliable). The
# summary prints once, on the first DECODE-sized MoE apply after a prefill forward.
_PROF = os.environ.get("VLLM_V100_CT_PROFILE", "0").lower() not in ("0", "off", "false", "")
_PROF_ACC = {}        # name -> [gpu_ms_total, wall_ms_total, calls]
_PROF_EVT = []        # pending (name, start_event, end_event, wall_ms) until drained
_PROF_SAW_PREFILL = [False]
_PROF_DONE = [False]


class _ProfSec:
    """Context manager: records a CUDA-event pair (GPU time, drained later) and a
    no-sync wall time (CPU/launch time). No-op unless VLLM_V100_CT_PROFILE=1."""
    __slots__ = ("name", "t0", "s", "e")

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if _PROF:
            self.t0 = _time.perf_counter()
            self.s = torch.cuda.Event(enable_timing=True)
            self.e = torch.cuda.Event(enable_timing=True)
            self.s.record()
        return self

    def __exit__(self, *a):
        if _PROF:
            self.e.record()
            _PROF_EVT.append((self.name, self.s, self.e,
                              (_time.perf_counter() - self.t0) * 1000.0))


def _prof_drain():
    if _PROF_EVT:
        torch.cuda.synchronize()
        for nm, s, e, wall in _PROF_EVT:
            a = _PROF_ACC.setdefault(nm, [0.0, 0.0, 0])
            a[0] += s.elapsed_time(e); a[1] += wall; a[2] += 1
        _PROF_EVT.clear()


def _prof_report(M, min_r):
    """large M => prefill in progress; first small M after => print once."""
    if not _PROF or _PROF_DONE[0]:
        return
    if M >= min_r:
        _PROF_SAW_PREFILL[0] = True
        return
    if not _PROF_SAW_PREFILL[0]:
        return
    _prof_drain()
    if _ct_rank() in (0, -1):
        tot_g = sum(v[0] for v in _PROF_ACC.values())
        tot_w = sum(v[1] for v in _PROF_ACC.values())
        print("[serve_fp8_v100 ct-PROFILE] PREFILL section breakdown "
              "(our code only; attention/TP-comm/overhead = TTFT minus this):", flush=True)
        for nm in sorted(_PROF_ACC, key=lambda k: -_PROF_ACC[k][1]):
            g, w, c = _PROF_ACC[nm]
            print(f"    {nm:26s} wall={w:9.1f}ms  gpu={g:9.1f}ms  calls={c:5d}"
                  f"  (gap={w-g:8.1f}ms = launch/python)", flush=True)
        print(f"    {'OUR TOTAL':26s} wall={tot_w:9.1f}ms  gpu={tot_g:9.1f}ms", flush=True)
    _PROF_DONE[0] = True


# Phase 1 (FP8-resident channel Linear): keep the FP8 weight in VRAM and route
# apply through the V100 W8A16 kernel (channel scale == degenerate block with
# block_h=1, block_w=K), instead of dequantizing the weight to FP16 at load.
# This halves the weight VRAM vs the first-cut dequant path. Opt-in while
# validating; per-layer it falls back to dequant-FP16 for non-CHANNEL strategy
# or when K is not 128-aligned (the A.1/A.2 kernels require it).
_CT_FP8_RESIDENT = os.environ.get(
    "VLLM_V100_CT_FP8_RESIDENT", "0").lower() not in ("0", "off", "false", "")
# Per-layer load-time self-check: run the resident kernel vs the dequant-FP16
# reference on a random probe for each resident layer and log any that diverge.
# Pinpoints an unsafe Linear family in ONE load (vs a multi-run bisect). On by
# default while resident is being validated; set =0 once Phase 1 is trusted.
_CT_SELFCHECK = os.environ.get(
    "VLLM_V100_CT_FP8_RESIDENT_SELFCHECK", "1").lower() not in ("0", "off", "false", "")
# Comma-separated prefix substrings to EXCLUDE from the resident path (force
# FP16 fallback), e.g. "qkv_proj,o_proj". Lets us ship Phase 1 minus a family
# that the self-check flags as unsafe, while keeping the rest of the VRAM win.
_CT_EXCLUDE = [s.strip() for s in os.environ.get(
    "VLLM_V100_CT_FP8_RESIDENT_EXCLUDE", "").split(",") if s.strip()]
# Phase 2a: load-time real-weight self-check of the grouped MoE kernel on the
# experts' w13 (gate_up) FP8 weights vs the FP16 dequant reference. Diagnostic
# only — execution stays FP16 until Phase 2b. Proves the grouped channel kernel
# is correct on the ACTUAL GLM expert weights (not just the random numtest), the
# same way the dense self-check caught gate_up. Default on while validating.
_CT_MOE_SELFCHECK = os.environ.get(
    "VLLM_V100_CT_FP8_MOE_SELFCHECK", "1").lower() not in ("0", "off", "false", "")
_CT_MOE_COUNTS = {"w13_ok": 0, "w13_bad": 0, "w2_ok": 0, "w2_bad": 0}
# Phase 2b (mixed CT-MoE): execute w13 (gate_up) FP8-resident via the grouped
# kernel, keep w2 (down) FP16 (its K=I/TP=176 isn't 128-aligned). GUARDED first
# cut: keeps the FP16 w13 too (no memory win yet) and validates the full mixed
# routed output against `_v100_unquant.apply` per layer; on mismatch/exception it
# permanently falls back to the proven FP16 path. Only engages when
# shared_experts_input is None (GLM's shared expert is a separate CT Linear).
_CT_MOE_W13_RESIDENT = os.environ.get(
    "VLLM_V100_CT_MOE_W13_RESIDENT", "0").lower() not in ("0", "off", "false", "")
# L2-rel tolerance for the mixed-vs-unquant full-path self-check.
_CT_MOE_MIX_TOL = float(os.environ.get("VLLM_V100_CT_MOE_MIX_TOL", "0.02"))
# Cap how many MoE layers keep the guarded w13 FP8 stash (which is +~184 MB/GPU
# PER LAYER on top of FP16 w13 — ~8.3 GB across all 45 layers => OOM). For the
# GUARDED validation run, cap to a few layers so it fits AND still runs the full
# mixed-vs-unquant self-check on real routing. 0 = no cap (the true Phase 2c path,
# which must also stop keeping FP16 w13 — see refactor).
_CT_MOE_W13_MAXLAYERS = int(os.environ.get("VLLM_V100_CT_MOE_W13_RESIDENT_MAXLAYERS", "0"))
_CT_MOE_W13_STASH_N = [0]
_CT_MOE_MIX_ENGAGED = [False]   # one-shot "mixed path ran" banner guard
# Phase 2c (the memory win): after building the unquant method (which needs FP16
# w13 transiently), FREE the FP16 w13 weight — the mixed apply uses the stashed
# FP8 w13, not the unquant kernel, so the FP16 copy (~16.6 GB across 45 layers) is
# dead weight. Reclaims ~8.3 GB/GPU net (FP8 stash kept). Off by default; this is
# the "trusted" path (no working unquant fallback once FP16 w13 is gone).
_CT_MOE_W13_FREE_FP16 = os.environ.get(
    "VLLM_V100_CT_MOE_W13_FREE_FP16", "0").lower() not in ("0", "off", "false", "")
_CT_MOE_W13_FREED_N = [0]
# Phase 2d: replace the w2 per-expert Python loop (`unique().tolist()` + one
# torch matmul per active expert) with one grouped FP16 routed GEMM launch. This
# is the decode hot path after w13 became FP8-resident.
_CT_MOE_W2_GROUPED = os.environ.get(
    "VLLM_V100_CT_MOE_W2_GROUPED", "1").lower() not in ("0", "off", "false", "")
_CT_MOE_W2_K_SPLIT = max(1, int(os.environ.get("VLLM_V100_CT_MOE_W2_K_SPLIT", "1")))
_CT_MOE_W2_CHUNK = max(1, int(os.environ.get("VLLM_V100_CT_MOE_W2_CHUNK", "60000")))
# Phase 4 Stage 1 (PREFILL perf): the grouped routed kernels are ONE-CTA-PER-ROW
# (gridDim.y=R, a per-row GEMV with zero cross-row weight reuse → 0.24 TFLOP/s),
# which is fine for DECODE (R=topk tiny, cudagraphed) but makes PREFILL ~94% of the
# 169s-TTFT@26k. At large R, route through per-expert TILED GEMMs that REUSE each
# expert weight across its rows: w13 via the a2 kernel (FP8, tiles BLOCK_M rows →
# ~7×), w2 via cuBLAS FP16 (already-resident weight, tensor cores). Dispatched ONLY
# when R >= MIN_R, so the decode/cudagraph per-row path is untouched (decode R is
# always < MIN_R). Off → keep the per-row grouped kernels everywhere.
_CT_MOE_PREFILL_TILED = os.environ.get(
    "VLLM_V100_CT_MOE_PREFILL_TILED", "1").lower() not in ("0", "off", "false", "")
# R threshold separating decode (small, cudagraphed) from prefill (large). Decode
# R = batch*topk; cudagraph capture sizes top out at 16 → R<=128. Prefill R = tokens
# *topk (hundreds+). 256 cleanly separates them.
_CT_MOE_PREFILL_TILED_MIN_R = max(
    1, int(os.environ.get("VLLM_V100_CT_MOE_PREFILL_TILED_MIN_R", "256")))
_CT_MOE_PREFILL_ENGAGED = [False]
# Phase 4 Stage 1.5: the per-expert tiled LOOP's real cost (profiler) is the
# `.tolist()` syncs + 128 sequential launches/layer (~17.5s prefill WALL @26k), NOT
# GPU math. The FUSED path does w13 in ONE kernel launch with GPU-side expert
# offsets (sync-free argsort+scatter_add+cumsum route-prep) and routes w2 through
# the per-row grouped kernel (also no .tolist()), so NO host sync on the hot path.
# Default on (within the tiled prefill regime); =0 reverts to the per-expert loop.
_CT_MOE_PREFILL_FUSED = os.environ.get(
    "VLLM_V100_CT_MOE_PREFILL_FUSED", "1").lower() not in ("0", "off", "false", "")
_CT_MOE_FUSED_ENGAGED = [False]
# Stage G1: route the DECODE w13 (small R) through the GROUPED COALESCED GEMV
# (warp->output-column, lanes->consecutive K) instead of the grouped a3 kernel
# (thread->column, N-strided, 28 sectors/req). NCU: 28->2.38 sectors/req, kernel
# 2.25x faster (187->83us). Only the small-R DECODE path — prefill (large R) uses
# the fused/tiled kernels (per-row GEMV is wrong for prefill). Default on
# (correctness-proven cos=1.0 vs a3+FP32); kill switch =0.
_CT_MOE_W13_COALESCED = os.environ.get(
    "VLLM_V100_CT_MOE_W13_COALESCED", "1").lower() not in ("0", "off", "false", "")
_CT_MOE_W13_COAL_ENGAGED = [False]
# Max routed rows (R) per grouped-w13 kernel launch. CUDA caps gridDim.y at
# 65535 and the kernel uses gridDim.y = R, so R must stay under it at long prefill
# (R = M*topk). Env-tunable mainly so the offline numtest can force chunking with
# small tensors.
_CT_MOE_W13_CHUNK = max(1, int(os.environ.get("VLLM_V100_CT_MOE_W13_CHUNK", "60000")))


def _is_volta() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) == (7, 0)


def _dequant_ct_weight_to_fp16(layer, strategy_name: str, weight_block_size):
    """FP8 weight + scale -> a single FP16 weight tensor [N, K].

    Layouts (compressed-tensors W8A16, post-load, pre-our-patch):
      weight       : float8_e4m3fn, [N, K]  (output, input — Linear layout)
      weight_scale : channel -> [N,1] ; tensor -> [num_partitions] ; block -> [N/bh, K/bw]
    """
    w = layer.weight.data
    N, K = w.shape
    wf = w.to(torch.float16)
    strat = strategy_name.upper()

    if "BLOCK" in strat and weight_block_size is not None:
        bh, bw = int(weight_block_size[0]), int(weight_block_size[1])
        sf = layer.weight_scale.data.to(torch.float16)
        scale_exp = sf.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)[:N, :K]
    elif "TENSOR" in strat:
        # Per-tensor — including FUSED modules (QKV, gate_up) that carry ONE
        # scale per logical shard (numel = #partitions, not 1). Expand to
        # per-output-channel exactly like stock convert_to_channelwise; a naive
        # reshape would crash on these (GPT review, 2026-06-08).
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
            convert_to_channelwise,
        )
        ws = convert_to_channelwise(layer.weight_scale, layer.logical_widths)
        scale_exp = ws.to(torch.float16).reshape(N, 1)
    else:
        # CHANNEL: one scale per output row.
        sf = layer.weight_scale.data.to(torch.float16)
        if sf.numel() == N:
            scale_exp = sf.reshape(N, 1)
        elif sf.numel() == 1:
            scale_exp = sf.reshape(1, 1)
        else:
            raise ValueError(
                f"serve_fp8_v100 ct: unexpected weight_scale numel={sf.numel()} "
                f"for strategy={strat} (N={N}); refusing to dequant unsafely."
            )

    return (wf * scale_exp).to(torch.float16).contiguous()


_BANNERED = False
# Per-process tally of the CT Linear path decision (resident kernel vs FP16
# fallback), so the e2e run reports real coverage instead of a single banner.
_CT_COUNTS = {"resident": 0, "fallback": 0}
_CT_FIRST_KIND = set()


def _ct_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return -1


def _ct_log_decision(layer, kind: str, detail: str) -> None:
    """One terse rank-0 line per CT Linear, carrying the running tally.

    rank 0 processes its shard of every Linear, and TP shards evenly so the
    resident/fallback decision is the same on all ranks — so rank 0's tally is
    the whole-model tally. grep `ct-layer` and count, or read the last line.
    """
    _CT_COUNTS[kind] = _CT_COUNTS.get(kind, 0) + 1
    if _ct_rank() not in (0, -1):
        return
    first = kind not in _CT_FIRST_KIND
    if first:
        _CT_FIRST_KIND.add(kind)
    print(f"[serve_fp8_v100 ct-layer] {kind:<8} {detail} "
          f"prefix={getattr(layer, 'prefix', '?')} "
          f"(resident={_CT_COUNTS['resident']} fallback={_CT_COUNTS['fallback']})"
          f"{' [first-of-kind]' if first else ''}", flush=True)


def patch_compressed_tensors_for_v100() -> bool:
    """Enable compressed-tensors W8A16-FP8 on sm_70. Returns True if applied.

    Feature-detected + additive: silently no-ops if the class is absent or the
    device is not Volta. Safe to call once at serve startup after the other
    V100 FP8 patches.
    """
    if not _is_volta():
        return False
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (  # noqa: E501
            CompressedTensorsW8A16Fp8,
        )
    except Exception as exc:  # class not present in this vLLM build
        print(f"[serve_fp8_v100 ct] CompressedTensorsW8A16Fp8 not importable, "
              f"skipping CT patch ({type(exc).__name__})", flush=True)
        return False

    # 0) Neutralize the Marlin/ScaledMM kernel init that create_weights() calls.
    #    On sm_70 init_wfp8_a16_linear_kernel raises "Failed to find a kernel that
    #    can implement the ScaledMM linear layer" — and create_weights runs BEFORE
    #    our patched process/apply, so the model dies at build. Our path never uses
    #    self.linear_kernel (we dequant + F.linear), so a no-op stub lets
    #    create_weights complete. Patched in the scheme's module namespace so it's
    #    localized to W8A16-FP8.
    try:
        import vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 as _w8a16mod  # noqa: E501

        class _NoOpW8A16Kernel:
            def process_weights_after_loading(self, layer):  # never called (we override)
                return None

            def apply_weights(self, layer, x, bias=None):     # tripwire if ever reached
                raise RuntimeError(
                    "serve_fp8_v100 ct: stub W8A16 kernel.apply_weights called — "
                    "the scheme's apply_weights override did not take effect."
                )

        _w8a16mod.init_wfp8_a16_linear_kernel = lambda *a, **k: _NoOpW8A16Kernel()
    except Exception as exc:
        print(f"[serve_fp8_v100 ct] could not neutralize init_wfp8_a16_linear_kernel "
              f"({type(exc).__name__}: {exc}) — create_weights may fail on sm_70", flush=True)

    # 1) lower the capability gate so get_scheme()'s final check passes on sm_70.
    @classmethod
    def patched_min_cap(cls):
        return 70
    CompressedTensorsW8A16Fp8.get_min_capability = patched_min_cap

    # 2) replace weight processing. Two paths:
    #    - FP8-RESIDENT (Phase 1, CHANNEL + K%128==0 + flag): keep the FP8 weight,
    #      expand the per-row channel scale into fake 128-wide block scales
    #      [N, K/128] (every K-block in a row carries the same value) so the V100
    #      W8A16 kernel runs WITH its A.3 split-K decode path. block_w=128 (NOT K)
    #      is deliberate: block_w=K would disable A.3 (K % (k_split*K) != 0) and
    #      force the slow A.1 fallback on the dominant decode case.
    #    - FP16 fallback (everything else): dequant FP8->FP16 once at load.
    def patched_process_weights_after_loading(self, layer):
        from vllm.model_executor.utils import replace_parameter
        strat = str(getattr(self, "strategy", "")).upper()
        wbs = getattr(self, "weight_block_size", None)
        w = layer.weight.data                       # FP8 e4m3, [N, K]
        N, K = w.shape

        prefix = str(getattr(layer, "prefix", ""))
        excluded = any(pat in prefix for pat in _CT_EXCLUDE)
        sf = getattr(layer, "weight_scale", None)
        # K must be 128-aligned (the vectorized uint4 K-loop reads 128-wide chunks;
        # a non-128 K would read out of bounds). N alignment is NO LONGER required
        # as of Phase 1.5: the A.1/A.2/A.3 kernels were hardened for partial-N
        # tiles (all threads reach the cooperative load + __syncthreads; only the
        # n-dependent work is masked). The per-layer self-check below is the hard
        # safety net — any layer whose kernel output still diverges from the
        # dequant-FP16 reference on its REAL weights auto-falls-back to FP16.
        resident_ok = (
            _CT_FP8_RESIDENT and not excluded and ("CHANNEL" in strat)
            and (K % 128 == 0) and sf is not None and sf.numel() == N
        )
        sc_why = None
        if resident_ok:
            Kb = K // 128
            # [N,1] -> [N, Kb] with each block sharing the row's channel scale.
            # Materialize once (contiguous); ~1/64 of the weight bytes.
            scale_blk = (sf.data.to(torch.float16).reshape(N, 1)
                         .expand(N, Kb).contiguous())
            # Per-layer self-check on the REAL weights: kernel vs dequant-FP16 on a
            # random probe. On divergence, AUTO-FALL-BACK this layer to FP16 (do
            # not commit resident) — robust against any shape the kernel mishandles.
            if _CT_SELFCHECK:
                try:
                    from fp8_w8a16_sm70.vllm_serve import _v100_fp8_gemm
                    wdq = w.to(torch.float16) * sf.data.to(torch.float16).reshape(N, 1)
                    probe = torch.randn(4, K, device=w.device, dtype=torch.float16) * 0.1
                    ref = F.linear(probe, wdq)
                    out, _v = _v100_fp8_gemm(probe, w.contiguous(), scale_blk, N, K, 1, 128)
                    l2 = ((out.float() - ref.float()).norm()
                          / ref.float().norm().clamp_min(1e-12)).item()
                    del wdq, ref, out, probe
                    if l2 >= 1e-2:
                        resident_ok = False
                        sc_why = f"selfcheck-FAIL(L2={l2:.3f})"
                        if _ct_rank() in (0, -1):
                            print(f"[serve_fp8_v100 ct-selfcheck] BAD->FP16 prefix={prefix} "
                                  f"N={N} K={K} L2rel={l2:.4f}", flush=True)
                except Exception as exc:  # on self-check error, be safe: fall back
                    resident_ok = False
                    sc_why = f"selfcheck-ERR({type(exc).__name__})"
                    if _ct_rank() in (0, -1):
                        print(f"[serve_fp8_v100 ct-selfcheck] ERR->FP16 prefix={prefix} "
                              f"{type(exc).__name__}: {exc}", flush=True)
        if resident_ok:
            replace_parameter(layer, "weight", w.contiguous())   # stays FP8
            replace_parameter(layer, "weight_scale", scale_blk)  # [N, K/128] FP16
            layer.input_scale = None
            layer._v100_ct_resident = True
            layer._v100_ct_NK = (N, K)
            _ct_log_decision(layer, "resident", f"N={N},K={K},Kb={Kb},bw=128")
            return

        # FP16 fallback: dequant FP8 -> FP16 once at load, bypass Marlin prep.
        why = (sc_why if sc_why is not None
               else "flag-off" if not _CT_FP8_RESIDENT
               else "excluded" if excluded
               else f"non-channel({strat})" if "CHANNEL" not in strat
               else f"K%128!=0(K={K})" if (K % 128)
               else f"N%128!=0(N={N})" if (N % 128)
               else "scale-shape")
        wdq = _dequant_ct_weight_to_fp16(layer, strat, wbs)
        replace_parameter(layer, "weight", wdq)        # weight is now FP16 [N, K]
        layer._v100_ct_resident = False
        layer._v100_ct_fp16 = True
        # scales/input-scale no longer needed for the FP16 path
        if hasattr(layer, "weight_scale"):
            try:
                del layer._parameters["weight_scale"]
            except Exception:
                layer.weight_scale = None
        layer.input_scale = None
        _ct_log_decision(layer, "fallback", f"strat={strat},why={why}")

    CompressedTensorsW8A16Fp8.process_weights_after_loading = (
        patched_process_weights_after_loading
    )

    # 3) replace apply. Resident layers -> V100 W8A16 kernel (block_h=1,block_w=128);
    #    FP16 fallback layers -> plain cuBLAS F.linear on the dequantized weight.
    def patched_apply_weights(self, layer, x, bias=None):
        if getattr(layer, "_v100_ct_resident", False):
            # lazy import avoids a module-load cycle (vllm_serve imports this file)
            from fp8_w8a16_sm70.vllm_serve import _v100_fp8_gemm
            N, K = layer._v100_ct_NK
            with _ProfSec("dense_linear_resident"):   # FP8-resident: WMMA/A.2/A.3
                out, _variant = _v100_fp8_gemm(
                    x.reshape(-1, K), layer.weight, layer.weight_scale, N, K, 1, 128)
            out = out.reshape(*x.shape[:-1], N)
            return out if bias is None else out + bias.to(out.dtype)
        w = layer.weight
        xin = x if x.dtype == w.dtype else x.to(w.dtype)
        b = bias if (bias is None or bias.dtype == w.dtype) else bias.to(w.dtype)
        with _ProfSec("dense_linear_fallback"):       # dequant-FP16: cuBLAS
            return F.linear(xin, w, b)

    CompressedTensorsW8A16Fp8.apply_weights = patched_apply_weights

    print(f"[serve_fp8_v100 ct pid={os.getpid()}] compressed-tensors W8A16-FP8 "
          f"patched for sm_70 (min_cap=70; FP8-resident channel="
          f"{'on' if _CT_FP8_RESIDENT else 'off'}, else dequant-to-FP16). "
          f"W8A8 left at cap=89 so its fallback to W8A16 fires.", flush=True)
    return True


_MOE_BANNERED = False


def _ct_moe_selfcheck_w13(layer) -> None:
    """Phase 2a: validate the grouped channel kernel on the REAL w13 experts.

    w13 is FP8 [E, 2I, H]: K=H (128-aligned), N=2I/TP (partial-N, e.g. 352). Run
    the grouped kernel (FP8 + expanded channel scale, block_h=1,block_w=128) on a
    random routed probe and compare to the per-expert dequant-FP16 reference. Logs
    a tally; does NOT change execution (PWAL still dequants to FP16). On error,
    just logs — never blocks load.
    """
    try:
        import torch as _t
        from fp8_w8a16_sm70.ext_loader import load_kernel
        ext = load_kernel(name="fp8_dequant_ext_vllm")
        w13 = layer.w13_weight.data                      # FP8 [E, N, H]
        s13 = getattr(layer, "w13_weight_scale", None)
        if s13 is None or w13.dim() != 3:
            return
        E, N, H = int(w13.size(0)), int(w13.size(1)), int(w13.size(2))
        if H % 128 != 0 or E == 0 or N == 0:
            return
        s = s13.data.to(_t.float16)
        if s.dim() == 2:
            s = s.unsqueeze(-1)                          # [E, N] -> [E, N, 1]
        elif s.dim() == 1:
            s = s.reshape(E, 1, 1)
        s = s.reshape(E, N, 1)
        Kb = H // 128
        scale_blk = s.expand(E, N, Kb).contiguous()      # [E, N, H/128] FP16
        w_u8 = w13.view(_t.uint8).contiguous()           # [E, N, H] bytes
        R = 16
        g = _t.Generator(device=w13.device).manual_seed(0)
        A = _t.randn(R, H, generator=g, device=w13.device, dtype=_t.float16) * 0.1
        eids = _t.randint(0, E, (R,), generator=g, device=w13.device, dtype=_t.int64)
        k_split = 8 if (H % 1024 == 0) else 4 if (H % 512 == 0) else 1
        out = ext.fp8_w8a16_grouped_routed_gemm_a3(
            A.contiguous(), eids.contiguous(), w_u8, scale_blk, N, H, 1, 128, k_split)
        ref = _t.empty(R, N, device=w13.device, dtype=_t.float32)
        for r in range(R):
            e = int(eids[r].item())
            wdq = w13[e].to(_t.float16) * s[e]            # [N, H]
            ref[r] = A[r].float() @ wdq.float().T
        l2 = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
        ok = l2 < 2e-2
        _CT_MOE_COUNTS["w13_ok" if ok else "w13_bad"] += 1
        tot = _CT_MOE_COUNTS["w13_ok"] + _CT_MOE_COUNTS["w13_bad"]
        if _ct_rank() in (0, -1):
            if not ok or tot <= 2:           # first couple + every failure, detailed
                print(f"[serve_fp8_v100 ct-moe-selfcheck] w13 {'OK' if ok else 'BAD'} "
                      f"E={E} N={N} H={H} L2rel={l2:.4f} prefix={getattr(layer,'prefix','?')} "
                      f"(ok={_CT_MOE_COUNTS['w13_ok']} bad={_CT_MOE_COUNTS['w13_bad']})", flush=True)
            else:                             # terse running tally so the LAST line is the true total
                print(f"[serve_fp8_v100 ct-moe-selfcheck] w13 tally "
                      f"ok={_CT_MOE_COUNTS['w13_ok']} bad={_CT_MOE_COUNTS['w13_bad']} "
                      f"(last L2rel={l2:.4f})", flush=True)
        del A, eids, out, ref, scale_blk
    except Exception as exc:  # never block load on a self-check error
        if _ct_rank() in (0, -1):
            print(f"[serve_fp8_v100 ct-moe-selfcheck] ERR "
                  f"{type(exc).__name__}: {exc}", flush=True)


def _ct_moe_selfcheck_w2(layer) -> None:
    """Phase 2d: validate the FP16 grouped-routed w2 kernel on the REAL w2 experts.

    The mixed apply keeps w2 in FP16 ([E, H, I], K=I/TP e.g. 176 — NOT 128-aligned)
    and routes it through `fp16_grouped_routed_gemm` (one grouped launch, replacing
    the per-expert loop). The offline numtest covered random weights; this closes
    the loop on the ACTUAL GLM w2 values/shapes at load time, exactly like the w13
    self-check. Run on a random routed probe vs the per-expert reference; log a
    tally. Does NOT change execution; on error, just logs (never blocks load).

    Call AFTER w2 is dequanted to FP16 (that's the tensor the mixed apply uses).
    """
    try:
        import torch as _t
        from fp8_w8a16_sm70.ext_loader import load_kernel
        ext = load_kernel(name="fp8_dequant_ext_vllm")
        if not hasattr(ext, "fp16_grouped_routed_gemm"):
            return
        w2 = layer.w2_weight.data                        # FP16 [E, H, I]
        if w2.dim() != 3 or w2.dtype != _t.float16:
            return
        E, N, K = int(w2.size(0)), int(w2.size(1)), int(w2.size(2))
        if E == 0 or N == 0 or K == 0 or (K % _CT_MOE_W2_K_SPLIT != 0):
            return
        R = 16
        g = _t.Generator(device=w2.device).manual_seed(0)
        A = _t.randn(R, K, generator=g, device=w2.device, dtype=_t.float16) * 0.1
        eids = _t.randint(0, E, (R,), generator=g, device=w2.device, dtype=_t.int64)
        out = ext.fp16_grouped_routed_gemm(
            A.contiguous(), eids.contiguous(), w2.contiguous(), _CT_MOE_W2_K_SPLIT)
        ref = _t.empty(R, N, device=w2.device, dtype=_t.float32)
        for r in range(R):
            e = int(eids[r].item())
            ref[r] = A[r].float() @ w2[e].float().T       # [N]
        l2 = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
        ok = l2 < 2e-2
        _CT_MOE_COUNTS["w2_ok" if ok else "w2_bad"] += 1
        tot = _CT_MOE_COUNTS["w2_ok"] + _CT_MOE_COUNTS["w2_bad"]
        if _ct_rank() in (0, -1):
            if not ok or tot <= 2:
                print(f"[serve_fp8_v100 ct-moe-selfcheck] w2 {'OK' if ok else 'BAD'} "
                      f"E={E} N={N} K={K} L2rel={l2:.4f} prefix={getattr(layer,'prefix','?')} "
                      f"(ok={_CT_MOE_COUNTS['w2_ok']} bad={_CT_MOE_COUNTS['w2_bad']})", flush=True)
            else:
                print(f"[serve_fp8_v100 ct-moe-selfcheck] w2 tally "
                      f"ok={_CT_MOE_COUNTS['w2_ok']} bad={_CT_MOE_COUNTS['w2_bad']} "
                      f"(last L2rel={l2:.4f})", flush=True)
        del A, eids, out, ref
    except Exception as exc:  # never block load on a self-check error
        if _ct_rank() in (0, -1):
            print(f"[serve_fp8_v100 ct-moe-selfcheck] w2 ERR "
                  f"{type(exc).__name__}: {exc}", flush=True)


def _v100_ct_tiled_prefill_moe(layer, route_x, expert_ids, mix, N13, H, I, w2):
    """Phase 4 Stage 1 — tiled per-expert prefill MoE (route_x already routed).

    Groups the routed rows by expert (one sort), then runs per-expert TILED GEMMs
    that REUSE each expert weight across its rows — the fix for the per-row grouped
    kernel's 0.24 TFLOP/s (no cross-row reuse). w13: the `a2` FP8 kernel (tiles
    BLOCK_M rows, ~7× the grouped kernel). w2: cuBLAS FP16 (weight already resident,
    tensor cores). Everything stays in sorted order until one final unsort. Returns
    `expert_out` [R, H] in ORIGINAL row order (pre route_w scaling). Only called at
    large R (prefill); the sort + .tolist() host-syncs are fine there (NOT on the
    cudagraphed decode path). Numerically equals the per-row grouped path (same FP8
    weights/scales; a2 and the grouped kernel both FP32-accumulate).
    """
    import torch as _t
    from fp8_w8a16_sm70.ext_loader import load_kernel
    from fp8_w8a16_sm70.vllm_serve import _moe_activation
    ext = load_kernel(name="fp8_dequant_ext_vllm")
    R = int(route_x.size(0))
    mu8, msc = mix["u8"], mix["scale"]
    E = int(mu8.size(0))

    with _ProfSec("moe.grouping"):
        order = _t.argsort(expert_ids)                   # rows grouped by expert
        eids_sorted = expert_ids.index_select(0, order)
        rx_sorted = route_x.index_select(0, order).contiguous()
        uniq, counts = _t.unique_consecutive(eids_sorted, return_counts=True)
        starts = _t.cumsum(counts, 0) - counts
        uniq_l, starts_l, counts_l = uniq.tolist(), starts.tolist(), counts.tolist()

    # w13: per-expert a2 FP8 GEMM (channel scale block_h=1, block_w=128) -> [R, 2I]
    with _ProfSec("moe.w13_loop"):
        w13_sorted = _t.empty(R, N13, device=route_x.device, dtype=_t.float16)
        for e, st, cnt in zip(uniq_l, starts_l, counts_l):
            if e < 0 or e >= E:
                w13_sorted[st:st + cnt].zero_(); continue   # invalid route -> 0
            w13_sorted[st:st + cnt] = ext.fp8_w8a16_gemm_a2(
                rx_sorted[st:st + cnt].contiguous(),
                mu8[e].reshape(-1), msc[e].reshape(-1), N13, H, 1, 128)

    with _ProfSec("moe.activation"):
        hidden_sorted = (_moe_activation(layer, w13_sorted[:, :I])
                         * w13_sorted[:, I:]).contiguous()   # [R, I]

    # w2: per-expert cuBLAS FP16 GEMM (weight already resident) -> [R, H]
    with _ProfSec("moe.w2_loop"):
        eo_sorted = _t.empty(R, H, device=route_x.device, dtype=_t.float16)
        for e, st, cnt in zip(uniq_l, starts_l, counts_l):
            if e < 0 or e >= E:
                eo_sorted[st:st + cnt].zero_(); continue
            eo_sorted[st:st + cnt] = hidden_sorted[st:st + cnt] @ w2[e].T

    with _ProfSec("moe.scatter"):
        expert_out = _t.empty(R, H, device=route_x.device, dtype=_t.float16)
        expert_out.index_copy_(0, order, eo_sorted)         # unsort to original order
    return expert_out


def _v100_ct_fused_prefill_moe(layer, route_x, expert_ids, mix, N13, H, I, w2):
    """Phase 4 Stage 1.5 — FUSED prefill MoE: w13 in ONE grouped-tiled kernel
    launch (GPU-side expert offsets), w2 via the per-row grouped kernel. NO Python
    per-expert loop and NO `.tolist()` on the hot path, which the profiler showed
    was ~17.5s of prefill wall (CPU/sync serialization), not GPU math. Stays in
    sorted order until one final unsort. Assumes valid expert ids in [0, E) (GLM
    top-k routing always is); the env kill switch reverts to the per-expert loop.
    """
    import torch as _t
    from fp8_w8a16_sm70.ext_loader import load_kernel
    from fp8_w8a16_sm70.vllm_serve import _moe_activation
    ext = load_kernel(name="fp8_dequant_ext_vllm")
    R = int(route_x.size(0))
    E = int(mix["u8"].size(0))
    BM = 8                                                  # == BLOCK_M_A2 in the kernel

    # Route-prep: ALL sync-free GPU ops (no .item / .tolist).
    with _ProfSec("moe.fused_prep"):
        order = _t.argsort(expert_ids)
        A_sorted = route_x.index_select(0, order).contiguous()
        counts = _t.zeros(E, dtype=_t.int32, device=route_x.device)
        counts.scatter_add_(0, expert_ids,
                            _t.ones_like(expert_ids, dtype=_t.int32))
        tiles = ((counts + BM - 1) // BM).to(_t.int32)
        e_tile_off = (_t.cumsum(tiles, 0) - tiles).to(_t.int32)
        e_row_off = (_t.cumsum(counts, 0) - counts).to(_t.int32)

    # w13: ONE fused launch -> [R, 2I] (sorted order)
    with _ProfSec("moe.fused_w13"):
        w13_sorted = ext.fp8_w8a16_grouped_tiled_gemm(
            A_sorted, e_tile_off, tiles, e_row_off, counts,
            mix["u8"], mix["scale"], N13, H, 1, 128)

    with _ProfSec("moe.activation"):
        hidden_sorted = (_moe_activation(layer, w13_sorted[:, :I])
                         * w13_sorted[:, I:]).contiguous()

    # w2: PER-EXPERT cuBLAS on the SORTED contiguous spans (tensor-core, ~0.5s GPU
    # at R=209k). NOT the per-row grouped kernel — that has no cross-row weight
    # reuse and measured 43s GPU at prefill R (the same pathology we fixed for w13;
    # Codex's catch). The sorted layout makes each expert's rows a contiguous slice
    # [e_row_off[e] : +counts[e]], so the loop is plain slicing (no boolean mask).
    # This re-introduces ONE .tolist() pair for the loop bounds — measured as the
    # next question (a fused FP16 w2 kernel would remove it).
    with _ProfSec("moe.w2_cublas"):
        counts_l = counts.tolist()                         # the one host sync for w2
        ero_l = e_row_off.tolist()
        eo_sorted = _t.empty(R, H, device=route_x.device, dtype=_t.float16)
        for e in range(E):
            cnt = counts_l[e]
            if cnt == 0:
                continue
            st = ero_l[e]
            eo_sorted[st:st + cnt] = hidden_sorted[st:st + cnt] @ w2[e].T

    with _ProfSec("moe.scatter"):
        expert_out = _t.empty(R, H, device=route_x.device, dtype=_t.float16)
        expert_out.index_copy_(0, order, eo_sorted)
    return expert_out


def _v100_ct_mixed_moe_routed(layer, x, topk_weights, topk_ids,
                              shared_experts_input=None):
    """Phase 2b mixed routed MoE: w13 FP8 grouped kernel + w2 FP16 per-expert.

    Mirrors `_our_moe_apply_grouped` (route -> w13 -> silu(gate)*up -> w2 ->
    route_w -> index_add scatter) but with w13 as the resident FP8 grouped kernel
    (channel scale, block_h=1, block_w=128) and w2 kept FP16. Returns the local
    (pre-all-reduce) MoE output [M, H], to match `_v100_unquant.apply`. Assumes
    dense top-k / no expert_map (TP, not EP).

    Shared expert: GLM owns it inside FusedMoE with `reduce_results=False`, so the
    shared output is LOCAL (pre-reduce) like the routed `index_add`, and (for
    GLM-Air) `routed_scaling_factor=1.0` + `norm_topk_prob` is handled upstream,
    so the combine is a plain local add `routed + shared`. The full-path
    self-check validates this against `unquant.apply`; on mismatch we fall back.
    """
    import torch as _t
    from fp8_w8a16_sm70.ext_loader import load_kernel
    from fp8_w8a16_sm70.vllm_serve import _moe_activation
    ext = load_kernel(name="fp8_dequant_ext_vllm")

    mix = layer._v100_w13_mix
    N13, H = int(mix["N"]), int(mix["H"])          # N13 = 2I_shard ; H = hidden
    I = N13 // 2
    M = int(x.shape[0])
    x2d = x.reshape(M, H).to(_t.float16).contiguous()

    topk = int(topk_ids.size(-1)) if topk_ids.dim() >= 2 else 1
    expert_ids = topk_ids.reshape(-1).to(_t.int64).contiguous()
    route_w = topk_weights.reshape(-1).to(_t.float16)
    token_idx = _t.arange(M, device=x.device).repeat_interleave(topk)
    route_x = x2d.index_select(0, token_idx)
    apply_on_input = bool(getattr(layer, "apply_router_weight_on_input", False))
    if apply_on_input:
        route_x = route_x * route_w.unsqueeze(-1)
    route_x = route_x.contiguous()

    R = route_x.size(0)
    w2 = layer.w2_weight.data                       # FP16 [E, H, I]
    # Profiler: mark prefill (large R) seen; print the breakdown on the first
    # decode-sized (small R) MoE apply that follows. No-op unless CT_PROFILE=1.
    _prof_report(R, _CT_MOE_PREFILL_TILED_MIN_R)

    # Phase 4 Stage 1: at large R (PREFILL), route through per-expert TILED GEMMs
    # (weight reuse) instead of the per-row grouped kernels — the per-row kernels
    # are ~0.24 TFLOP/s and make prefill ~94% of TTFT. Decode (small R) falls
    # through to the per-row + cudagraph path below, UNTOUCHED.
    if (_CT_MOE_PREFILL_TILED and R >= _CT_MOE_PREFILL_TILED_MIN_R
            and hasattr(ext, "fp8_w8a16_gemm_a2")):
        _use_fused = (_CT_MOE_PREFILL_FUSED
                      and hasattr(ext, "fp8_w8a16_grouped_tiled_gemm"))
        if not _CT_MOE_PREFILL_ENGAGED[0] and _ct_rank() in (0, -1):
            _CT_MOE_PREFILL_ENGAGED[0] = True
            print(f"[serve_fp8_v100 ct-moe-prefill] tiled prefill ENGAGED "
                  f"(R={R} >= {_CT_MOE_PREFILL_TILED_MIN_R}); "
                  f"{'FUSED w13(1 launch)+grouped w2' if _use_fused else 'per-expert a2(w13)+cuBLAS(w2)'}",
                  flush=True)
        if _use_fused:
            if not _CT_MOE_FUSED_ENGAGED[0] and _ct_rank() in (0, -1):
                _CT_MOE_FUSED_ENGAGED[0] = True
                print("[serve_fp8_v100 ct-moe-prefill] FUSED path active "
                      "(no per-expert loop / no .tolist on hot path)", flush=True)
            expert_out = _v100_ct_fused_prefill_moe(
                layer, route_x, expert_ids, mix, N13, H, I, w2)
        else:
            expert_out = _v100_ct_tiled_prefill_moe(
                layer, route_x, expert_ids, mix, N13, H, I, w2)
        if not apply_on_input:
            expert_out = expert_out * route_w.unsqueeze(-1)
        out = _t.zeros(M, H, device=x.device, dtype=_t.float16)
        out.index_add_(0, token_idx, expert_out)
        se = getattr(layer, "shared_experts", None)
        if shared_experts_input is not None and se is not None:
            mlp = getattr(se, "_layer", se)
            shared = mlp(shared_experts_input.reshape(M, H).to(_t.float16))
            if isinstance(shared, tuple):
                shared = shared[0]
            out = out + shared.reshape(M, H)
        return out

    # w13 (gate_up): FP8 grouped kernel -> [R, 2I]. The kernel launches with
    # gridDim.y = R (routed rows), which CUDA caps at 65535. At long-context
    # prefill R = M*topk exceeds it (e.g. MAXLEN=8192 -> M~8192*topk8 = 65536 ->
    # "invalid configuration argument"). Chunk R to stay under the limit; the
    # rest (activation, per-expert w2, index_add) are torch ops with no grid cap.
    k_split = 8 if (H % 1024 == 0) else 4 if (H % 512 == 0) else 1
    _W13_CHUNK = _CT_MOE_W13_CHUNK
    if R <= _W13_CHUNK:
        if (_CT_MOE_W13_COALESCED and H % 128 == 0
                and hasattr(ext, "fp8_w8a16_grouped_gemv_coalesced")):
            if not _CT_MOE_W13_COAL_ENGAGED[0] and _ct_rank() in (0, -1):
                _CT_MOE_W13_COAL_ENGAGED[0] = True
                print("[serve_fp8_v100 ct-moe-w13] decode w13 = grouped COALESCED "
                      "GEMV (warp->col, lanes->K); first call", flush=True)
            w13 = ext.fp8_w8a16_grouped_gemv_coalesced(
                route_x, expert_ids, mix["u8"], mix["scale"], N13, H, 1, 128)
        else:
            w13 = ext.fp8_w8a16_grouped_routed_gemm_a3(
                route_x, expert_ids, mix["u8"], mix["scale"], N13, H, 1, 128, k_split)
    else:
        _parts = []
        for _i in range(0, R, _W13_CHUNK):
            _j = min(_i + _W13_CHUNK, R)
            _parts.append(ext.fp8_w8a16_grouped_routed_gemm_a3(
                route_x[_i:_j].contiguous(), expert_ids[_i:_j].contiguous(),
                mix["u8"], mix["scale"], N13, H, 1, 128, k_split))
        w13 = _t.cat(_parts, dim=0)
    hidden = (_moe_activation(layer, w13[:, :I]) * w13[:, I:]).contiguous()  # [R, I]

    # w2 (down): FP16 grouped routed GEMM -> [R, H]. The fallback preserves the
    # original implementation and is useful for A/B or emergency isolation.
    if (_CT_MOE_W2_GROUPED and hasattr(ext, "fp16_grouped_routed_gemm")
            and hidden.size(1) % _CT_MOE_W2_K_SPLIT == 0):
        # The w2 grouped kernel launches with gridDim.y = R (routed rows), which
        # CUDA caps at 65535 — same long-context-prefill hazard the w13 path
        # chunks for (R = M*topk can exceed 65535). Chunk R identically so w2
        # doesn't crash with "invalid configuration argument" once w13 already
        # survived via its own chunking. k_split tunes K-occupancy.
        w2c = w2.contiguous()
        _R2 = hidden.size(0)
        if _R2 <= _CT_MOE_W2_CHUNK:
            expert_out = ext.fp16_grouped_routed_gemm(
                hidden, expert_ids, w2c, _CT_MOE_W2_K_SPLIT)
        else:
            _p2 = []
            for _i in range(0, _R2, _CT_MOE_W2_CHUNK):
                _j = min(_i + _CT_MOE_W2_CHUNK, _R2)
                _p2.append(ext.fp16_grouped_routed_gemm(
                    hidden[_i:_j].contiguous(), expert_ids[_i:_j].contiguous(),
                    w2c, _CT_MOE_W2_K_SPLIT))
            expert_out = _t.cat(_p2, dim=0)
    else:
        expert_out = _t.zeros(route_x.size(0), H, device=x.device, dtype=_t.float16)
        for e in _t.unique(expert_ids).tolist():
            if e < 0:
                continue                            # invalid route -> stays 0
            m = (expert_ids == e)
            expert_out[m] = hidden[m] @ w2[e].T

    if not apply_on_input:
        expert_out = expert_out * route_w.unsqueeze(-1)

    out = _t.zeros(M, H, device=x.device, dtype=_t.float16)
    out.index_add_(0, token_idx, expert_out)

    # Shared expert (owned by FusedMoE for GLM): local add (reduce_results=False).
    # layer.shared_experts is vLLM's SharedExperts wrapper, which holds the real
    # MLP (Glm4MoeMLP) in `._layer` and computes via `output = self._layer(input)`.
    se = getattr(layer, "shared_experts", None)
    if shared_experts_input is not None and se is not None:
        mlp = getattr(se, "_layer", se)      # wrapper -> ._layer ; raw module -> itself
        shared = mlp(shared_experts_input.reshape(M, H).to(_t.float16))
        if isinstance(shared, tuple):
            shared = shared[0]
        out = out + shared.reshape(M, H)
    return out


def patch_compressed_tensors_moe_for_v100() -> bool:
    """Enable compressed-tensors FP8 *MoE* (e.g. RedHatAI gemma-4-26B-A4B) on sm_70.

    The Linear hook above doesn't cover MoE experts — they route through
    `CompressedTensorsW8A8Fp8MoEMethod`, whose __init__ calls select_fp8_moe_backend
    → raises "No FP8 MoE backend supports the deployment configuration" on Volta
    (and there's NO W8A8→W8A16 MoE fallback). FIRST CUT (FP16-resident, like the
    Linear hook): dequant FP8 experts → FP16 at load and run the proven UNQUANTIZED
    FP16 fused-MoE path (the same one the BF16 26B-A4B already uses on V100).

    NOTE: FP16-resident → no FP8 memory saving (perf-equal to the BF16 model); this
    is a robustness/capability hook so any CT-FP8 MoE *loads* on V100 rather than
    hard-crashing. FP8-resident MoE (memory win) would need a channel-aware MoE
    kernel — deferred. Handles CHANNEL/TENSOR (scale broadcasts over the K dim);
    BLOCK would need scale-block expansion (added when a block CT-MoE model appears).
    """
    if not _is_volta():
        return False
    try:
        # The CT-MoE method class and its module-global `select_fp8_moe_backend`
        # (which we neutralize below) live in different places across engines:
        #   vLLM 0.21+ : split out into a `compressed_tensors_moe_w8a8_fp8` submodule
        #   vLLM 0.19  : defined inline in `compressed_tensors_moe` itself
        # `_moemod` must be the module that OWNS select_fp8_moe_backend as a global
        # (the same module where the class __init__ resolves it), so bind to whichever
        # layout this engine ships. Try the 0.21 submodule first, fall back to inline.
        try:
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
                compressed_tensors_moe_w8a8_fp8 as _moemod,
            )
        except ImportError:
            from vllm.model_executor.layers.quantization.compressed_tensors import (  # noqa: E501
                compressed_tensors_moe as _moemod,
            )
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
        from vllm.model_executor.utils import replace_parameter
    except Exception as exc:
        print(f"[serve_fp8_v100 ct-moe] not importable, skipping CT-MoE patch "
              f"({type(exc).__name__}: {exc})", flush=True)
        return False

    CTMoE = _moemod.CompressedTensorsW8A8Fp8MoEMethod

    class _V100ModularExperts:
        # Modular-kernel face for engines (vLLM 0.19) whose FP16 fused-MoE method
        # has no `experts_cls` and reports `is_monolithic=False` (it routes through
        # `moe_kernel`). The CT-MoE `is_monolithic` property calls
        # `self.experts_cls.is_monolithic()`, so a stand-in with that staticmethod
        # lets the patched method present the same modular face as the FP16 path
        # without crashing on the `None` the stubbed backend leaves behind.
        @staticmethod
        def is_monolithic():
            return False

    # 0) Neutralize the Volta-incompatible backend selection in the CT-MoE module
    #    so the original __init__ completes (fp8_backend/experts_cls -> None). We
    #    rebuild a real UNQUANTIZED backend per-instance below.
    _moemod.select_fp8_moe_backend = lambda *a, **k: (None, None)

    _orig_init = CTMoE.__init__

    def patched_init(self, weight_quant, input_quant, moe, layer_name=None):
        _orig_init(self, weight_quant, input_quant, moe, layer_name)  # backend stubbed
        # Build the proven FP16 unquantized method to borrow its kernel/apply.
        self._v100_unquant = UnquantizedFusedMoEMethod(moe)
        # so the inherited is_monolithic (reads experts_cls) matches the FP16 path.
        # Newer engines (0.21) expose experts_cls on the unquant method; older
        # layouts (0.19-tf5) don't use it — copy it across only when present so the
        # patch binds on both without an AttributeError at load.
        _ec = getattr(self._v100_unquant, "experts_cls", None)
        if _ec is not None:
            self.experts_cls = _ec          # 0.21 legacy path (unchanged)
        elif getattr(self, "experts_cls", None) is None:
            # 0.19 modular-kernel path: the FP16 method routes through moe_kernel and
            # exposes no experts_cls; the stubbed backend left experts_cls=None, which
            # would make the is_monolithic property call None.is_monolithic() and crash
            # at load. Present a modular face (is_monolithic=False) like the FP16 path;
            # moe_kernel is wired below in process_weights_after_loading.
            self.experts_cls = _V100ModularExperts

    def _dequant_expert(w_fp8, scale):
        # w_fp8: [E, R, C] float8 ; channel/tensor scale: [E, R, 1] (or [E,1,1]/[E,R]).
        wf = w_fp8.to(torch.float16)
        if scale is None:
            return wf.contiguous()
        s = scale.to(torch.float16)
        if s.dim() == 2:           # [E, R] -> [E, R, 1]
            s = s.unsqueeze(-1)
        elif s.dim() == 1:         # [E] -> [E,1,1]
            s = s.reshape(-1, 1, 1)
        return (wf * s).contiguous()

    def patched_process_weights_after_loading(self, layer):
        global _MOE_BANNERED
        strat = str(getattr(self, "weight_quant", None)
                    and self.weight_quant.strategy).upper()
        if "BLOCK" in strat:
            raise NotImplementedError(
                "serve_fp8_v100 ct-moe: BLOCK-strategy CT MoE not yet supported "
                "on V100 (needs block-scale expansion); CHANNEL/TENSOR only.")
        # Phase 2a (diagnostic only): validate the grouped channel kernel on the
        # REAL w13 experts BEFORE dequant. Execution below is unchanged (FP16).
        if _CT_MOE_SELFCHECK and ("CHANNEL" in strat):
            _ct_moe_selfcheck_w13(layer)
        # Phase 2b: stash w13 FP8 + expanded channel scale for the mixed apply,
        # captured BEFORE w13 is dequanted to FP16 just below. Guarded cut keeps
        # both reps (no memory win yet); the mixed apply validates vs FP16 then
        # auto-falls-back. Stored as a clone so it survives replace_parameter.
        layer._v100_w13_mix = None
        _cap_hit = bool(_CT_MOE_W13_MAXLAYERS) and (_CT_MOE_W13_STASH_N[0] >= _CT_MOE_W13_MAXLAYERS)
        if _CT_MOE_W13_RESIDENT and ("CHANNEL" in strat) and not _cap_hit:
            try:
                _w = layer.w13_weight.data
                _s = getattr(layer, "w13_weight_scale", None)
                if _s is not None and _w.dim() == 3 and (int(_w.size(2)) % 128 == 0):
                    E_, N_, H_ = int(_w.size(0)), int(_w.size(1)), int(_w.size(2))
                    s_ = _s.data.to(torch.float16)
                    if s_.dim() == 2:
                        s_ = s_.unsqueeze(-1)
                    elif s_.dim() == 1:
                        s_ = s_.reshape(E_, 1, 1)
                    s_ = s_.reshape(E_, N_, 1)
                    layer._v100_w13_mix = {
                        "u8": _w.contiguous().view(torch.uint8).clone(),   # [E,N,H] bytes
                        "scale": s_.expand(E_, N_, H_ // 128).contiguous(),  # [E,N,H/128]
                        "N": N_, "H": H_, "validated": None,
                    }
                    _CT_MOE_W13_STASH_N[0] += 1
            except Exception as exc:
                if _ct_rank() in (0, -1):
                    print(f"[serve_fp8_v100 ct-moe-w13] stash failed: "
                          f"{type(exc).__name__}: {exc}", flush=True)
        w13s = getattr(layer, "w13_weight_scale", None)
        w2s = getattr(layer, "w2_weight_scale", None)
        w13 = _dequant_expert(layer.w13_weight.data,
                              None if w13s is None else w13s.data)
        w2 = _dequant_expert(layer.w2_weight.data,
                             None if w2s is None else w2s.data)
        replace_parameter(layer, "w13_weight", w13)   # now FP16 [E, 2I, H]
        replace_parameter(layer, "w2_weight", w2)     # now FP16 [E, H, I]
        # Phase 2d real-weight self-check: validate the FP16 grouped w2 kernel on
        # the ACTUAL dequanted w2 (K=176 tail) vs the per-expert reference. Runs
        # only when the grouped w2 path is active (it's the kernel being checked).
        # Diagnostic only — execution is unchanged.
        if (_CT_MOE_SELFCHECK and _CT_MOE_W2_GROUPED
                and getattr(layer, "_v100_w13_mix", None) is not None):
            _ct_moe_selfcheck_w2(layer)
        for n in ("w13_weight_scale", "w2_weight_scale",
                  "w13_input_scale", "w2_input_scale"):
            if n in getattr(layer, "_parameters", {}):
                del layer._parameters[n]
            try:
                setattr(layer, n, None)
            except Exception:
                pass
        # Build the FP16 fused-MoE kernel via the unquantized path (V100-proven).
        self._v100_unquant.process_weights_after_loading(layer)
        # Expose the FP16 method's real modular kernel as our moe_kernel. It lives under
        # `.moe_kernel` on the 0.21 build but under `.kernel` on the 0.19-tf5 build (the
        # base sets `moe_kernel=None`; only the unquant subclass populates `.kernel` in
        # _setup_kernel). Either is the real, modular (is_monolithic=False) kernel.
        # Exposing it makes supports_internal_mk True so FusedMoE.maybe_init_modular_kernel
        # skips the modular wrap (whose maybe_make_prepare_finalize raises for our stubbed
        # backend), and is_monolithic=False so the runner dispatches to our apply(
        # layer, x, topk_weights, topk_ids, shared_experts_input). It is only PROBED for
        # metadata (is_monolithic / output_is_reduced / topk dtype), never executed — the
        # routed compute runs through patched_apply below. (0.21: identical to before, its
        # `.moe_kernel` is already populated; 0.19: the `.kernel` fallback fixes the gate.)
        _mk = self._v100_unquant.moe_kernel
        if _mk is None:
            _mk = getattr(self._v100_unquant, "kernel", None)
        self.moe_kernel = _mk
        self.moe_quant_config = getattr(self._v100_unquant, "moe_quant_config", None)
        # Phase 2c: with the mixed path validated, FREE the FP16 w13. The mixed
        # apply uses the stashed FP8 w13 (not the unquant kernel), so the FP16
        # copy (~368 MB/layer) is dead weight once the unquant build is done.
        # Replace with an empty placeholder; the old FP16 tensor frees when its
        # refcount drops. Trusted path: no working unquant fallback after this.
        if (_CT_MOE_W13_FREE_FP16
                and getattr(layer, "_v100_w13_mix", None) is not None):
            try:
                placeholder = torch.empty(0, dtype=layer.w13_weight.dtype,
                                          device=layer.w13_weight.device)
                replace_parameter(layer, "w13_weight", placeholder)
                layer._v100_w13_fp16_freed = True
                _CT_MOE_W13_FREED_N[0] += 1
                if _CT_MOE_W13_FREED_N[0] <= 1 and _ct_rank() in (0, -1):
                    print(f"[serve_fp8_v100 ct-moe-w13] FREED FP16 w13 "
                          f"(true-resident; FP8 stash kept). prefix="
                          f"{getattr(layer, 'prefix', '?')}", flush=True)
            except Exception as exc:
                if _ct_rank() in (0, -1):
                    print(f"[serve_fp8_v100 ct-moe-w13] free FP16 w13 failed: "
                          f"{type(exc).__name__}: {exc}", flush=True)
        if not _MOE_BANNERED:
            try:
                from vllm.distributed import get_tensor_model_parallel_rank
                rank = get_tensor_model_parallel_rank()
            except Exception:
                rank = -1
            print(f"[serve_fp8_v100 ct-moe rank={rank} pid={os.getpid()}] CT FP8 MoE "
                  f"V100 path: dequant experts FP8->FP16 (strategy={strat}) -> "
                  f"unquantized FP16 fused-MoE. FP16-resident (no FP8 mem saving).",
                  flush=True)
            _MOE_BANNERED = True

    def patched_apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
        mix = getattr(layer, "_v100_w13_mix", None)
        # Mixed (w13 FP8) path: routed-only FP8 GEMM, and reproduce the shared
        # expert STORAGE side-effect so the MoERunner's downstream combine
        # (moe_runner.py:552) picks it up — `unquant.apply` returns routed-only
        # too and the runner adds the stored shared output AFTER apply. We do NOT
        # add shared to `routed` ourselves (that double-counts; see diagnosis).
        # NOTE: no apply-level self-check here (it's BLIND to the shared-storage
        # side-effect — both sides return routed-only); validate END-TO-END text
        # vs the FP16 baseline. On any error, disable this layer's mixed path.
        if not (_CT_MOE_W13_RESIDENT and mix is not None
                and mix.get("disabled") is not True):
            return self._v100_unquant.apply(
                layer, x, topk_weights, topk_ids, shared_experts_input)
        try:
            # Return routed-only FP8. The MoERunner already STORES the shared
            # output during this forward (proven: our explicit se.apply() tripped
            # the `assert _output is None` double-store guard) and combines it
            # AFTER apply (moe_runner.py:552). So we must NOT touch shared here —
            # doing so either double-stores (assert) or, if we also run unquant
            # for a ref, consumes it (GPT's sharp edge). Just return routed.
            routed = _v100_ct_mixed_moe_routed(
                layer, x, topk_weights, topk_ids, None)   # routed-only FP8
            if not _CT_MOE_MIX_ENGAGED[0] and _ct_rank() in (0, -1):
                _CT_MOE_MIX_ENGAGED[0] = True
                print("[serve_fp8_v100 ct-moe-w13] mixed path ENGAGED "
                      "(w13 FP8 routed + runner-combined shared); first call", flush=True)
            return routed
        except Exception as exc:
            mix["disabled"] = True
            if _ct_rank() in (0, -1):
                print(f"[serve_fp8_v100 ct-moe-w13] mixed ERR->disable "
                      f"{type(exc).__name__}: {exc} "
                      f"prefix={getattr(layer, 'prefix', '?')}", flush=True)
            return self._v100_unquant.apply(
                layer, x, topk_weights, topk_ids, shared_experts_input)

    def patched_get_fused_moe_quant_config(self, layer):
        return self._v100_unquant.get_fused_moe_quant_config(layer)

    CTMoE.__init__ = patched_init
    CTMoE.process_weights_after_loading = patched_process_weights_after_loading
    CTMoE.apply = patched_apply
    CTMoE.get_fused_moe_quant_config = patched_get_fused_moe_quant_config

    print(f"[serve_fp8_v100 ct-moe pid={os.getpid()}] compressed-tensors FP8 MoE "
          f"patched for sm_70 (dequant experts->FP16 + unquantized FP16 fused-MoE). "
          f"CHANNEL/TENSOR only; FP16-resident.", flush=True)
    return True
