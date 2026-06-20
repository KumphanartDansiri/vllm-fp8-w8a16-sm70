# ======================================================================================
# V100 MLA-prefill interpose — unblock MLA models (GLM-4.7-Flash, DeepSeek-V2/V3) on sm_70.
#
# THE BLOCKER (docs ref_mla_prefill_blocks_volta):
#   vLLM 0.21 splits MLA into a *prefill* backend subsystem
#   (vllm/v1/attention/backends/mla/prefill/). For "Hopper (SM90) and older" — which
#   includes Volta sm_70 — the ONLY prefill backend offered is FLASH_ATTN, whose
#   _flash_attn_varlen_diff_headdims() calls vLLM's vendored flash_attn, which rejects
#   sm_70 at kernel-call time ("FlashAttention only supports Ampere GPUs or newer").
#   The model LOADS and DECODES fine (TritonMLA decode is sm-agnostic); only the FIRST
#   token (MLA prefill) crashes with an engine 500. The other prefill backends
#   (FLASHINFER/TRTLLM_RAGGED/TOKENSPEED_MLA) are Blackwell-only; there is no
#   Triton/naive/SDPA MLA prefill anywhere in the tree.
#
# THE FIX (one method, no kernel change, no selector patch):
#   Replace FlashAttnPrefillBackend._flash_attn_varlen_diff_headdims — the single choke
#   point both run_prefill_new_tokens (causal self-attn over new tokens) and
#   run_prefill_context_chunk (full attn over cached context, returns LSE) go through —
#   with one that routes to the ai-bond flash-attention-v100 DENSE varlen kernel.
#
#   The backend is still SELECTED (FlashAttnPrefillBackend.is_available() is True because
#   the flash_attn_varlen symbol exists on CUDA); the crash was only at *call* time, so
#   intercepting the method is sufficient — no selector/registry patch needed.
#
# WHY ai-bond fits MLA natively (verified against GLM-4.7-Flash config 2026-06-15):
#   GLM-4.7-Flash MLA dims: qk_nope_head_dim=192 + qk_rope_head_dim=64 => qk_head_dim=256,
#   v_head_dim=256. ai-bond's monomorphic kernel compiles D in {16,32,64,128,256}, so
#   D=256 is an EXACT tile — q/k/v need NO padding and there is no asymmetric-head case.
#   (DeepSeek-style qk=192/v=128 is also reachable: zero-pad q/k/v up to D=256; zeros
#   contribute 0 to QK^T and to the V-weighted sum, so the result is exact when the
#   softmax_scale of the TRUE qk dim is passed — which vLLM passes as self.scale.)
#
#   The MLA prefill q/k/v are DENSE (cu_seqlens-packed [T, H, D], block_table=None), not
#   paged — so the ai-bond paged-page-size%256 constraint and the D=128 tile/page straddle
#   bug DO NOT apply here (both are paged-path only). LSE layout matches: ai-bond emits
#   [H_Q, T_Q] (kernel fused_mha_forward_varlen.cu:128) == vLLM merge_attn_states
#   [NUM_HEADS, NUM_TOKENS] — so the chunked-context merge is correct.
#
# DEPENDENCY: only the compiled flash_attn_v100_cuda.so on PYTHONPATH (the ai-bond
#   `flash_attn` python shim must NOT be importable — it half-satisfies vLLM's optional
#   flash-attn probe and crashes GLM4 rotary init, common.py:138). We therefore call the
#   LOW-LEVEL flash_attn_v100_cuda.varlen_fwd directly (mirroring fa_v100_prefill.py).
#
# Kill switch: VLLM_V100_MLA_PREFILL=0 (default OFF — opt-in; without it MLA models keep
#   crashing exactly as before, so this is purely additive). fp16 KV only.
# ======================================================================================
import os

import torch

_ENABLED = os.environ.get("VLLM_V100_MLA_PREFILL", "0").lower() not in (
    "0", "off", "false", "")

# ai-bond compiles these head dims; q/k/v get zero-padded UP to the smallest one >= qk_d.
_AIBOND_DIMS = (16, 32, 64, 128, 256)

_fa_cuda = None                 # flash_attn_v100_cuda module (lazy)
_logged_route = False
_logged_fallback_reasons = set()


def _log(msg):
    print(f"[fa_v100_mla pid={os.getpid()}] {msg}", flush=True)


def _target_dim(qk_d):
    for d in _AIBOND_DIMS:
        if qk_d <= d:
            return d
    return None                 # qk_head_dim > 256 -> unsupported


def _pad_last(t, to):
    if t.shape[-1] == to:
        return t.contiguous()
    return torch.nn.functional.pad(t, [0, to - t.shape[-1]], value=0).contiguous()


def patch_mla_prefill_for_v100():
    """Route vLLM's MLA prefill through ai-bond on sm_70 (vLLM 0.21 AND 0.19).

    The choke point `_flash_attn_varlen_diff_headdims` has a byte-identical
    signature on both engines, but lives in a different class:
      * 0.21 split MLA prefill into a dedicated backend subsystem, so the method
        lives on `vllm.v1.attention.backends.mla.prefill.flash_attn.
        FlashAttnPrefillBackend`.
      * 0.19 has no `mla/prefill/` subsystem — prefill is inline on the MLA impl
        base `vllm.model_executor.layers.attention.mla_attention.MLACommonImpl`.
        The V100 decode backend `TritonMLAImpl` overrides this method as a pure
        pass-through to `super()`, so patching the base catches it via super().
    Both `_run_prefill_new_tokens_fa` (causal) and `_run_prefill_context_chunk_fa`
    (non-causal, returns LSE) funnel through it on both engines.
    """
    global _fa_cuda
    if not _ENABLED:
        _log("disabled (VLLM_V100_MLA_PREFILL unset)")
        return

    import flash_attn_v100_cuda as _fa   # raises -> caller's try/except = no-op
    _fa_cuda = _fa

    try:                                 # vLLM 0.21: dedicated prefill subsystem
        from vllm.v1.attention.backends.mla.prefill.flash_attn import (
            FlashAttnPrefillBackend as _prefill_cls,
        )
        _engine = "0.21 (FlashAttnPrefillBackend)"
    except ModuleNotFoundError:          # vLLM 0.19: inline on the MLA impl base
        from vllm.model_executor.layers.attention.mla_attention import (
            MLACommonImpl as _prefill_cls,
        )
        _engine = "0.19 (MLACommonImpl base; reached via TritonMLAImpl super())"
    orig = _prefill_cls._flash_attn_varlen_diff_headdims

    def _diff_headdims_v100(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs,
    ):
        global _logged_route

        def _fallback(reason):
            if reason not in _logged_fallback_reasons:
                _logged_fallback_reasons.add(reason)
                _log(f"fallback to vLLM FA ({reason}) — will crash on sm_70 if "
                     f"no working backend; gate left intact for off-Volta safety")
            return orig(self, q, k, v, return_softmax_lse=return_softmax_lse,
                        softmax_scale=softmax_scale, **kwargs)

        # ── routing gates ─────────────────────────────────────────────────
        if q.dtype != torch.float16 or k.dtype != torch.float16 \
                or v.dtype != torch.float16:
            return _fallback(f"non-fp16 q/k/v ({q.dtype}/{k.dtype}/{v.dtype})")

        qk_d = q.shape[-1]
        v_d = v.shape[-1]
        tgt = _target_dim(qk_d)
        if tgt is None:
            return _fallback(f"qk_head_dim {qk_d} > 256")
        if v_d > tgt:
            return _fallback(f"v_head_dim {v_d} > target {tgt}")

        try:
            cu_q = kwargs["cu_seqlens_q"]
            cu_k = kwargs["cu_seqlens_k"]
            max_q = int(kwargs["max_seqlen_q"])
            max_k = int(kwargs["max_seqlen_k"])
        except KeyError as e:
            return _fallback(f"missing varlen kwarg {e}")
        causal = bool(kwargs.get("causal", False))
        scale = float(softmax_scale if softmax_scale is not None
                      else qk_d ** -0.5)

        # Zero-pad q/k/v up to the ai-bond tile (no-op when qk_d == v_d == 256, the
        # GLM-4.7-Flash case). Padding is exact: extra dims are 0 in QK^T and in the
        # V-weighted sum, and `scale` is the scale of the TRUE qk dim (vLLM's self.scale).
        qp = _pad_last(q, tgt)
        kp = _pad_last(k, tgt)
        vp = _pad_last(v, tgt)

        if not _logged_route:
            _logged_route = True
            _log(f"MLA prefill -> flash_attn_v100 (qk_d={qk_d} v_d={v_d} tile={tgt} "
                 f"Hq={q.shape[1]} Hk={k.shape[1]} causal={causal} "
                 f"max_q={max_q} max_k={max_k} scale={scale:.6g})")

        # Low-level DENSE varlen contract (block_table=None), mirroring the public
        # wrapper's inner forward (flash_attn_interface.py:210):
        #   q,k,v, out=None, cu_q, cu_k, seqused_k=None, leftpad_k=None,
        #   block_table=None, alibi=None, max_q, max_k, dropout=0.0, scale,
        #   zero_tensors=False, causal, win_l=-1, win_r=-1, softcap=0.0,
        #   return_softmax=False, gen=None, num_splits=0
        out, lse, _dmask, _rng = _fa_cuda.varlen_fwd(
            qp, kp, vp, None,
            cu_q, cu_k,
            None, None,
            None, None,
            max_q, max_k,
            0.0, scale,
            False, causal,
            -1, -1, 0.0,
            False, None, 0,
        )

        # out: [T_Q, H_Q, tile] -> slice back to the real value head dim.
        attn_out = out[..., :v_d]
        if return_softmax_lse:
            # lse: [H_Q, T_Q] == merge_attn_states [NUM_HEADS, NUM_TOKENS].
            return attn_out, lse
        return attn_out

    _prefill_cls._flash_attn_varlen_diff_headdims = _diff_headdims_v100
    _log(f"MLA prefill interposed with flash_attn_v100 on vLLM {_engine} "
         "(decode stays TritonMLA; kill switch VLLM_V100_MLA_PREFILL=0)")

    # Unblocking prefill exposes the next wall (TritonMLA decode smem overflow);
    # fix it in the same breath so MLA actually generates end-to-end on V100.
    try:
        patch_mla_decode_smem_for_v100()
    except Exception as exc:  # noqa: BLE001
        _log(f"MLA decode smem-clamp skipped: {type(exc).__name__}: {exc}")

    # EXPERIMENTAL, separately gated (VLLM_V100_MLA_DECODE_CUDAGRAPH=1): try to let
    # TritonMLA decode be cudagraph-captured (default OFF -> decode stays eager).
    try:
        patch_mla_decode_cudagraph_for_v100()
    except Exception as exc:  # noqa: BLE001
        _log(f"MLA decode cudagraph override skipped: {type(exc).__name__}: {exc}")


def patch_mla_decode_smem_for_v100():
    """Make the TritonMLA *decode* kernel fit V100's 96KB shared-memory limit.

    Unblocking MLA prefill (above) exposes the NEXT wall: TritonMLA decode runs
    `_fwd_grouped_kernel_stage1` (triton_decode_attention.py), which for MLA latent
    dims (kv_lora_rank+qk_rope = 512+64 -> BLOCK_DMODEL=512, BLOCK_DPE=64) requests
    ~100KB shared memory at the default num_stages=2 — over V100's 98304 B hardware
    limit -> `triton.runtime.errors.OutOfResources` -> EngineCore dies on the first
    decode step. vLLM already uses num_stages=1 as its own fix for this overflow, but
    only guards `BLOCK_DMODEL >= 1024`; the 512 case slips through.

    Fix: intercept the kernel launch and, ONLY for the MLA-sized latent
    (BLOCK_DMODEL >= 256), shrink the per-iteration KV tile BLOCK_N 32 -> 16. The
    overflow is small (102400 vs 98304 = 4KB over), and for MLA the source keeps a
    SINGLE c_kv smem buffer sized by BLOCK_N (num_stages does NOT multiply it — the
    "single c_kv in smem" note at _decode_grouped_att_m_fwd), so halving BLOCK_N
    halves the dominant buffer -> well under 96KB. BLOCK_N is the online-softmax loop
    chunk size: smaller = more iterations, identical result (correctness-neutral) and
    it does NOT enter the grid calc, so no launch-shape mismatch. We also drop
    num_stages 2 -> 1 (free; harmless even though it doesn't move smem for MLA).
    Volta-only (sm_7x); leaves small-head GQA decode (BLOCK_DMODEL <= 128) and all
    newer GPUs untouched.
    """
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    if cap[0] != 7:
        _log(f"MLA decode smem-clamp skipped (sm_{cap[0]}{cap[1]} != Volta)")
        return

    import vllm.v1.attention.ops.triton_decode_attention as tda
    orig_kernel = tda._fwd_grouped_kernel_stage1

    class _SmemClampedKernel:
        """Proxy over the Triton JITFunction: clamps num_stages at launch for the
        MLA-sized latent. _decode_grouped_att_m_fwd resolves the kernel name from
        module globals at call time, so reassigning the attribute reroutes it."""

        def __init__(self, kernel):
            self._kernel = kernel

        def __getitem__(self, grid):
            launcher = self._kernel[grid]

            def run(*args, **kw):
                if kw.get("BLOCK_DMODEL", 0) >= 256:
                    if kw.get("BLOCK_N", 0) > 16:
                        kw["BLOCK_N"] = 16        # halve the dominant c_kv smem tile
                    if kw.get("num_stages", 1) > 1:
                        kw["num_stages"] = 1
                return launcher(*args, **kw)

            return run

        def __getattr__(self, name):
            return getattr(self._kernel, name)

    tda._fwd_grouped_kernel_stage1 = _SmemClampedKernel(orig_kernel)
    _log("TritonMLA decode kernel smem-clamped for V100 "
         "(BLOCK_DMODEL>=256 -> BLOCK_N=16 + num_stages=1; fits 96KB)")


def patch_mla_decode_cudagraph_for_v100():
    """EXPERIMENTAL (VLLM_V100_MLA_DECODE_CUDAGRAPH=1): let TritonMLA decode be
    cudagraph-captured on V100.

    TritonMLA is the only V100-compatible MLA decode backend, but its metadata
    builder (MLACommonMetadataBuilder) inherits the base default
    `_cudagraph_support = AttentionCGSupport.NEVER` — every MLA backend that DOES
    declare UNIFORM_BATCH (flashmla/flashinfer_mla/flashattn_mla/...) is sm80+ only.
    Consequence: vLLM silently downgrades `cudagraph_mode=FULL_DECODE_ONLY` -> NONE
    (compilation.py:1346-1369) so MLA decode runs EAGER -> ~6 tok/s, with none of the
    project's cudagraph decode speedup.

    This override declares UNIFORM_SINGLE_TOKEN_DECODE (the exact regime
    FULL_DECODE_ONLY captures: uniform batch of single-token decodes), so capture is
    attempted. It is GATED + opt-in because TritonMLA's split-KV decode was never
    validated under capture: if the captured region has a host-dependent launch it
    will fail at capture time (informative). Pure decode batches only — prefill
    (our ai-bond interpose) stays eager and is never captured.
    """
    if os.environ.get("VLLM_V100_MLA_DECODE_CUDAGRAPH", "0").lower() in (
            "0", "off", "false", ""):
        return
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    if cap[0] != 7:
        return
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonMetadataBuilder,
    )
    MLACommonMetadataBuilder._cudagraph_support = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE)
    _log("EXPERIMENTAL: TritonMLA decode _cudagraph_support -> "
         "UNIFORM_SINGLE_TOKEN_DECODE (FULL_DECODE_ONLY capture enabled; "
         "VLLM_V100_MLA_DECODE_CUDAGRAPH=0 to disable)")
