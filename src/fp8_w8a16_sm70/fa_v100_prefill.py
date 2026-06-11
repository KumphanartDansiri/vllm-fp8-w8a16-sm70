# ======================================================================================
# V100 FlashAttention prefill interpose (Stage 1).
#
# Routes PREFILL attention batches (max_seqlen_q > 1) from vLLM 0.21's TRITON_ATTN
# backend to the ai-bond flash-attention-v100 kernel (8.4x faster at 26k: 112.6 vs
# 945.4 ms/layer, docs/FA_V100_AUDIT.md Turn 8). Decode batches (max_seqlen_q == 1)
# fall through to the original Triton unified_attention UNTOUCHED — the validated
# 30.7 tok/s FULL_DECODE_ONLY cudagraph path is preserved by construction (decode-only
# graphs never contain a prefill batch, so capture never sees the ai-bond path).
#
# Design (audit Turns 1-9, Claude+Codex converged):
#   - Interpose at unified_attention's STANDARD contract (q, k_cache, v_cache, out,
#     cu_seqlens_q, seqused_k, block_table, ...) — same paged layout ai-bond consumes
#     natively. No backend swap, no API relayout, no caller changes.
#   - Calls LOW-LEVEL flash_attn_v100_cuda.varlen_fwd (the public python wrapper
#     hardcodes seqused_k=None/out=None — audit Turn 5).
#   - Synthesizes length-valid cu_seqlens_k[i] = i*max_seqlen_k (kernel computes
#     seqlen_k = min(cu_diff, seqused_k); a zero dummy breaks attention — Turn 2).
#   - Requires kv-cache block_size % 256 == 0 (ai-bond page constraint; launch with
#     --block-size 256) and the BLOCK_N_128=128 tile fix (Turn 7/8 straddle bug).
#   - fp16 KV only; any unsupported feature falls back to Triton per-call.
#
# Kill switch: VLLM_V100_FLASH_ATTN=0 (default OFF until e2e A/B validates).
# ======================================================================================
import os

import torch

_ENABLED = os.environ.get("VLLM_V100_FLASH_ATTN", "0").lower() not in (
    "0", "off", "false", "")

_fa_cuda = None                 # flash_attn_v100_cuda module (lazy)
_cu_k_cache = {}                # (B, max_sk, dev) -> synthesized cu_seqlens_k
_logged_route = False
_logged_densify = False
_logged_fallback_reasons = set()


def _log(msg):
    print(f"[fa_v100_prefill pid={os.getpid()}] {msg}", flush=True)


def _fallback(reason, orig, kwargs):
    global _logged_fallback_reasons
    if reason not in _logged_fallback_reasons:
        _logged_fallback_reasons.add(reason)
        _log(f"fallback to Triton ({reason})")
    return orig(**kwargs)


def _synth_cu_seqlens_k(batch, max_seqlen_k, device):
    key = (batch, max_seqlen_k, device.index)
    t = _cu_k_cache.get(key)
    if t is None:
        # length-valid: per-row diff = max_seqlen_k >= seqused_k[i] -> kernel's
        # min(diff, seqused_k) yields exactly seqused_k[i]
        t = torch.arange(batch + 1, dtype=torch.int32, device=device) * max_seqlen_k
        _cu_k_cache.clear()     # one live entry; shapes change between steps
        _cu_k_cache[key] = t
    return t


def _dense_block_layout(t):
    # ai-bond indexes within a block as dense [block_size, H_k, D] row-major and
    # uses t.stride(0) for the block stride — so strided views from
    # kv_cache.unbind() are fine as long as the inner 3 dims are dense.
    return (t.stride(-1) == 1
            and t.stride(2) == t.size(3)
            and t.stride(1) == t.size(2) * t.size(3))


def patch_triton_prefill_for_v100():
    """Wrap TRITON_ATTN's unified_attention; prefill -> ai-bond, decode -> Triton."""
    global _fa_cuda
    if not _ENABLED:
        _log("disabled (VLLM_V100_FLASH_ATTN unset)")
        return

    import flash_attn_v100_cuda as _fa  # raises -> caller's try/except = no-op
    _fa_cuda = _fa

    import vllm.v1.attention.backends.triton_attn as triton_mod
    orig = triton_mod.unified_attention

    def unified_attention_v100(
        q, k, v, out,
        cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
        softmax_scale, causal, window_size, block_table, softcap,
        q_descale, k_descale, v_descale,
        **kw,
    ):
        global _logged_route
        kwargs = dict(
            q=q, k=k, v=v, out=out, cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q, seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale,
            causal=causal, window_size=window_size, block_table=block_table,
            softcap=softcap, q_descale=q_descale, k_descale=k_descale,
            v_descale=v_descale, **kw)

        # ── routing gates: anything unsupported -> original Triton ──────────
        if max_seqlen_q <= 1:
            return orig(**kwargs)           # decode: untouched (incl. cudagraph)
        if block_table is None or seqused_k is None:
            return _fallback("no block_table/seqused_k", orig, kwargs)
        if q.dtype != torch.float16:
            return _fallback(f"dtype {q.dtype}", orig, kwargs)
        if q.size(2) not in (64, 128, 256):
            # ai-bond dispatches 16/32/64/128/256; we only route dims we've gated
            # (128 fully e2e-proven; 64/256 page-safe tiles: 256%BLOCK_N==0).
            # D=256 still needs its own longseq+perf gate before serving Qwen3.5/
            # Gemma-4 — this check prevents a kernel TORCH_CHECK crash for others.
            return _fallback(f"head_dim {q.size(2)}", orig, kwargs)
        if window_size is not None and tuple(window_size) != (-1, -1):
            # ai-bond HAS a window path but it is UNVALIDATED by our gates;
            # Gemma-4 sliding layers (window 1024) must stay on Triton until a
            # window correctness gate passes.
            return _fallback(f"sliding_window {tuple(window_size)}", orig, kwargs)
        if k.size(1) % 256 != 0:
            return _fallback(f"block_size {k.size(1)} %256!=0", orig, kwargs)
        if not (_dense_block_layout(k) and _dense_block_layout(v)):
            return _fallback("non-dense kv block layout", orig, kwargs)
        if q_descale is not None:
            return _fallback("q_descale", orig, kwargs)
        # k_descale/v_descale are NOT a quantization signal: the triton backend
        # always passes 1.0-filled layer._k/_v_scale.expand(...) tensors even for
        # fp16 "auto" KV cache (triton_attn.py forward); they are only consumed
        # when the cache dtype is fp8. Gate on the cache dtype itself instead.
        if k.dtype != torch.float16 or v.dtype != torch.float16:
            return _fallback(f"kv cache dtype {k.dtype}", orig, kwargs)
        if kw.get("alibi_slopes") is not None:
            return _fallback("alibi", orig, kwargs)
        if kw.get("sinks") is not None:
            return _fallback("attention sinks", orig, kwargs)
        if kw.get("output_scale") is not None or kw.get("qq_bias") is not None:
            return _fallback("output_scale/qq_bias", orig, kwargs)
        if kw.get("mm_prefix_range") is not None:
            return _fallback("prefix-lm range", orig, kwargs)
        if kw.get("chunk_lookback", -1) not in (-1, None):
            return _fallback("chunked-attention lookback", orig, kwargs)
        kvq = kw.get("kv_quant_mode")
        if kvq is not None and getattr(kvq, "name", "NONE") != "NONE":
            return _fallback(f"kv_quant_mode {kvq}", orig, kwargs)

        wl, wr = window_size if window_size is not None else (-1, -1)
        scale = softmax_scale if softmax_scale is not None else q.size(-1) ** -0.5
        batch = cu_seqlens_q.numel() - 1
        cu_k = _synth_cu_seqlens_k(batch, max_seqlen_k, q.device)

        # ai-bond's low-level kernel assumes DENSE [T, H*D] q/out rows (it only
        # checks stride(-1)==1) — but vLLM passes q as a .split() view of the
        # fused QKV projection (row stride = full qkv width, e.g. 1792 vs 1536
        # for GLM-Air/rank) -> silent garbage (longseq_check qkv-split case).
        # Densify q (one ~0.2ms copy at 24k vs ~5s attention); same guard for out.
        hqd = q.size(1) * q.size(2)
        if q.stride(0) != hqd or q.stride(1) != q.size(2):
            global _logged_densify
            if not _logged_densify:
                _logged_densify = True
                _log(f"densify q stride={tuple(q.stride())} -> contiguous "
                     f"({hqd}-dense; vLLM qkv-split view, ~0.2ms @24k)")
            q = q.contiguous()
        out_dense = out.stride(0) == hqd and out.stride(1) == out.size(2)
        out_buf = out if out_dense else torch.empty_like(q)

        if not _logged_route:
            _logged_route = True
            _log(f"prefill -> flash_attn_v100 (T_Q={q.size(0)} B={batch} "
                 f"Hq={q.size(1)} Hk={k.size(2)} D={q.size(2)} "
                 f"block={k.size(1)} max_sk={max_seqlen_k})")

        # 22-arg low-level contract validated on-GPU (audit Turn 8):
        # q,k,v,out,cu_q,cu_k,seqused_k,leftpad,block_table,alibi,max_sq,max_sk,
        # dropout,scale,zero_tensors,causal,win_l,win_r,softcap,ret_softmax,gen,splits
        _fa_cuda.varlen_fwd(
            q, k, v, out_buf,
            cu_seqlens_q, cu_k,
            seqused_k, None,
            block_table, None,
            max_seqlen_q, max_seqlen_k,
            0.0, scale,
            False, causal,
            int(wl), int(wr), float(softcap or 0.0),
            False, None, 0,
        )
        if out_buf is not out:
            out.copy_(out_buf)
        return  # out written in-place, mirroring unified_attention

    triton_mod.unified_attention = unified_attention_v100
    _log("TRITON_ATTN prefill interposed with flash_attn_v100 "
         "(decode untouched; kill switch VLLM_V100_FLASH_ATTN=0)")
