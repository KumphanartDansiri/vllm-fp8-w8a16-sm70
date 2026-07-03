# SPDX-License-Identifier: Apache-2.0
"""
SM70 FP8 backend selection + TurboMind adapter  (Stage E, VLLM_V100_FP8_BACKEND).

Chooses, per FP8 weight, between:
  * "ours"      -- our coalesced/tiled W8A16 GEMV (compat/fallback/control path; handles
                   block / channel / tensor scale; the only path for channel-scale W8A8 and
                   for TP shards that break block-128 alignment).
  * "turbomind" -- the upstream lmdeploy SM70 s884 FP8 (E4M3) grouped GEMM (fast path where the
                   checkpoint format matches). Consumed via prepare -> meta(k_ld,q_ld) ->
                   grouped_gemm. NEVER via the broken `_auto` entry point.

`VLLM_V100_FP8_BACKEND = ours | turbomind | auto`  (default: auto).

Eligibility for TurboMind (auto picks it ONLY IF ALL hold; else falls back to ours + reason):
  1. quant format is block-FP8 with group_size 128 (weight_block_size == (128,128));
  2. scales representable as fp32 block scales (implied by BLOCK strategy);
  3. LOCAL (post-TP-shard) dims satisfy block-128 alignment: N%128==0 and K%128==0
     -- this is where TP8 on Qwen (I/tp=64) is ruled out;
  4. prepare metadata (k_ld,q_ld) is obtainable (prepare op present);
  5. the SM70 FP8 ops are actually built into this image;
  6. NOT channel-scale / tensor-scale;
  7. the call path never uses `_auto`.

Evidence: Stage A (contract + `_auto` hazard), Stage C (perf envelope), Stage D (real
block-FP8 round-trips cos=1.0; channel rejects; TP8 block-128 break), Stage E source audit
(engine is upstream lmdeploy). See docs/FP8_ENGINE_*.md.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

BLOCK = 128
_VALID_MODES = ("ours", "turbomind", "auto")


def _log(msg: str) -> None:
    try:
        from vllm.logger import init_logger
        init_logger(__name__).info_once("[fp8-backend] %s", msg)
    except Exception:
        print(f"[fp8-backend] {msg}", flush=True)


def backend_mode() -> str:
    """Resolved VLLM_V100_FP8_BACKEND (ours|turbomind|auto); default auto."""
    m = os.environ.get("VLLM_V100_FP8_BACKEND", "auto").strip().lower()
    return m if m in _VALID_MODES else "auto"


def ops_available(need_moe: bool = False) -> bool:
    """True iff the upstream-lmdeploy SM70 FP8 ops are built into this image."""
    try:
        import torch
        C = torch.ops._C
        ok = hasattr(C, "fp8_sm70_prepare") and hasattr(C, "fp8_gemm_sm70_out")
        if need_moe:
            ok = ok and hasattr(C, "fp8_moe_gemm_sm70_out") and \
                hasattr(C, "awq_moe_build_strided_ptrs")
        return bool(ok)
    except Exception:
        return False


def turbomind_eligible(
    *,
    strategy: str,
    weight_block_size: Optional[Tuple[int, int]],
    local_n: int,
    local_k: int,
    need_moe: bool = False,
    has_ops: Optional[bool] = None,
) -> Tuple[bool, str]:
    """Codex eligibility predicate. Returns (eligible, reason).

    `local_n`/`local_k` are the PER-RANK (post-TP-shard) output/input dims of the weight;
    for MoE w2 that means K = intermediate/tp — this is where TP8 block-128 breaks.
    `has_ops` overrides op detection (for unit tests); None => auto-detect.
    """
    strat = (strategy or "").upper()
    if "BLOCK" not in strat:                                   # (1)(6) channel/tensor -> ours
        return False, f"not block-scale (strategy={strategy!r})"
    if weight_block_size is None or tuple(weight_block_size) != (BLOCK, BLOCK):  # (1)
        return False, f"weight_block_size={weight_block_size} != (128,128)"
    if local_n % BLOCK != 0 or local_k % BLOCK != 0:          # (3) local shard alignment
        return False, (f"local dims [{local_n},{local_k}] not block-128 aligned "
                       f"(likely a TP shard: N%128={local_n % BLOCK}, K%128={local_k % BLOCK})")
    if local_n <= 0 or local_k <= 0:
        return False, f"degenerate dims [{local_n},{local_k}]"
    have = ops_available(need_moe) if has_ops is None else has_ops  # (4)(5)
    if not have:
        return False, "SM70 FP8 ops not present in this image"
    return True, "block-128 eligible (group_size=128, dims aligned, ops present)"


def select_backend(
    *,
    strategy: str,
    weight_block_size: Optional[Tuple[int, int]],
    local_n: int,
    local_k: int,
    need_moe: bool = False,
    mode: Optional[str] = None,
    has_ops: Optional[bool] = None,
    quiet: bool = False,
) -> Tuple[str, str]:
    """Return (backend, reason) in {"ours","turbomind"}. Logs the choice unless quiet.

    - mode "ours"      -> always ours.
    - mode "turbomind" -> turbomind if eligible, else FAIL LOUDLY (explicit request, no silent
                          fallback -- silent wrong backend is the hazard, not the miss).
    - mode "auto"      -> turbomind iff eligible, else ours + one-line reason.
    """
    mode = (mode or backend_mode())
    elig, reason = turbomind_eligible(
        strategy=strategy, weight_block_size=weight_block_size,
        local_n=local_n, local_k=local_k, need_moe=need_moe, has_ops=has_ops)

    if mode == "ours":
        backend, why = "ours", "forced (VLLM_V100_FP8_BACKEND=ours)"
    elif mode == "turbomind":
        if not elig:
            raise RuntimeError(
                f"VLLM_V100_FP8_BACKEND=turbomind requested but not eligible: {reason}. "
                f"Refusing to fall back silently — set =auto to allow the ours fallback.")
        backend, why = "turbomind", f"forced + eligible ({reason})"
    else:  # auto
        backend, why = ("turbomind", reason) if elig else ("ours", f"fallback: {reason}")

    if not quiet:
        _log(f"backend={backend}  [{why}]  dims=[{local_n},{local_k}] moe={need_moe}")
    return backend, why


# ── TurboMind call wrappers (the required contract; NEVER `_auto`) ───────────────
# These pass straight through to the upstream-lmdeploy SM70 FP8 ops via vLLM's _custom_ops,
# always threading the packed leading dims (k_ld,q_ld) from prepare. They raise if the ops
# are absent (image without the engine) — callers must gate on select_backend()=="turbomind".

def prepare(qweight, scales, group_size: int = BLOCK):
    """Pack an FP8 [N,K] weight + block scales -> (tm_weight, tm_scales, meta[k_ld,q_ld])."""
    from vllm import _custom_ops as ops
    if scales.dtype not in (getattr(__import__("torch"), "float32"),):
        scales = scales.float()               # prepare requires fp32 block scales
    return ops.fp8_sm70_prepare(qweight, scales, group_size)


def gemm_out(out, x, tm_weight, tm_scales, k_ld, q_ld, group_size: int = BLOCK):
    """Dense FP8 W8A16 GEMM via the explicit-ld (correct) entry point. Never `_auto`."""
    from vllm import _custom_ops as ops
    ops.fp8_gemm_sm70_out(out, x, tm_weight, tm_scales, group_size, int(k_ld), int(q_ld))


def moe_gemm_out(out, sorted_x, expert_offsets, ptrs_w, ptrs_s, num_experts,
                 k, n, group_size: int = BLOCK, gated_silu: bool = False):
    """Grouped FP8 MoE GEMM. Strided ptrs carry the packed ld (built via prepare meta)."""
    from vllm import _custom_ops as ops
    ops.fp8_moe_gemm_sm70_out(out, sorted_x, expert_offsets, ptrs_w, ptrs_s,
                              int(num_experts), int(k), int(n), int(group_size), bool(gated_silu))


# ── self-test: exercises the DECISION logic without needing the engine ───────────
def _selftest() -> int:
    cases = [
        # (desc, kwargs, expect_backend)  -- has_ops forced so logic is deterministic
        ("block-128 dense, ops present",
         dict(strategy="BLOCK", weight_block_size=(128, 128), local_n=1024, local_k=2048,
              has_ops=True, mode="auto"), "turbomind"),
        ("block-128 MoE, ops present",
         dict(strategy="BLOCK", weight_block_size=(128, 128), local_n=1024, local_k=2048,
              need_moe=True, has_ops=True, mode="auto"), "turbomind"),
        ("channel-scale -> ours",
         dict(strategy="CHANNEL", weight_block_size=None, local_n=4096, local_k=1408,
              has_ops=True, mode="auto"), "ours"),
        ("tensor-scale -> ours",
         dict(strategy="TENSOR", weight_block_size=None, local_n=4096, local_k=4096,
              has_ops=True, mode="auto"), "ours"),
        ("TP8 shard I/tp=64 breaks block-128 -> ours",
         dict(strategy="BLOCK", weight_block_size=(128, 128), local_n=4096, local_k=64,
              has_ops=True, mode="auto"), "ours"),
        ("ops absent -> ours (even if block-128)",
         dict(strategy="BLOCK", weight_block_size=(128, 128), local_n=1024, local_k=2048,
              has_ops=False, mode="auto"), "ours"),
        ("mode=ours forces ours",
         dict(strategy="BLOCK", weight_block_size=(128, 128), local_n=1024, local_k=2048,
              has_ops=True, mode="ours"), "ours"),
    ]
    fails = 0
    for desc, kw, expect in cases:
        got, why = select_backend(quiet=True, **kw)
        ok = got == expect
        fails += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] {desc:48s} -> {got:9s} ({why})")

    # mode=turbomind must FAIL LOUDLY when ineligible
    try:
        select_backend(strategy="CHANNEL", weight_block_size=None, local_n=8, local_k=8,
                       has_ops=True, mode="turbomind", quiet=True)
        print("  [FAIL] mode=turbomind on ineligible should raise"); fails += 1
    except RuntimeError:
        print("  [ok] mode=turbomind on ineligible raises (no silent fallback)")

    print(f"\nselftest: {'PASS' if fails == 0 else f'FAIL ({fails})'}")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
