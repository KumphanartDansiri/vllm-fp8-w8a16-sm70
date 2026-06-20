#!/usr/bin/env python3
# ======================================================================================
# Numerical correctness check for the V100 MLA-prefill interpose
# (src/fp8_w8a16_sm70/fa_v100_mla_prefill.py) — validates the ai-bond dense varlen
# kernel against an fp32 torch reference for the EXACT shapes vLLM's MLA prefill emits,
# WITHOUT loading a model. Run inside the cu126 serving image with the ai-bond
# flash_attn_v100_cuda.so on PYTHONPATH (see mla_prefill_v100_numcheck.sh).
#
# Cases (mirror vllm/model_executor/layers/attention/mla_attention.py):
#   A. new_tokens  : causal=True,  Sq==Sk   (run_prefill_new_tokens)
#   B. context     : causal=False, Sq< Sk   (run_prefill_context_chunk; LSE consumed
#                     by merge_attn_states -> LSE shape/value MUST match)
#   C. merge       : two context chunks merged via the SAME log-sum-exp math
#                     merge_attn_states uses -> proves chunked-prefill coherence.
#
# Shapes:
#   GLM-4.7-Flash : qk_head_dim=256 (192 nope+64 rope), v_head_dim=256  -> NO padding
#   DeepSeek-like : qk_head_dim=192, v_head_dim=128                     -> pad to 256
# ======================================================================================
import sys

import torch

try:
    import flash_attn_v100_cuda as fa
except Exception as e:  # noqa: BLE001
    print(f"FAIL: cannot import flash_attn_v100_cuda ({type(e).__name__}: {e})")
    print("  -> stage the ai-bond .so on PYTHONPATH (use mla_prefill_v100_numcheck.sh)")
    sys.exit(3)

DEV = "cuda"
AIBOND_DIMS = (16, 32, 64, 128, 256)


def target_dim(d):
    return next((x for x in AIBOND_DIMS if d <= x), None)


def pad_last(t, to):
    return t if t.shape[-1] == to else torch.nn.functional.pad(t, [0, to - t.shape[-1]])


def aibond_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal):
    """Mirror fa_v100_mla_prefill: pad to ai-bond tile, low-level dense varlen_fwd."""
    qk_d, v_d = q.shape[-1], v.shape[-1]
    tgt = target_dim(qk_d)
    qp = pad_last(q, tgt).contiguous()
    kp = pad_last(k, tgt).contiguous()
    vp = pad_last(v, tgt).contiguous()
    out, lse, _dm, _rng = fa.varlen_fwd(
        qp, kp, vp, None, cu_q, cu_k, None, None, None, None,
        int(max_q), int(max_k), 0.0, float(scale), False, bool(causal),
        -1, -1, 0.0, False, None, 0,
    )
    return out[..., :v_d], lse          # out [T_q,Hq,v_d], lse [Hq,T_q]


def ref_varlen(q, k, v, cu_q, cu_k, scale, causal):
    """fp32 reference. Returns out [T_q,Hq,v_d], lse [Hq,T_q] (natural-log full LSE)."""
    B = cu_q.numel() - 1
    Hq, v_d = q.shape[1], v.shape[2]
    out = torch.zeros(q.shape[0], Hq, v_d, dtype=torch.float32, device=DEV)
    lse = torch.full((Hq, q.shape[0]), float("-inf"), dtype=torch.float32, device=DEV)
    for b in range(B):
        qs, qe = int(cu_q[b]), int(cu_q[b + 1])
        ks, ke = int(cu_k[b]), int(cu_k[b + 1])
        qb = q[qs:qe].float().permute(1, 0, 2)          # [Hq,Sq,D]
        kb = k[ks:ke].float().permute(1, 0, 2)          # [Hk,Sk,D]
        vb = v[ks:ke].float().permute(1, 0, 2)          # [Hk,Sk,Dv]
        s = torch.matmul(qb, kb.transpose(-1, -2)) * scale   # [Hq,Sq,Sk]
        Sq, Sk = qe - qs, ke - ks
        if causal:
            # align like FA: query i attends keys j <= j0+i, j0 = Sk-Sq
            j0 = Sk - Sq
            ii = torch.arange(Sq, device=DEV).view(Sq, 1)
            jj = torch.arange(Sk, device=DEV).view(1, Sk)
            s = s.masked_fill(jj > (ii + j0), float("-inf"))
        m = s.max(dim=-1, keepdim=True).values
        p = torch.exp(s - m)
        denom = p.sum(dim=-1, keepdim=True)
        ob = torch.matmul(p / denom, vb)                # [Hq,Sq,Dv]
        out[qs:qe] = ob.permute(1, 0, 2)
        lse[:, qs:qe] = (m.squeeze(-1) + torch.log(denom.squeeze(-1)))
    return out, lse


def merge(o1, l1, o2, l2):
    """log-sum-exp merge of two partial attentions (== merge_attn_states math)."""
    m = torch.maximum(l1, l2)
    w1 = torch.exp(l1 - m).unsqueeze(-1)                # [Hq,T,1]
    w2 = torch.exp(l2 - m).unsqueeze(-1)
    o = (o1.permute(1, 0, 2) * w1 + o2.permute(1, 0, 2) * w2) / (w1 + w2)
    lse = m + torch.log(torch.exp(l1 - m) + torch.exp(l2 - m))
    return o.permute(1, 0, 2), lse


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-20))


def make_varlen(seqs_q, seqs_k, Hq, Hk, qk_d, v_d):
    Tq, Tk = sum(seqs_q), sum(seqs_k)
    g = torch.Generator(device=DEV).manual_seed(0)
    q = (torch.randn(Tq, Hq, qk_d, dtype=torch.float16, device=DEV, generator=g) * 0.5)
    k = (torch.randn(Tk, Hk, qk_d, dtype=torch.float16, device=DEV, generator=g) * 0.5)
    v = (torch.randn(Tk, Hk, v_d, dtype=torch.float16, device=DEV, generator=g) * 0.5)
    cu_q = torch.tensor([0, *torch.cumsum(torch.tensor(seqs_q), 0).tolist()],
                        dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, *torch.cumsum(torch.tensor(seqs_k), 0).tolist()],
                        dtype=torch.int32, device=DEV)
    return q, k, v, cu_q, cu_k


def check(name, qk_d, v_d, scale, fails):
    Hq = Hk = 20                                        # GLM-4.7-Flash heads (MHA, no GQA)
    print(f"\n=== {name}: qk_d={qk_d} v_d={v_d} tile={target_dim(qk_d)} "
          f"Hq={Hq} scale={scale:.6g} ===")

    # A. new_tokens — causal, Sq==Sk
    sq = [37, 200, 1, 512]
    q, k, v, cu_q, cu_k = make_varlen(sq, sq, Hq, Hk, qk_d, v_d)
    o, l = aibond_varlen(q, k, v, cu_q, cu_k, max(sq), max(sq), scale, True)
    ro, rl = ref_varlen(q, k, v, cu_q, cu_k, scale, True)
    co, cl = cos(o, ro), cos(l, rl)
    print(f"A new_tokens(causal,Sq==Sk):  out cos={co:.6f}  lse cos={cl:.6f}")
    fails.append(("A " + name, co < 0.999 or cl < 0.999))

    # B. context — non-causal, Sq < Sk
    sq2 = [8, 16, 1, 32]
    sk2 = [128, 256, 64, 300]
    q, k, v, cu_q, cu_k = make_varlen(sq2, sk2, Hq, Hk, qk_d, v_d)
    o, l = aibond_varlen(q, k, v, cu_q, cu_k, max(sq2), max(sk2), scale, False)
    ro, rl = ref_varlen(q, k, v, cu_q, cu_k, scale, False)
    co, cl = cos(o, ro), cos(l, rl)
    print(f"B context(non-causal,Sq<Sk):  out cos={co:.6f}  lse cos={cl:.6f}")
    fails.append(("B " + name, co < 0.999 or cl < 0.999))

    # C. two-chunk merge (chunked-prefill coherence via LSE)
    qN = [8, 16, 1, 32]
    k1 = [64, 100, 32, 150]
    k2 = [64, 156, 32, 150]
    q1, ka, va, cuq, cuk1 = make_varlen(qN, k1, Hq, Hk, qk_d, v_d)
    _, kb, vb, _, cuk2 = make_varlen(qN, k2, Hq, Hk, qk_d, v_d)
    o1, l1 = aibond_varlen(q1, ka, va, cuq, cuk1, max(qN), max(k1), scale, False)
    o2, l2 = aibond_varlen(q1, kb, vb, cuq, cuk2, max(qN), max(k2), scale, False)
    om, lm = merge(o1.float(), l1, o2.float(), l2)
    # reference: full attention over concat(ka,kb) per sequence
    Tk = sum(k1) + sum(k2)
    kcat = torch.empty(Tk, Hk, qk_d, dtype=torch.float16, device=DEV)
    vcat = torch.empty(Tk, Hk, v_d, dtype=torch.float16, device=DEV)
    cuc = [0]
    for b in range(len(qN)):
        s1, e1 = int(cuk1[b]), int(cuk1[b + 1])
        s2, e2 = int(cuk2[b]), int(cuk2[b + 1])
        off = cuc[-1]
        kcat[off:off + (e1 - s1)] = ka[s1:e1]
        vcat[off:off + (e1 - s1)] = va[s1:e1]
        kcat[off + (e1 - s1):off + (e1 - s1) + (e2 - s2)] = kb[s2:e2]
        vcat[off + (e1 - s1):off + (e1 - s1) + (e2 - s2)] = vb[s2:e2]
        cuc.append(off + (e1 - s1) + (e2 - s2))
    cuc = torch.tensor(cuc, dtype=torch.int32, device=DEV)
    ro, rl = ref_varlen(q1, kcat, vcat, cuq, cuc, scale, False)
    co, cl = cos(om, ro), cos(lm, rl)
    print(f"C merged 2 chunks (LSE merge): out cos={co:.6f}  lse cos={cl:.6f}")
    fails.append(("C " + name, co < 0.999 or cl < 0.999))


def main():
    cap = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    fails = []
    # GLM-4.7-Flash: symmetric 256 (no padding) — the immediate target.
    check("GLM-4.7-Flash", 256, 256, 256 ** -0.5, fails)
    # DeepSeek-like asymmetric: pad qk 192->256, v 128->256 (generalization).
    check("DeepSeek-like", 192, 128, 192 ** -0.5, fails)

    print("\n========================= SUMMARY =========================")
    bad = [n for n, f in fails if f]
    for n, f in fails:
        print(f"  {'FAIL' if f else 'PASS'}  {n}")
    if bad:
        print(f"\nRESULT: FAIL ({len(bad)} case(s): {bad})")
        sys.exit(1)
    print("\nRESULT: PASS — ai-bond dense varlen matches fp32 reference for MLA shapes "
          "(out + LSE). MLA-prefill interpose is numerically sound.")


if __name__ == "__main__":
    main()
