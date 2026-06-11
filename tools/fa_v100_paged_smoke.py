#!/usr/bin/env python3
# ======================================================================================
# Adapter-contract smoke test for ai-bond flash-attention-v100 paged varlen.
#
# Calls the LOW-LEVEL flash_attn_v100_cuda.varlen_fwd DIRECTLY — NOT the public python
# wrapper, which hardcodes seqused_k=None and out=None (flash_attn_interface.py:206-211,
# Codex review finding). The low-level call is the exact call the vLLM adapter will make.
#
# Validates the frozen 9-item adapter spec (docs/FA_V100_AUDIT.md):
#   - paged KV layout [num_blocks, block_size=256, H_K, D]
#   - non-contiguous (interleaved) block_table indirection
#   - seqused_k CLAMP: synthesized cu_seqlens_k diff (= max_seqlen_k) > seqused_k[i],
#     and KV tail beyond seqused is filled with GARBAGE (100.0) — if the kernel fails
#     to clamp, the garbage dominates softmax and the test FAILS loudly
#   - synthesized cu_seqlens_k[i] = i * max_seqlen_k (C1: length-valid, NOT zero dummy)
#   - out= preallocated, written IN-PLACE (data_ptr identity + content checked)
#   - LSE shape [H_Q, T_Q]
#   - GQA (H_Q % H_K == 0), fp16, causal
# Compares against an fp32 gather reference. Exit 0 = PASS.
# ======================================================================================
import sys
import torch

try:
    import flash_attn_v100_cuda
except Exception as e:  # pragma: no cover
    print(f"[SMOKE] cannot import flash_attn_v100_cuda: {e}")
    sys.exit(2)

DEV = "cuda"
DT = torch.float16
BLOCK = 256          # Route A block size (ai-bond requires % 256 == 0)
D = 128              # head_dim (GLM-Air / Qwen)
H_Q, H_K = 12, 2     # GQA group = 6
SEQS = [300, 512]    # 300 -> partial last block (44 valid + 212 garbage rows), 512 -> full
GARBAGE = 100.0      # fill beyond-seqused KV slots; over-read => softmax blowout => FAIL
SCALE = D ** -0.5


def build_paged():
    B = len(SEQS)
    blocks_per = [(L + BLOCK - 1) // BLOCK for L in SEQS]
    total_blocks = sum(blocks_per)
    # Interleaved physical assignment to exercise block_table indirection:
    # seq b, logical block j -> physical block (j*B + b)  => seq0=[0,2], seq1=[1,3]
    block_table = torch.zeros(B, max(blocks_per), dtype=torch.int32)
    for b, nb in enumerate(blocks_per):
        for j in range(nb):
            block_table[b, j] = j * B + b

    # GARBAGE-initialized paged KV: only valid token slots get real data.
    k_paged = torch.full((total_blocks, BLOCK, H_K, D), GARBAGE, device=DEV, dtype=DT)
    v_paged = torch.full((total_blocks, BLOCK, H_K, D), GARBAGE, device=DEV, dtype=DT)

    q_list, kref, vref = [], [], []
    for b, L in enumerate(SEQS):
        q_list.append(torch.randn(L, H_Q, D, device=DEV, dtype=DT))
        kb = torch.randn(L, H_K, D, device=DEV, dtype=DT)
        vb = torch.randn(L, H_K, D, device=DEV, dtype=DT)
        kref.append(kb); vref.append(vb)
        for t in range(L):
            pb = int(block_table[b, t // BLOCK]); off = t % BLOCK
            k_paged[pb, off] = kb[t]
            v_paged[pb, off] = vb[t]

    q = torch.cat(q_list, 0).contiguous()
    cu_q = torch.tensor([0] + list(torch.tensor(SEQS).cumsum(0).tolist()),
                        dtype=torch.int32, device=DEV)
    seqused_k = torch.tensor(SEQS, dtype=torch.int32, device=DEV)
    max_sk = max(SEQS)
    # C1: length-valid synthesized cu_seqlens_k. diff = max_sk (512) which for seq0
    # EXCEEDS seqused_k[0]=300 -> kernel must clamp via min(diff, seqused) or it reads
    # the garbage rows 300..511.
    cu_k = (torch.arange(len(SEQS) + 1, dtype=torch.int32, device=DEV) * max_sk)
    return (q, k_paged, v_paged, cu_q, cu_k, seqused_k,
            max(SEQS), max_sk, block_table.to(DEV), kref, vref)


def reference(q, cu_q, kref, vref):
    g = H_Q // H_K
    outs = []
    for b, L in enumerate(SEQS):
        qb = q[int(cu_q[b]):int(cu_q[b + 1])].float()       # [L,H_Q,D]
        kb = kref[b].float(); vb = vref[b].float()           # [L,H_K,D]
        ob = torch.empty_like(qb)
        idx = torch.arange(L, device=DEV)
        causal_mask = idx.unsqueeze(1) < idx.unsqueeze(0)    # k>q masked (Sq==Sk)
        for h in range(H_Q):
            hk = h // g
            s = (qb[:, h, :] @ kb[:, hk, :].T) * SCALE
            s = s.masked_fill(causal_mask, float("-inf"))
            p = torch.softmax(s, dim=-1)
            ob[:, h, :] = p @ vb[:, hk, :]
        outs.append(ob)
    return torch.cat(outs, 0)


def main():
    torch.manual_seed(0)
    (q, k_paged, v_paged, cu_q, cu_k, seqused_k,
     max_sq, max_sk, block_table, kref, vref) = build_paged()

    out = torch.empty_like(q)  # preallocated -> spec item 7 (in-place out=)

    # Low-level call: mirrors flash_attention_varlen_forward(...) C++ signature
    # (q, k, v, out, cu_q, cu_k, seqused_k, leftpad_k, block_table, alibi_slopes,
    #  max_sq, max_sk, p_dropout, scale, zero_tensors, causal, win_l, win_r,
    #  softcap, return_softmax, gen, num_splits)
    ret = flash_attn_v100_cuda.varlen_fwd(
        q, k_paged, v_paged, out,
        cu_q, cu_k,
        seqused_k, None,
        block_table, None,
        max_sq, max_sk,
        0.0, SCALE,
        False, True,
        -1, -1, 0.0,
        False, None, 0,
    )
    got, lse = ret[0], ret[1]

    inplace_ok = got.data_ptr() == out.data_ptr()
    lse_ok = tuple(lse.shape) == (H_Q, q.shape[0])

    ref = reference(q, cu_q, kref, vref).to(DT)
    err = (out.float() - ref.float()).abs()      # check OUR buffer, not just `got`
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0).item()
    max_abs = err.max().item()

    print(f"[SMOKE] block_size={BLOCK} GQA={H_Q}/{H_K} D={D} seqs={SEQS} "
          f"interleaved block_table, garbage-tail clamp test")
    print(f"[SMOKE] in-place out= : {'OK' if inplace_ok else 'FAIL (kernel allocated)'}")
    print(f"[SMOKE] LSE [H_Q,T_Q] : {'OK' if lse_ok else f'FAIL {tuple(lse.shape)}'}")
    print(f"[SMOKE] numerics      : cos={cos:.6f} max_abs={max_abs:.4f}")
    ok = inplace_ok and lse_ok and cos > 0.999 and max_abs < 0.05
    print(f"[SMOKE] {'PASS' if ok else 'FAIL'} (low-level varlen_fwd adapter contract)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
