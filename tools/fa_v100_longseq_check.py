#!/usr/bin/env python3
# ======================================================================================
# Long-sequence correctness check for ai-bond varlen_fwd — closes the validation hole
# found in the e2e A/B (TTFT 19.6s = predicted speed, but garbage output @24k while
# the 512-token smoke + test.py passed).
#
# Sweeps seqlens, paged block=256, shuffled block table, GLM-Air per-rank shape
# (Hq=12 Hk=1 D=128), and ALSO mimics vLLM's interleaved KV cache:
#   kv = (num_blocks, 2, block, H, D); k = kv[:,0], v = kv[:,1]  (strided unbind view)
# Reference: chunked fp32 exact attention (no SDPA dependency).
# Reports cos/max_abs per length -> first failing length = the signature.
# ======================================================================================
import sys
import torch

try:
    import flash_attn_v100_cuda
except Exception as e:  # pragma: no cover
    print(f"[LONGSEQ] cannot import flash_attn_v100_cuda: {e}")
    sys.exit(2)

DEV = "cuda"
DT = torch.float16
BLOCK = 256
D = 128
H_Q, H_K = 12, 1
LENS = [512, 1024, 2048, 4096, 8192, 16384, 24064]
SCALE = D ** -0.5
CHUNK = 1024


SHUFFLE = True


def run_case(L, interleaved):
    torch.manual_seed(L)
    nb = (L + BLOCK - 1) // BLOCK
    if interleaved:
        kv = torch.randn(nb, 2, BLOCK, H_K, D, device=DEV, dtype=DT)
        k_cache, v_cache = kv.unbind(1)           # strided views, like vLLM
    else:
        k_cache = torch.randn(nb, BLOCK, H_K, D, device=DEV, dtype=DT)
        v_cache = torch.randn(nb, BLOCK, H_K, D, device=DEV, dtype=DT)

    perm = torch.randperm(nb) if SHUFFLE else torch.arange(nb)
    block_table = torch.empty(1, nb, dtype=torch.int32, device=DEV)
    block_table[0] = perm.to(torch.int32).to(DEV)

    # logical K/V (gather in logical order) for the reference
    k_log = k_cache[perm.to(DEV)].reshape(-1, H_K, D)[:L].float()
    v_log = v_cache[perm.to(DEV)].reshape(-1, H_K, D)[:L].float()

    q = torch.randn(L, H_Q, D, device=DEV, dtype=DT)
    out = torch.empty_like(q)
    cu_q = torch.tensor([0, L], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, L], dtype=torch.int32, device=DEV)
    seqused = torch.tensor([L], dtype=torch.int32, device=DEV)

    flash_attn_v100_cuda.varlen_fwd(
        q, k_cache, v_cache, out, cu_q, cu_k, seqused, None,
        block_table, None, L, L, 0.0, SCALE,
        False, True, -1, -1, 0.0, False, None, 0)

    # chunked exact fp32 reference — proper GQA head mapping (h -> h // group)
    qf = q.float()
    ref = torch.empty(L, H_Q, D, device=DEV, dtype=torch.float32)
    g = H_Q // H_K
    kTs = [k_log[:, j, :].T.contiguous() for j in range(H_K)]   # [D, L] each
    vvs = [v_log[:, j, :].contiguous() for j in range(H_K)]     # [L, D] each
    rows = torch.arange(L, device=DEV)
    for s in range(0, L, CHUNK):
        e = min(s + CHUNK, L)
        for h in range(H_Q):
            j = h // g
            sc = (qf[s:e, h, :] @ kTs[j]) * SCALE     # [chunk, L]
            mask = rows.unsqueeze(0) > rows[s:e].unsqueeze(1)
            sc.masked_fill_(mask, float("-inf"))
            ref[s:e, h, :] = torch.softmax(sc, dim=-1) @ vvs[j]
        del sc, mask
    err = (out.float() - ref).abs()
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0).item()
    # location of worst error (row) — diagnostic for the failure signature
    worst_row = int(err.max(dim=2).values.max(dim=1).values.argmax())
    return cos, err.max().item(), worst_row


def run_qkv_split_case(L):
    """q as a .split() view of a fused QKV row (row stride 1792 != Hq*D=1536),
    mimicking GLM-Air per-rank qkv_proj output. Triton handles via explicit
    strides; ai-bond's low-level kernel assumes dense H_Q*D rows."""
    torch.manual_seed(L)
    qkv = torch.randn(L, H_Q * D + 2 * H_K * D, device=DEV, dtype=DT)  # 1792
    q = qkv[:, : H_Q * D].view(L, H_Q, D)        # strides (1792, 128, 1)
    q_dense = q.contiguous()
    nb = (L + BLOCK - 1) // BLOCK
    k_cache = torch.randn(nb, BLOCK, H_K, D, device=DEV, dtype=DT)
    v_cache = torch.randn(nb, BLOCK, H_K, D, device=DEV, dtype=DT)
    block_table = torch.arange(nb, dtype=torch.int32, device=DEV).view(1, nb)
    cu_q = torch.tensor([0, L], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, L], dtype=torch.int32, device=DEV)
    seqused = torch.tensor([L], dtype=torch.int32, device=DEV)

    def call(qt):
        out = torch.empty(L, H_Q, D, device=DEV, dtype=DT)
        flash_attn_v100_cuda.varlen_fwd(
            qt, k_cache, v_cache, out, cu_q, cu_k, seqused, None,
            block_table, None, L, L, 0.0, SCALE,
            False, True, -1, -1, 0.0, False, None, 0)
        return out

    out_strided, out_dense = call(q), call(q_dense)
    err = (out_strided.float() - out_dense.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out_strided.float().flatten(), out_dense.float().flatten(), dim=0).item()
    native = cos > 0.9999 and err < 1e-3
    # INFORMATIONAL, not a gate: the raw kernel is KNOWN to assume dense q rows
    # (the vLLM adapter densifies q — fa_v100_prefill.py). This line flips to
    # SUPPORTED if ai-bond upstream ever adds explicit q strides.
    print(f"[LONGSEQ] qkv-split-q    L={L:6d}  cos={cos:.6f}  max_abs={err:9.4f}  "
          f"q.stride(0)={q.stride(0)} vs dense {q_dense.stride(0)}  "
          f"kernel-strided-q: {'SUPPORTED' if native else 'NOT-SUPPORTED (known; adapter densifies)'}",
          flush=True)


def _ref_prefix(q, k_log, v_log, sk, q_len):
    """fp32 reference for bottom-right causal: q row i = abs pos (sk - q_len + i),
    attends keys 0..abs_pos. k_log/v_log: [sk, H_K, D] logical order."""
    g = H_Q // H_K
    off = sk - q_len
    qf = q.float()
    ref = torch.empty(q_len, H_Q, D, device=DEV, dtype=torch.float32)
    keys = torch.arange(sk, device=DEV)
    for s in range(0, q_len, CHUNK):
        e = min(s + CHUNK, q_len)
        rows_abs = torch.arange(off + s, off + e, device=DEV)
        for h in range(H_Q):
            hk = h // g
            sc = (qf[s:e, h, :] @ k_log[:, hk, :].float().T) * SCALE
            sc.masked_fill_(keys.unsqueeze(0) > rows_abs.unsqueeze(1), float("-inf"))
            ref[s:e, h, :] = torch.softmax(sc, dim=-1) @ v_log[:, hk, :].float()
    return ref


def _paged_seq(L, seed):
    torch.manual_seed(seed)
    nb = (L + BLOCK - 1) // BLOCK
    k_cache = torch.full((nb, BLOCK, H_K, D), 100.0, device=DEV, dtype=DT)
    v_cache = torch.full((nb, BLOCK, H_K, D), 100.0, device=DEV, dtype=DT)
    k_log = torch.randn(L, H_K, D, device=DEV, dtype=DT)
    v_log = torch.randn(L, H_K, D, device=DEV, dtype=DT)
    k_cache.view(-1, H_K, D)[:L] = k_log
    v_cache.view(-1, H_K, D)[:L] = v_log
    return k_cache, v_cache, k_log, v_log, nb


def run_prefix_case(q_len, sk):
    """Sq < Sk: q covers only the LAST q_len rows of a seq with seqused_k=sk.
    This is the chunked-prefill / decode-row regime (audit: previously
    unvalidated; required for multi-user since mixed batches contain such rows)."""
    k_cache, v_cache, k_log, v_log, nb = _paged_seq(sk, sk + q_len)
    q = torch.randn(q_len, H_Q, D, device=DEV, dtype=DT)
    out = torch.empty_like(q)
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, sk], dtype=torch.int32, device=DEV)
    seqused = torch.tensor([sk], dtype=torch.int32, device=DEV)
    block_table = torch.arange(nb, dtype=torch.int32, device=DEV).view(1, nb)
    flash_attn_v100_cuda.varlen_fwd(
        q, k_cache, v_cache, out, cu_q, cu_k, seqused, None,
        block_table, None, q_len, sk, 0.0, SCALE,
        False, True, -1, -1, 0.0, False, None, 0)
    ref = _ref_prefix(q, k_log, v_log, sk, q_len)
    err = (out.float() - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0).item()
    ok = cos > 0.999 and err < 0.1
    print(f"[LONGSEQ] prefix Sq<Sk   q={q_len:6d} sk={sk:6d}  cos={cos:.6f}  "
          f"max_abs={err:9.4f}  {'PASS' if ok else '*** FAIL ***'}", flush=True)
    return ok


def run_mixed_batch_case():
    """One varlen call, B=2: seq0 decode-like (q=1, sk=8192) + seq1 fresh prefill
    (q=512, sk=512) — the multi-user mixed batch the adapter routes when
    max_seqlen_q>1."""
    sk0, q0, sk1, q1 = 8192, 1, 512, 512
    k0, v0, k0_log, v0_log, nb0 = _paged_seq(sk0, 7)
    # one shared pool: stack seq1's blocks after seq0's
    torch.manual_seed(8)
    nb1 = (sk1 + BLOCK - 1) // BLOCK
    k1_log = torch.randn(sk1, H_K, D, device=DEV, dtype=DT)
    v1_log = torch.randn(sk1, H_K, D, device=DEV, dtype=DT)
    k_cache = torch.cat([k0, torch.full((nb1, BLOCK, H_K, D), 100.0, device=DEV, dtype=DT)])
    v_cache = torch.cat([v0, torch.full((nb1, BLOCK, H_K, D), 100.0, device=DEV, dtype=DT)])
    k_cache[nb0:].view(-1, H_K, D)[:sk1] = k1_log
    v_cache[nb0:].view(-1, H_K, D)[:sk1] = v1_log
    max_nb = max(nb0, nb1)
    block_table = torch.zeros(2, max_nb, dtype=torch.int32, device=DEV)
    block_table[0, :nb0] = torch.arange(nb0, dtype=torch.int32, device=DEV)
    block_table[1, :nb1] = torch.arange(nb0, nb0 + nb1, dtype=torch.int32, device=DEV)

    q = torch.randn(q0 + q1, H_Q, D, device=DEV, dtype=DT)
    out = torch.empty_like(q)
    cu_q = torch.tensor([0, q0, q0 + q1], dtype=torch.int32, device=DEV)
    max_sk = max(sk0, sk1)
    cu_k = torch.arange(3, dtype=torch.int32, device=DEV) * max_sk  # adapter synth
    seqused = torch.tensor([sk0, sk1], dtype=torch.int32, device=DEV)
    flash_attn_v100_cuda.varlen_fwd(
        q, k_cache, v_cache, out, cu_q, cu_k, seqused, None,
        block_table, None, max(q0, q1), max_sk, 0.0, SCALE,
        False, True, -1, -1, 0.0, False, None, 0)
    ref0 = _ref_prefix(q[:q0], k0_log, v0_log, sk0, q0)
    ref1 = _ref_prefix(q[q0:], k1_log, v1_log, sk1, q1)
    ref = torch.cat([ref0, ref1])
    err = (out.float() - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0).item()
    ok = cos > 0.999 and err < 0.1
    print(f"[LONGSEQ] mixed-batch    (q=1,sk=8192)+(q=512,sk=512)  cos={cos:.6f}  "
          f"max_abs={err:9.4f}  {'PASS' if ok else '*** FAIL ***'}", flush=True)
    return ok


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128, help="head_dim (64/128/256)")
    ap.add_argument("--hq", type=int, default=12, help="per-rank query heads")
    ap.add_argument("--hk", type=int, default=1, help="per-rank kv heads")
    ap.add_argument("--quick", action="store_true", help="short lengths only")
    ap.add_argument("--noshuffle", action="store_true", help="identity block table")
    args = ap.parse_args()
    global D, H_Q, H_K, LENS, SHUFFLE
    D, H_Q, H_K = args.d, args.hq, args.hk
    if args.quick:
        LENS = [512, 2048]
    if args.noshuffle:
        SHUFFLE = False
    print(f"[LONGSEQ] === shape: D={D} H_Q={H_Q} H_K={H_K} block={BLOCK} ===",
          flush=True)
    bad = 0
    for L in (512, 2048):
        run_qkv_split_case(L)
    for q_len, sk in ((1, 8192), (64, 8192), (4096, 24064)):
        bad += (not run_prefix_case(q_len, sk))
    bad += (not run_mixed_batch_case())
    for inter in (False, True):
        tag = "interleaved-kv" if inter else "separate-kv"
        for L in LENS:
            try:
                cos, mx, wrow = run_case(L, inter)
                ok = cos > 0.999 and mx < 0.1
                print(f"[LONGSEQ] {tag:14s} L={L:6d}  cos={cos:.6f}  "
                      f"max_abs={mx:9.4f}  worst_row={wrow:6d}  "
                      f"{'PASS' if ok else '*** FAIL ***'}", flush=True)
                bad += (not ok)
            except torch.cuda.OutOfMemoryError:
                print(f"[LONGSEQ] {tag:14s} L={L:6d}  OOM (reference) — skipped",
                      flush=True)
            torch.cuda.empty_cache()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
