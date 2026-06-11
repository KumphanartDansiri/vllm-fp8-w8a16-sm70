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


def run_case(L, interleaved):
    torch.manual_seed(L)
    nb = (L + BLOCK - 1) // BLOCK
    if interleaved:
        kv = torch.randn(nb, 2, BLOCK, H_K, D, device=DEV, dtype=DT)
        k_cache, v_cache = kv.unbind(1)           # strided views, like vLLM
    else:
        k_cache = torch.randn(nb, BLOCK, H_K, D, device=DEV, dtype=DT)
        v_cache = torch.randn(nb, BLOCK, H_K, D, device=DEV, dtype=DT)

    perm = torch.randperm(nb)                     # shuffled physical pages
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

    # chunked exact fp32 reference (per q-row-chunk, all heads)
    qf = q.float()
    ref = torch.empty(L, H_Q, D, device=DEV, dtype=torch.float32)
    kT = k_log[:, 0, :].T.contiguous()            # [D, L] (H_K=1)
    vv = v_log[:, 0, :].contiguous()              # [L, D]
    rows = torch.arange(L, device=DEV)
    for s in range(0, L, CHUNK):
        e = min(s + CHUNK, L)
        for h in range(H_Q):
            sc = (qf[s:e, h, :] @ kT) * SCALE     # [chunk, L]
            mask = rows.unsqueeze(0) > rows[s:e].unsqueeze(1)
            sc.masked_fill_(mask, float("-inf"))
            ref[s:e, h, :] = torch.softmax(sc, dim=-1) @ vv
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


def main():
    bad = 0
    for L in (512, 2048):
        run_qkv_split_case(L)
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
