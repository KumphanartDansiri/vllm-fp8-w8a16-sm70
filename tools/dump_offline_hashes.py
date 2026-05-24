"""
dump_offline_hashes.py
──────────────────────
Print SHA-256 hashes (+ shape/stride/contiguity + head/tail elements) of the
post-loading weight and scale shards an offline emulator produces for the 5
target layers × 4 ranks the runtime instrumentation in serve_fp8_v100.py
also hashes.

Diff this script's output against the runtime log:
  - hash match  → vLLM's loader writes the same bytes as the emulator;
                  bug is downstream of load (activations, all-reduce, ...)
  - hash differ → vLLM loads different bytes despite the loader trace
                  saying they should match; bug is in load/PWAL path

Run:
    GPUS=any ./run_docker.sh dev-test dump_offline_hashes.py
or run on the host CPU directly (no GPU needed):
    python3 dump_offline_hashes.py
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "safetensors"]
    )
    from safetensors import safe_open

import torch


MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/mnt/models/Qwen3.6-27B-FP8"))
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
BLOCK_N = 128

# Per-config dims for Qwen3-Next 27B linear_attn.
KEY_DIM = 2048      # = linear_num_key_heads (16) * linear_key_head_dim (128)
VALUE_DIM = 6144    # = linear_num_value_heads (48) * linear_value_head_dim (128)


def load_index():
    return json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]


def fetch(wmap, name):
    with safe_open(MODEL_DIR / wmap[name], framework="pt") as f:
        return f.get_tensor(name)


def shard_col(t, tp_size, tp_rank):
    n = t.shape[0]
    assert n % tp_size == 0, f"{n} not divisible by tp_size={tp_size}"
    per = n // tp_size
    return t[tp_rank * per:(tp_rank + 1) * per, :].contiguous()


def shard_row(t, tp_size, tp_rank):
    k = t.shape[1]
    assert k % tp_size == 0
    per = k // tp_size
    return t[:, tp_rank * per:(tp_rank + 1) * per].contiguous()


def fuse(parts):
    return torch.cat(parts, dim=0).contiguous()


def post_pwal_scale(scale_bf16):
    """Mirror what _our_process_weights_after_loading does to the scale:
    cast BF16 -> FP16 (then .contiguous()). Weight stays raw FP8 bytes."""
    return scale_bf16.to(torch.float16).contiguous()


def hash_dump(prefix, rank, weight, scale, block_size=(128, 128)):
    w_bytes = weight.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    s_bytes = scale.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    w_hash = hashlib.sha256(w_bytes).hexdigest()
    s_hash = hashlib.sha256(s_bytes).hexdigest()
    w_head = list(weight.contiguous().view(torch.uint8).reshape(-1)[:4].cpu().tolist())
    w_tail = list(weight.contiguous().view(torch.uint8).reshape(-1)[-4:].cpu().tolist())
    s_head = scale.contiguous().reshape(-1)[:4].cpu().tolist()
    s_tail = scale.contiguous().reshape(-1)[-4:].cpu().tolist()
    print(
        f"[OFFLINE-HASH rank={rank}] {prefix}\n"
        f"  weight  shape={tuple(weight.shape)} dtype={weight.dtype} "
        f"stride={weight.stride()} contig={weight.is_contiguous()} "
        f"sha256={w_hash[:32]} head={w_head} tail={w_tail}\n"
        f"  scale   shape={tuple(scale.shape)} dtype={scale.dtype} "
        f"stride={scale.stride()} contig={scale.is_contiguous()} "
        f"sha256={s_hash[:32]} head={s_head} tail={s_tail}\n"
        f"  weight_block_size={list(block_size)}",
        flush=True,
    )


def main():
    wmap = load_index()
    print(f"Model: {MODEL_DIR}  TP_SIZE={TP_SIZE}")
    print(f"Hashing the offline-emulator shards for the same 5 layers the runtime "
          f"instrumentation hashes.\n")

    # ---------- self_attn (layer 3) ----------
    SA = "model.language_model.layers.3.self_attn"
    q_w = fetch(wmap, f"{SA}.q_proj.weight"); q_s = fetch(wmap, f"{SA}.q_proj.weight_scale_inv")
    k_w = fetch(wmap, f"{SA}.k_proj.weight"); k_s = fetch(wmap, f"{SA}.k_proj.weight_scale_inv")
    v_w = fetch(wmap, f"{SA}.v_proj.weight"); v_s = fetch(wmap, f"{SA}.v_proj.weight_scale_inv")
    o_w = fetch(wmap, f"{SA}.o_proj.weight"); o_s = fetch(wmap, f"{SA}.o_proj.weight_scale_inv")

    # ---------- linear_attn (layer 0) ----------
    LA = "model.language_model.layers.0.linear_attn"
    qkv_w = fetch(wmap, f"{LA}.in_proj_qkv.weight"); qkv_s = fetch(wmap, f"{LA}.in_proj_qkv.weight_scale_inv")
    z_w   = fetch(wmap, f"{LA}.in_proj_z.weight");   z_s   = fetch(wmap, f"{LA}.in_proj_z.weight_scale_inv")
    out_w = fetch(wmap, f"{LA}.out_proj.weight");    out_s = fetch(wmap, f"{LA}.out_proj.weight_scale_inv")

    # ---------- mlp (layer 0) ----------
    M = "model.language_model.layers.0.mlp"
    dn_w = fetch(wmap, f"{M}.down_proj.weight"); dn_s = fetch(wmap, f"{M}.down_proj.weight_scale_inv")

    # Pre-split linear_attn.in_proj_qkv into its 3 partitions (key, key, value)
    qkvz_p1_w, qkvz_p1_s = qkv_w[0:KEY_DIM, :], qkv_s[0:KEY_DIM // BLOCK_N, :]
    qkvz_p2_w, qkvz_p2_s = qkv_w[KEY_DIM:2*KEY_DIM, :], qkv_s[KEY_DIM // BLOCK_N:2*KEY_DIM // BLOCK_N, :]
    qkvz_p3_w, qkvz_p3_s = (qkv_w[2*KEY_DIM:2*KEY_DIM + VALUE_DIM, :],
                            qkv_s[2*KEY_DIM // BLOCK_N:(2*KEY_DIM + VALUE_DIM) // BLOCK_N, :])

    for rank in range(TP_SIZE):
        print(f"\n══════════════ rank={rank} ══════════════")

        # self_attn.qkv_proj (3-way merged column-parallel from q/k/v)
        sa_qkv_w = fuse([shard_col(q_w, TP_SIZE, rank),
                         shard_col(k_w, TP_SIZE, rank),
                         shard_col(v_w, TP_SIZE, rank)])
        sa_qkv_s = fuse([shard_col(q_s, TP_SIZE, rank),
                         shard_col(k_s, TP_SIZE, rank),
                         shard_col(v_s, TP_SIZE, rank)])
        hash_dump("layers.3.self_attn.qkv_proj", rank, sa_qkv_w, post_pwal_scale(sa_qkv_s))

        # self_attn.o_proj (row-parallel on K)
        hash_dump("layers.3.self_attn.o_proj", rank,
                  shard_row(o_w, TP_SIZE, rank),
                  post_pwal_scale(shard_row(o_s, TP_SIZE, rank)))

        # linear_attn.in_proj_qkvz (4-way merged column-parallel:
        # partitions 0,1,2 from in_proj_qkv; partition 3 from in_proj_z)
        la_qkvz_w = fuse([shard_col(qkvz_p1_w, TP_SIZE, rank),
                          shard_col(qkvz_p2_w, TP_SIZE, rank),
                          shard_col(qkvz_p3_w, TP_SIZE, rank),
                          shard_col(z_w, TP_SIZE, rank)])
        la_qkvz_s = fuse([shard_col(qkvz_p1_s, TP_SIZE, rank),
                          shard_col(qkvz_p2_s, TP_SIZE, rank),
                          shard_col(qkvz_p3_s, TP_SIZE, rank),
                          shard_col(z_s, TP_SIZE, rank)])
        hash_dump("layers.0.linear_attn.in_proj_qkvz", rank, la_qkvz_w, post_pwal_scale(la_qkvz_s))

        # linear_attn.out_proj (row-parallel on K)
        hash_dump("layers.0.linear_attn.out_proj", rank,
                  shard_row(out_w, TP_SIZE, rank),
                  post_pwal_scale(shard_row(out_s, TP_SIZE, rank)))

        # mlp.down_proj (row-parallel on K)
        hash_dump("layers.0.mlp.down_proj", rank,
                  shard_row(dn_w, TP_SIZE, rank),
                  post_pwal_scale(shard_row(dn_s, TP_SIZE, rank)))


if __name__ == "__main__":
    main()
