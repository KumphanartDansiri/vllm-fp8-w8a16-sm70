"""Tiny NCCL init probe.

Triggers a real ncclCommInitRank() + one all_reduce so NCCL_DEBUG=VERSION
prints the actual loaded library version banner. Use to confirm an NCCL
override took effect at runtime, independent of torch.cuda.nccl.version()
(which is torch's compile-time constant).

Launch:
    docker run --rm --gpus '"device=0,1"' -e NCCL_DEBUG=VERSION --shm-size=2g \\
        -v "$PWD":/work -w /work <image> \\
        torchrun --standalone --nproc-per-node=2 tools/nccl_probe.py
"""
import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    x = torch.ones(1, device=f"cuda:{rank}") * (rank + 1)
    dist.all_reduce(x)
    expected = sum(range(1, world + 1))
    print(f"rank {rank}/{world}: all_reduce result = {x.item()} "
          f"(expected {expected})", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
