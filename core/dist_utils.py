import torch
import torch.distributed as dist
import datetime
import numpy as np
import random
import os
import subprocess

def setup_distributed_runtime(
    rank,
    local_rank,
    world_size,
    *,
    seed,
    num_workers,
    print_cpu_affinity=False,
):
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=30),
        device_id=device,
    )

    ## Seeding on each rank
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

    ## DataLoader workers get one thread, so reserve the remaining CPUs for the main process
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    main_threads = max(1, min(8, cpus - num_workers))

    torch.set_num_threads(main_threads)
    torch.set_num_interop_threads(1)

    print0(
        f"Distributed setup: world_size={world_size}, "
        f"main_threads={main_threads}, "
        f"workers={num_workers}"
    )

    if print_cpu_affinity:
        print_affinity(rank, local_rank, world_size)

    return device


def print_affinity(rank, local_rank, world_size):

    # Cores this process is actually allowed to run on (post-binding).
    try:
        # sched_getaffinity is the ground truth for the current process.
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:
        allowed = None  # not available on all platforms

    # Compact the core list into ranges for readable output, e.g. [0-15].
    def compact(cores):
        if not cores:
            return "unknown"
        ranges, start, prev = [], cores[0], cores[0]
        for c in cores[1:]:
            if c == prev + 1:
                prev = c
            else:
                ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
                start = prev = c
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        return ",".join(ranges)

    gpu_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_id)
    node = os.environ.get("SLURMD_NODENAME", "?")

    msg = (
        f"[rank {rank:2d} | local {local_rank} | world {world_size}] "
        f"node={node} "
        f"gpu=cuda:{gpu_id} ({gpu_name}) "
        f"ncores={len(allowed) if allowed else '?'} "
        f"cores=[{compact(allowed)}]"
    )
    # flush so lines from all ranks aren't buffered/interleaved oddly.
    print(msg, flush=True)
