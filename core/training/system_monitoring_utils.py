import os
import psutil
import torch



def log_memory(epoch):
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())

    print("LOG_MEMORY")
    print({
        "epoch": epoch,
        "ram_used_gb": vm.used / 1e9,
        "ram_available_gb": vm.available / 1e9,
        "ram_percent": vm.percent,
        "proc_rss_gb": proc.memory_info().rss / 1e9,
    })

def log_gpu(epoch, rank):
    print("LOG_GPU")
    free, total = torch.cuda.mem_get_info(rank)
    print({
        "epoch": epoch,
        "gpu": rank,
        "free_gb": free / 1e9,
        "total_gb": total / 1e9,
        "allocated_gb": torch.cuda.memory_allocated(rank) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(rank) / 1e9,
    })

def log_vmstat(epoch):
    with open("/proc/vmstat") as f:
        data = f.read().splitlines()

    keys = ["pgfault", "pgmajfault", "pgscan_kswapd", "pgsteal_kswapd"]

    stats = {k: None for k in keys}
    for line in data:
        k, v = line.split()
        if k in stats:
            stats[k] = int(v)

    print("LOG VMSTAT")
    print({"epoch": epoch, **stats})

    
def print_affinity(rank, local_rank, world_size):
    import subprocess

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
