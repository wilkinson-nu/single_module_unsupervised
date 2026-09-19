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
