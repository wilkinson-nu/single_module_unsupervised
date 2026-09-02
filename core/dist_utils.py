import torch
import torch.distributed as dist
import datetime
import builtins

def setup_dist(rank, local_rank, world_size):
    torch.cuda.set_device(local_rank)    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=30),
        device_id=torch.device(f"cuda:{local_rank}"),
    )

def print0(*args, **kwargs):
    """Print only from rank 0, or normally if distributed is not initialized."""
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_rank() == 0
    ):
        kwargs.setdefault("flush", True)
        builtins.print(*args, **kwargs)
