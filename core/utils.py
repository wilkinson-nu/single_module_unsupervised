## For very high-level functions which might be reused in many places
import torch.distributed as dist
import builtins

def print0(*args, **kwargs):
    """Print only from rank 0, or normally if distributed is not initialized."""
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_rank() == 0
    ):
        kwargs.setdefault("flush", True)
        builtins.print(*args, **kwargs)
