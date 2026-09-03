import torch
import numpy as np
import random
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, DistributedSampler, Subset

## Import dataset utils
from core.data.datasets import paired_2d_dataset_ME, cat_ME_collate_fn, single_2d_dataset_ME

## Basic utils
from core.utils import print0


def worker_init_fn(worker_id):
    threadpool_limits(limits=1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def make_distributed_dataloader(
    dataset,
    *,
    rank,
    world_size,
    batch_size,
    collate_fn,
    num_workers,
    shuffle,
    drop_last,
    seed=0,
    pin_memory=True,
):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "sampler": sampler,
        "shuffle": False,  # The sampler controls shuffling.
        "num_workers": num_workers,
        "worker_init_fn": worker_init_fn,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
    }

    # These options are invalid or irrelevant with zero workers.
    if num_workers > 0:
        kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })

    return DataLoader(**kwargs)

    
def build_paired_training_data(
    *,
    data_dir,
    nevents,
    transform,
    rank,
    world_size,
    batch_size,
    num_workers,
    seed,
):
    dataset = paired_2d_dataset_ME(
        data_dir,
        aug_transform=transform,
        max_events=nevents,
    )

    if len(dataset) < nevents:
        raise ValueError(
            f"Requested {nevents} training events, "
            f"but dataset contains only {len(dataset)}"
        )

    loader = make_distributed_dataloader(
        dataset,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        collate_fn=cat_ME_collate_fn,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )

    print0(f"Loaded {len(dataset)} training events")
    return dataset, loader


def build_monitoring_data(
    *,
    data_dir,
    train_events,
    nbank,
    nquery,
    transform,
    collate_fn,
    rank,
    world_size,
    batch_size,
    num_workers,
    seed,
):
    # Ensure DistributedSampler does not need to pad either subset.
    nbank = (nbank // world_size) * world_size
    nquery = (nquery // world_size) * world_size

    required_events = train_events + nbank + nquery

    full_dataset = single_2d_dataset_ME(
        data_dir,
        transform=transform,
        max_events=required_events,
    )

    if len(full_dataset) < required_events:
        raise ValueError(
            f"Monitoring requires {required_events} total events, "
            f"but dataset contains only {len(full_dataset)}"
        )

    bank_start = train_events
    query_start = bank_start + nbank

    bank_dataset = Subset(
        full_dataset,
        range(bank_start, query_start),
    )
    query_dataset = Subset(
        full_dataset,
        range(query_start, query_start + nquery),
    )

    common = {
        "rank": rank,
        "world_size": world_size,
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "shuffle": False,
        "drop_last": False,
        "seed": seed,
    }

    bank_loader = make_distributed_dataloader(
        bank_dataset,
        **common,
    )
    query_loader = make_distributed_dataloader(
        query_dataset,
        **common,
    )

    print0(f"Loaded {nbank} bank and {nquery} query events for monitoring")
    return bank_loader, query_loader


def build_supervised_dataloaders(
    *,
    data_dir,
    ntrain,
    nval,
    train_transform,
    val_transform,
    rank,
    world_size,
    batch_size,
    num_workers,
    collate_fn,
    seed,
):
    required_events = ntrain + nval

    train_full = single_2d_dataset_ME(
        data_dir,
        transform=train_transform,
        max_events=required_events,
    )
    val_full = single_2d_dataset_ME(
        data_dir,
        transform=val_transform,
        max_events=required_events,
    )

    if len(train_full) < required_events:
        raise ValueError(
            f"Requested {required_events} total events "
            f"({ntrain} train + {nval} validation), but only "
            f"{len(train_full)} are available"
        )

    train_dataset = Subset(
        train_full,
        range(0, ntrain),
    )
    val_dataset = Subset(
        val_full,
        range(ntrain, required_events),
    )

    train_loader = make_distributed_dataloader(
        train_dataset,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )

    val_loader = make_distributed_dataloader(
        val_dataset,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        seed=seed,
    )

    print0(
        f"Loaded {required_events} events: "
        f"{ntrain} training and {nval} validation"
    )

    return train_dataset, train_loader, val_dataset, val_loader
