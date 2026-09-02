import torch
import numpy as np
import random
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, DistributedSampler, Subset

## Import dataset utils
from core.data.datasets import paired_2d_dataset_ME, cat_ME_collate_fn
from core.data.datasets import single_2d_dataset_ME, solo_labelled_collate_fn

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


def build_knn_monitoring_data(
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


    
def get_training_dataloader(args, rank, world_size):

    ## Get the augmentation from the argument name
    aug_transform = get_transform(args.out_image_size,
                                  args.aug_type,
                                  args.aug_prob,
                                  getattr(args, "aug_val", None))
    
    ## Get the concrete dataset
    dataset = paired_2d_dataset_ME(args.data_dir, 
                                   nom_transform=DoNothing(),
                                   aug_transform=aug_transform,
                                   max_events=args.nevents)

    ## Make sure we have sufficient events
    assert len(dataset) >= args.nevents
    print0(f"Loaded {len(dataset)} training events")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset,
                            collate_fn=cat_ME_collate_fn,
                            batch_size=args.batch_size,
                            shuffle=False,  # Set to False, as DistributedSampler handles shuffling
                            num_workers=args.num_workers,
                            worker_init_fn=worker_init_fn,
                            drop_last=True,
                            persistent_workers=True,
                            prefetch_factor=2,
                            sampler=sampler)
    
    return dataset, dataloader


def get_monitoring_dataloaders(args, rank, world_size):

    ## Get a default "no augmentation" set to crop to the right size image
    nom_transform = get_transform(args.out_image_size,
                                  "no_aug")

    ## Rounding to make sure everything is cleanly divisible into the number of ranks
    nbank = (args.knn_nbank // world_size) * world_size
    nquery = (args.knn_nquery // world_size) * world_size
    
    ## Get the full dataset for monitoring...
    dataset_full = single_2d_dataset_ME(args.data_dir, 
                                        transform=nom_transform, 
                                        max_events=args.nevents + nbank + nquery)

    ## Make sure we have enough events available
    assert len(dataset_full) >= args.nevents + nbank + nquery
    
    ## ... and drop the events used for training
    bank_indices = list(range(args.nevents, args.nevents + nbank))
    query_indices = list(range(args.nevents + nbank, args.nevents + nbank + nquery))

    print0(f"Loaded {nbank} bank and {nquery} events for monitoring")

    ## Get the concrete dataset
    bank_dataset = Subset(dataset_full, bank_indices)
    query_dataset = Subset(dataset_full, query_indices)

    bank_sampler = DistributedSampler(bank_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    query_sampler = DistributedSampler(query_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    ## Slightly hacky way to manipulate the labels
    collate_fn = partial(solo_labelled_collate_fn,
                         label_clamp=LABEL_CLAMP,
                         derived_labels=DERIVED_LABELS)
    
    bank_dataloader = DataLoader(bank_dataset,
                                 collate_fn=collate_fn,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 worker_init_fn=worker_init_fn,
                                 drop_last=False,
                                 persistent_workers=True,
                                 prefetch_factor=2,
                                 sampler=bank_sampler)
    query_dataloader = DataLoader(query_dataset,
                                  collate_fn=collate_fn,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  worker_init_fn=worker_init_fn,
                                  drop_last=False,
                                  persistent_workers=True,
                                  prefetch_factor=2,
                                  sampler=query_sampler)
    return bank_dataloader, query_dataloader

