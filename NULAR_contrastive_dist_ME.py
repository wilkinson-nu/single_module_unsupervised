import numpy as np
import argparse
import sys
import MinkowskiEngine as ME
import torch
import time
import datetime
import math
import random
from collections import defaultdict
from functools import partial

## The parallelisation libraries
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.profiler import profile, record_function, ProfilerActivity

## Includes from my libraries for this project
from core.losses.ntxent import NTXentMerged, NTXentMergedMultiGPU
from core.losses.clustering import ClusteringLossMerged, ClusteringLossMergedMultiGPU, SharpenedClusterLoss
from datasets.nularbox.encoder import get_encoder
from core.models.projection_head import get_projhead
from core.models.clustering_head import get_clusthead
from core.analysis.metrics import argmax_consistency, uniformity, alignment, simclr_geometry_metrics
from core.training.logging import log_scalar, log_grad_norm, log_grad_rms, log_grad_over_wgt, log_weight_norm
from core.training.scheduling import get_opt_and_sched, cosine_scheduler, update_weight_decay

from core.training.system_monitoring_utils import log_memory, log_gpu, log_vmstat, print_affinity
import psutil, os
from threadpoolctl import threadpool_limits

## For logging
from torch.utils.tensorboard import SummaryWriter

## Import transformations
from core.data.augmentations_2d import DoNothing
from datasets.nularbox.augmentations_2d import get_transform

## Import dataset
from core.data.datasets import paired_2d_dataset_ME, cat_ME_collate_fn
from core.data.datasets import single_2d_dataset_ME, solo_labelled_collate_fn

## Supervised for kNN monitoring
from core.supervised import LABEL_CLAMP, DERIVED_LABELS, DEFAULT_CLASSIFIER_CONFIG
from core.supervised import ClassificationMetrics
from core.analysis.knn_monitoring import MONITOR_LABELS, extract_features, knn_votes

## For parallelising things
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

def print_model_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")
            total_params += param.numel()
    print("Total parameters =", total_params)

def worker_init_fn(worker_id):
    threadpool_limits(limits=1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

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
    if rank==0: print(f"Loaded {len(dataset)} training events")

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

    if rank==0: print(f"Loaded {nbank} bank and {nquery} events for monitoring")

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

    
def load_pretrained(encoder, heads, file_name):
    checkpoint = torch.load(file_name, map_location='cpu')
    encoder.module.load_state_dict(checkpoint['encoder_state_dict'])

    ## Load heads as requested
    for name, head in heads.items():
        key = f'{name}_head_state_dict'
        if key in checkpoint:
            head.module.load_state_dict(checkpoint[key])
    return

def load_checkpoint(encoder, heads, optimizer, scheduler, state_file_name):
    checkpoint = torch.load(state_file_name, map_location='cpu')
    encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'].cpu())
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

    ## If we have a scheduler, load it
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    ## Load heads as requested
    for	name, head in heads.items():
        key = f'{name}_head_state_dict'
        if key in checkpoint:
            head.module.load_state_dict(checkpoint[key])

    ## Load metrics
    metrics = defaultdict(list, checkpoint.get("metrics", {}))
    
    start_epoch = checkpoint['epoch'] + 1

    return start_epoch, metrics

def save_checkpoint(encoder, heads, optimizer, scheduler, state_file_name, iteration, metrics, args):

    state_dict = {
        'epoch': iteration,
        'encoder_state_dict': encoder.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
        'metrics': dict(metrics),
        'args':vars(args)
    }

    ## Save a scheduler if we have one
    if scheduler is not None:
        state_dict['scheduler_state_dict'] = scheduler.state_dict()
    
    ## Save heads as needed:
    for name, head in heads.items():
        state_dict[f'{name}_head_state_dict'] = head.module.state_dict()

    torch.save(state_dict, state_file_name)


## Wrapped training function
def run_training(rank, local_rank, world_size, args):

    ## For parallel work
    setup_dist(rank, local_rank, world_size)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    device = torch.device(f'cuda:{local_rank}')

    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    main_threads = max(1, min(8, cpus - args.num_workers))
    if rank==0: print("Torch has", main_threads, "threads/task")
    torch.set_num_threads(main_threads)
    torch.set_num_interop_threads(1)

    ## A debugging test
    print_affinity(rank, local_rank, world_size)
    dist.barrier()

    if bool(args.run_profiler) and rank==0:
        torch.cuda.set_sync_debug_mode("warn")

    torch.autograd.set_detect_anomaly(False)
    
    ## For timing
    tstart = time.time()
    
    ## Setup the encoder
    encoder = get_encoder(args)
    encoder = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(encoder)
    
    encoder_nchan_instance = encoder.get_nchan()
    encoder_nchan_cluster = encoder.get_nchan()
    encoder .to(device)
    encoder = DDP(encoder, device_ids=[local_rank])  ## Sort out parallel models (e.g., one is sent to each GPU)

    ## Dictionary of heads
    heads = {}
    
    ## Dictionary of loss functions
    loss_fns = {}

    ## Set up head and loss for projection space
    proj_head = get_projhead(encoder_nchan_instance, args)
    proj_head = nn.SyncBatchNorm.convert_sync_batchnorm(proj_head)
    proj_head .to(device)
    proj_head = DDP(proj_head, device_ids=[local_rank])
    heads["proj"] = proj_head
    loss_fns["proj"] = NTXentMergedMultiGPU(args.proj_temp)
        
    ## Optionally include the head and loss for the clustering space
    if args.clust_arch != "none":
        clust_head = get_clusthead(encoder_nchan_cluster, args)
        clust_head = nn.SyncBatchNorm.convert_sync_batchnorm(clust_head)
        clust_head .to(device)
        clust_head = DDP(clust_head, device_ids=[local_rank])
        heads["clust"] = clust_head
    
        if args.sharpened_cluster_loss == 0:
            loss_fns["clust"] = ClusteringLossMergedMultiGPU(args.clust_temp, args.entropy_scale)
        else:
            loss_fns["clust"] = SharpenedClusterLoss(args.clust_temp, 0.05, args.entropy_scale)
        
    ## Set up the distributed dataset
    train_dataset, train_loader = get_training_dataloader(args, rank, world_size)
    bank_loader, query_loader = get_monitoring_dataloaders(args, rank, world_size)
    nbatches   = len(train_loader)
    
    ## So we don't constantly ask args
    num_iterations = args.nepoch
    log_dir = args.log
    instance_scale = args.instance_scale
    clip_gradients = bool(args.clip_gradients)
    norm_encoder = bool(args.norm_encoder)
    weight_decay = args.weight_decay
    weight_decay_final = args.weight_decay_final
    
    writer = None
    if rank==0 and log_dir is not None:
        print("Training with", num_iterations, "iterations")
        writer = SummaryWriter(log_dir=log_dir)

    ## Sort out the optimizer (one for each GPU...)
    nstep_total = nbatches*args.nepoch
    optimizer, scheduler = get_opt_and_sched(args, encoder, heads, nstep_total, world_size)
    
    ## Set up metrics
    metrics = defaultdict(list)

    ## Setup kNN monitoring
    knn_metrics = ClassificationMetrics(
    {n: {'n_classes': c, 'weight': 1.0} for n, c in MONITOR_LABELS.items()},
    device=device)
    
    ## Load the checkpoint if one has been given
    start_iteration = 0
    if args.restart:
        if not args.state_file:
            if rank==0: print("Restart requested, but no state file provided, aborting")
            sys.exit()
        start_iteration, metrics = load_checkpoint(encoder, heads, optimizer, scheduler, args.state_file)
        global_iter = start_iteration*nbatches
        if rank==0: print("Restarting from iteration", start_iteration)

    ## Load the pretrained model if given
    if args.pretrained:
        if args.restart:
            if rank==0: print("Restart requested along with a pretraining file, abort!")
            sys.exit()
        load_pretrained(encoder, heads, args.pretrained)

    ## Stuff in a profiler
    if bool(args.run_profiler) and rank==0:
        
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.__enter__()
        
    ## Loop over the desired iterations
    global_iter = 0
    for iteration in range(start_iteration, start_iteration+num_iterations):

        # Ensure shuffling with the sampler each epoch
        train_loader.sampler.set_epoch(iteration)
        
        tot_loss_tensor = torch.tensor(0.0, device=device)  
        losses_tensor = {name: torch.tensor(0.0, device=device) for name in heads.keys()}       
        entropy_tensor = torch.tensor(0.0, device=device)
        
        ## For monitoring
        total_acc_tensor = torch.tensor(0.0, device=device)
        total_enc_align_tensor = torch.tensor(0.0, device=device)
        total_enc_unif_tensor = torch.tensor(0.0, device=device)
        total_proj_align_tensor = torch.tensor(0.0, device=device)
        total_proj_unif_tensor = torch.tensor(0.0, device=device)
        total_clust_align_tensor = torch.tensor(0.0, device=device)
        total_clust_unif_tensor = torch.tensor(0.0, device=device)

        ## Add more monitoring tools
        nbuffer = 5
        buffer_enc = []
        buffer_proj = []
        
        # Set train mode for the encoder and any heads
        encoder.train()
        for h in heads.values(): h.train()
        
        # Iterate over batches of images with the dataloader
        t0 = time.time()
        first_batch_latency = None
        for cat_bcoords, cat_bfeats, this_batch_size in train_loader:

            if first_batch_latency is None:
                first_batch_latency = time.time() - t0

            ## Update weight decay to allow for scheduling
            this_wd = update_weight_decay(optimizer,
			                  weight_decay,
                                          weight_decay_final,
                                          global_iter,
                                          nstep_total)
            
            ## Send to the device, then make the sparse tensors
            cat_bcoords = cat_bcoords.to(device, non_blocking=True)
            cat_bfeats  = cat_bfeats .to(device, non_blocking=True)
            cat_batch   = ME.SparseTensor(cat_bfeats, cat_bcoords, device=device)

            ## Now do the forward passes
            encoded_batch = encoder(cat_batch, this_batch_size)

            if global_iter % args.extra_log_rate == 0:
                with torch.no_grad():
                    hn = encoded_batch.detach().float().norm(dim=1)
                    gathered = [torch.zeros_like(hn) for _ in range(world_size)]
                    dist.all_gather(gathered, hn.contiguous())
                    allhn = torch.cat(gathered)
                    if rank == 0:
                        q = torch.quantile(allhn, torch.tensor([0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0], device=allhn.device))
                        for name, v in zip(['min', 'p1', 'p10', 'p50', 'p90', 'p99', 'max'], q.cpu()):
                            log_scalar(writer, metrics, f'hnorm/{name}', v.item(), global_iter)

            
            ## L2 norm the encoder
            if norm_encoder: encoded_batch = torch.nn.functional.normalize(encoded_batch, p=2, dim=1)

            ## Deal with the projection loss
            proj_batch = heads["proj"](encoded_batch)
            proj_loss = loss_fns["proj"](proj_batch)*instance_scale
                
            tot_loss = proj_loss
            losses_tensor["proj"] += proj_loss.detach()

            ## Add to metrics
            total_enc_align_tensor += alignment(encoded_batch)
            total_enc_unif_tensor += uniformity(encoded_batch)
            total_proj_align_tensor += alignment(proj_batch)
            total_proj_unif_tensor += uniformity(proj_batch)

            ## Get a few batches for calculating the running deff
            if len(buffer_enc) < nbuffer:
                with torch.no_grad():
                    buffer_enc .append(encoded_batch.detach().to("cpu", non_blocking=False))
                    buffer_proj.append(proj_batch.detach().to("cpu", non_blocking=False))
                    
            ## Optionally deal with clustering loss
            if "clust" in heads:
                clust_batch = heads["clust"](encoded_batch)
                clust_loss, clust_entropy = loss_fns["clust"](clust_batch)
                tot_loss += clust_loss + clust_entropy
                losses_tensor["clust"] += clust_loss.detach()
                entropy_tensor += clust_entropy.detach()
                total_acc_tensor += argmax_consistency(clust_batch)
                total_clust_align_tensor += alignment(clust_batch)
                total_clust_unif_tensor += uniformity(clust_batch)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            tot_loss .backward()

            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                for h in heads.values(): torch.nn.utils.clip_grad_norm_(h.parameters(), max_norm=1.0)
            
            ## Update optimizer and scheduler
            optimizer.step()
            if scheduler: scheduler.step()

            ## Increment global_iter
            global_iter += 1
            
            ## keep track of losses
            tot_loss_tensor += tot_loss.detach()

            ## Extra logging
            if global_iter % args.extra_log_rate == 0 and rank == 0:

                W_MIN = 1e-6   # ignore params still at (or near) zero init -- their
                               # relative step is 0/0 noise and dominates any ranking

                ## if rank == 0:
                ##     print(f"\nLARS diagnostics at step {global_iter}")
                ##     for gi, g in enumerate(optimizer.param_groups):
                ##         print(f"  group {gi} lr = {g['lr']:.4e}")

                for gi, stats in optimizer.last_stats.items():
                    if not stats:
                        continue

                    # One host transfer per group rather than three per parameter.
                    packed = torch.tensor(
                        [[s['relative_step'], s['trust_ratio'], s['weight_norm'], s['grad_norm']]
                         for s in stats], dtype=torch.float32)
                    rel, tru, wn, gn = packed.unbind(dim=1)
                    names = [s['name'] for s in stats]

                    keep = wn > W_MIN
                    n_drop = int((~keep).sum())
                    if keep.sum() == 0:
                        continue

                    rel_k, tru_k = rel[keep], tru[keep]

                    if rank == 0:
                        log_scalar(writer, metrics, f'lars/g{gi}_lr',
                                   optimizer.param_groups[gi]['lr'], global_iter)
                        log_scalar(writer, metrics, f'lars/g{gi}_rel_median',
                                   rel_k.median().item(), global_iter)
                        log_scalar(writer, metrics, f'lars/g{gi}_rel_min',
                                   rel_k.min().item(), global_iter)
                        log_scalar(writer, metrics, f'lars/g{gi}_rel_max',
                                   rel_k.max().item(), global_iter)
                        log_scalar(writer, metrics, f'lars/g{gi}_n_excluded',
                                   n_drop, global_iter)

                        # log10 so the ~3 decades of spread are visible; clamp kills -inf
                        writer.add_histogram(f'lars/g{gi}_log10_rel_step',
                                             rel_k.clamp(min=1e-12).log10(), global_iter)
                        writer.add_histogram(f'lars/g{gi}_log10_trust',
                                             tru_k.clamp(min=1e-12).log10(), global_iter)

                        ## med = rel_k.median().item()
                        ## print(f"Group {gi}: rel step min={rel_k.min():.2e} "
                        ##       f"median={med:.2e} max={rel_k.max():.2e}; "
                        ##       f"trust min={tru_k.min():.2e} max={tru_k.max():.2e} "
                        ##       f"({n_drop} params below ||w||={W_MIN:g} omitted)")
                        ## 
                        ## order = torch.argsort(rel_k, descending=True)[:5]
                        ## kept_idx = torch.nonzero(keep).squeeze(1)
                        ## print("Largest:")
                        ## for j in order:
                        ##     i = kept_idx[j].item()
                        ##     print(f"  {names[i]:50s} ||w||={wn[i]:.3e} ||g||={gn[i]:.3e} "
                        ##           f"trust={tru[i]:.3e} rel_step={rel[i]:.3e}")

        # Manage CUDA memory for ME
        torch.cuda.empty_cache()

        ## Although the gradients are handled correctly by GatherLayer, the losses are global
        ## Strictly speaking this step isn't necessary as each mini-batch gives the same loss value
        ## But I kept it in to avoid my own headaches...
        dist.all_reduce(tot_loss_tensor, op=dist.ReduceOp.SUM)
        for name in heads.keys(): dist.all_reduce(losses_tensor[name], op=dist.ReduceOp.SUM)
        dist.all_reduce(entropy_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_acc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_enc_align_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_enc_unif_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_proj_align_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_proj_unif_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_clust_align_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_clust_unif_tensor, op=dist.ReduceOp.SUM)        
        
        av_tot_loss = tot_loss_tensor.item() / (nbatches * world_size)
        av_losses = {
            name: losses_tensor[name].item() / (nbatches * world_size)
            for name in heads.keys()
        }
        av_entropy = entropy_tensor.item() / (nbatches * world_size)
        av_acc = total_acc_tensor.item() / (nbatches * world_size)
        av_enc_unif = total_enc_unif_tensor.item() / (nbatches * world_size)
        av_enc_align = total_enc_align_tensor.item() / (nbatches * world_size)
        av_proj_unif = total_proj_unif_tensor.item() / (nbatches * world_size)
        av_proj_align = total_proj_align_tensor.item() / (nbatches * world_size)
        av_clust_unif = total_clust_unif_tensor.item() / (nbatches * world_size)
        av_clust_align = total_clust_align_tensor.item() / (nbatches * world_size)

        ## Other geometry calculations
        enc_geom = simclr_geometry_metrics(buffer_enc, device, norm_encoder)
        proj_geom = simclr_geometry_metrics(buffer_proj, device, True)

        ## kNN monitoring
        knn_results = None
        if iteration % args.knn_every == 0:
            knn_tstart = time.time()
            bank_f, bank_l = extract_features(encoder, bank_loader,  device, MONITOR_LABELS)
            qry_f,  qry_l  = extract_features(encoder, query_loader, device, MONITOR_LABELS)
            
            knn_metrics.reset()
            votes = {n: knn_votes(qry_f, bank_f, bank_l[n], c,
                                  k=args.knn_k, T=args.knn_T)
                     for n, c in MONITOR_LABELS.items()}
            knn_metrics.update(votes, qry_l)
            ## NB: no .reduce() -- every rank holds the full gathered bank and query,
            ## so the result is already global. Reducing would multiply counts by world_size.
            knn_results = knn_metrics.compute()
            # print(f"kNN time taken: {(time.time()-knn_tstart):.2f}")
            
        ## Reporting, but only for rank 0
        if rank==0:
            metrics["iteration"].append(iteration)
            log_scalar(writer, metrics, 'loss/total', av_tot_loss, iteration)              
            log_scalar(writer, metrics, 'loss/proj', av_losses["proj"], iteration)

            ## Add metrics for debugging/training diagnostics
            log_scalar(writer, metrics, 'monitor/proj_alignment', av_proj_align, iteration)
            log_scalar(writer, metrics, 'monitor/proj_uniformity', av_proj_unif, iteration)
            log_scalar(writer, metrics, 'monitor/enc_alignment', av_enc_align, iteration)
            log_scalar(writer, metrics, 'monitor/enc_uniformity', av_enc_unif, iteration)
                
            ## Extensive logging for gradient debugging
            log_grad_norm(encoder.module, "encoder", writer, iteration)
            log_grad_rms(encoder.module, "encoder", writer, iteration)
            log_grad_over_wgt(encoder.module, "encoder", writer, iteration)
            log_weight_norm(encoder.module, "encoder", writer, iteration)
            
            log_grad_norm(heads["proj"].module, "proj", writer, iteration)
            log_grad_rms(heads["proj"].module, "proj", writer, iteration)
            log_grad_over_wgt(heads["proj"].module, "proj", writer, iteration)
            log_weight_norm(heads["proj"].module, "proj", writer, iteration)
            
            ## Eigenvalue debugging
            log_scalar(writer, metrics, "eigen/enc_deff", enc_geom["deff"], iteration)
            log_scalar(writer, metrics, "eigen/proj_deff", proj_geom["deff"], iteration)

            log_scalar(writer, metrics, "eigen/enc_rankme", enc_geom["rankme"], iteration)
            log_scalar(writer, metrics, "eigen/proj_rankme", proj_geom["rankme"], iteration)            
            
            log_scalar(writer, metrics, "eigen/enc_l1_ratio", enc_geom["l1_ratio"], iteration)
            log_scalar(writer, metrics, "eigen/proj_l1_ratio", proj_geom["l1_ratio"], iteration)
            
            for i,val in enumerate(enc_geom["eigvals"]):
                log_scalar(writer, metrics, f"eigen/enc_lambda{i}", val, iteration)
            for i,val in enumerate(proj_geom["eigvals"]):
                log_scalar(writer, metrics, f"eigen/proj_lambda{i}", val, iteration)

            log_scalar(writer, metrics, 'monitor/enc_pos', enc_geom["pos"], iteration)
            log_scalar(writer, metrics, 'monitor/enc_hard_neg', enc_geom["hard_neg"], iteration)
            log_scalar(writer, metrics, 'monitor/enc_mean_neg', enc_geom["mean_neg"], iteration)
            log_scalar(writer, metrics, 'monitor/enc_gap', enc_geom["gap"], iteration)
            log_scalar(writer, metrics, 'monitor/enc_gap_std', enc_geom["gap_std"], iteration)
            
            log_scalar(writer, metrics, 'monitor/proj_pos', proj_geom["pos"], iteration)
            log_scalar(writer, metrics, 'monitor/proj_hard_neg', proj_geom["hard_neg"], iteration)
            log_scalar(writer, metrics, 'monitor/proj_mean_neg', proj_geom["hard_neg"], iteration)            
            log_scalar(writer, metrics, 'monitor/proj_gap', proj_geom["gap"], iteration)
            log_scalar(writer, metrics, 'monitor/proj_gap_std', proj_geom["gap_std"], iteration)
                
            if "clust" in heads:
                log_scalar(writer, metrics, 'loss/clust', av_losses["clust"]+av_entropy, iteration)
                log_scalar(writer, metrics, 'loss/entropy', av_entropy, iteration)
                log_scalar(writer, metrics, 'loss/clust_only', av_losses["clust"], iteration)
                log_scalar(writer, metrics, 'monitor/acc', av_acc, iteration)
                log_scalar(writer, metrics, 'monitor/clust_alignment', av_clust_align, iteration)
                log_scalar(writer, metrics, 'monitor/clust_uniformity', av_clust_unif, iteration)

                ## Extensive logging for gradient debugging
                log_grad_norm(heads["clust"].module, "clust", writer, iteration)
                log_grad_rms(heads["clust"].module, "clust", writer, iteration)
                log_grad_over_wgt(heads["clust"].module, "clust", writer, iteration)
                log_weight_norm(heads["clust"].module, "clust", writer, iteration)

            if knn_results is not None:
                for name, m in knn_results.items():
                    log_scalar(writer, metrics, f'knn/{name}_accuracy',           m['accuracy'],           iteration)
                    log_scalar(writer, metrics, f'knn/{name}_mean_per_class_acc', m['mean_per_class_acc'], iteration)
                    log_scalar(writer, metrics, f'knn/{name}_mae',                m['mae'],                iteration)
                    log_scalar(writer, metrics, f'knn/{name}_recall_nonzero',     m['recall_nonzero'],     iteration)
                
            if scheduler: 
                log_scalar(writer, metrics, 'train/lr', scheduler.get_last_lr()[0], iteration)
            log_scalar(writer, metrics, 'train/weight_decay', this_wd, iteration)

            ## Build a string to report the outcome
            iter_string = f"Processed {iteration} / {start_iteration + num_iterations}; loss = {av_tot_loss:.4f}"
            
            if "clust" in heads:
                iter_string += f" ({av_losses['proj']:.4f} + {av_losses['clust']:.4f} + {av_entropy:.4f}); acc = {av_acc:.4f}"
            print(iter_string)
            print(f"Time taken: {(time.time()-tstart):.2f}")
            
        ## For checkpointing
        #if rank==0 and iteration%25 == 0 and iteration != 0:
        #    save_checkpoint(encoder, heads, optimizer, args.state_file+".check"+str(iteration), iteration, metrics, args)

        ## Add per GPU logging
        allocated_gb = torch.tensor(torch.cuda.memory_allocated() / 1e9, device=device)
        reserved_gb  = torch.tensor(torch.cuda.memory_reserved()  / 1e9, device=device)
        peak_alloc_gb = torch.tensor(torch.cuda.max_memory_allocated() / 1e9, device=device)
        torch.cuda.reset_peak_memory_stats()

        all_allocated  = [torch.zeros(1, device=device) for _ in range(world_size)]
        all_reserved   = [torch.zeros(1, device=device) for _ in range(world_size)]
        all_peak_alloc = [torch.zeros(1, device=device) for _ in range(world_size)]
        
        dist.all_gather(all_allocated,  allocated_gb.unsqueeze(0))
        dist.all_gather(all_reserved,   reserved_gb.unsqueeze(0))
        dist.all_gather(all_peak_alloc, peak_alloc_gb.unsqueeze(0))
            
        ## Enhanced logging
        if rank == 0:
            vm = psutil.virtual_memory()
            proc = psutil.Process(os.getpid())
            io = psutil.disk_io_counters()

            log_scalar(writer, metrics, 'syst_monitor/vm_used_gb', vm.used / 1e9, iteration)
            log_scalar(writer, metrics, 'syst_monitor/vm_avail_gb', vm.available / 1e9, iteration)
            log_scalar(writer, metrics, 'syst_monitor/vm_cached_gb', getattr(vm, "cached", 0) / 1e9, iteration)
            log_scalar(writer, metrics, 'syst_monitor/rss_gb', proc.memory_info().rss / 1e9, iteration)
            log_scalar(writer, metrics, 'syst_monitor/num_fds', proc.num_fds(), iteration)
            log_scalar(writer, metrics, 'syst_monitor/io_read', io.read_bytes, iteration)
            log_scalar(writer, metrics, 'syst_monitor/io_write', io.write_bytes, iteration)
            log_scalar(writer, metrics, 'syst_monitor/mem_pressure', vm.available / vm.total, iteration)
            log_scalar(writer, metrics, 'syst_monitor/first_batch_latency', first_batch_latency, iteration)

            for gpu_rank in range(world_size):
                log_scalar(writer, metrics, f'syst_monitor/gpu{gpu_rank}_allocated_gb',  all_allocated[gpu_rank].item(),  iteration)
                log_scalar(writer, metrics, f'syst_monitor/gpu{gpu_rank}_reserved_gb',   all_reserved[gpu_rank].item(),   iteration)
                log_scalar(writer, metrics, f'syst_monitor/gpu{gpu_rank}_peak_alloc_gb', all_peak_alloc[gpu_rank].item(), iteration)
                
    ## Final version of the model
    if rank==0:
        save_checkpoint(encoder, heads, optimizer, scheduler, args.state_file, iteration, metrics, args)
        if log_dir: writer.close()

    ## Report profiler if requested
    if bool(args.run_profiler) and rank == 0:
        prof.__exit__(None, None, None)
        
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=100))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    ## Clear things up
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()

    
## Do the business
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("SimCLR training module")

    # Basic job setup
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--nevents', type=int)
    parser.add_argument('--log', type=str, default=None)    
    parser.add_argument('--state_file', type=str)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--nepoch', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    
    ## Training dynamics
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--lars_trust_coeff', type=float, default=0.001)
    parser.add_argument('--lars_momentum', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--weight_decay_final', type=float, default=-1)
    parser.add_argument('--weight_decay_head', type=int, choices=[0,1], default=0)
    parser.add_argument('--clip_gradients', type=int, choices=[0,1], default=0)
    parser.add_argument('--norm_encoder', type=int, choices=[0,1], default=0)
    
    ## Image size and augmentations
    parser.add_argument('--out_image_size', type=int, default=256)
    parser.add_argument('--aug_type', type=str, default=None)
    parser.add_argument('--aug_prob', type=float, default=1)
    parser.add_argument('--aug_val', type=float)

    ## Encoder architecture choices
    parser.add_argument('--enc_act', type=str, default="relu")
    parser.add_argument('--enc_arch', type=str, default=None)
    parser.add_argument('--enc_arch_pool', type=str, default="avg")
    parser.add_argument('--enc_res_pool', type=int, choices=[0,1], default=0)
    parser.add_argument('--enc_stem_norm', type=int, choices=[0,1], default=0)
    parser.add_argument('--enc_init_stem_stride', type=int, default=2)
    parser.add_argument('--enc_final_stem_stride', type=int, default=2)
    parser.add_argument('--enc_stem_pool', type=str, default='none')
    parser.add_argument('--enc_stem_deep', type=int, choices=[0,1], default=1)
    parser.add_argument('--enc_layer1_norm', type=int, choices=[0,1], default=1)
    parser.add_argument('--enc_final_linear', type=int, default=-1)
    parser.add_argument('--enc_stem_channels', type=int, default=-1)

    ## (Optional) clustering head
    parser.add_argument('--clust_arch', type=str, default="none")
    parser.add_argument('--clust_temp', type=float, default=0.5)
    parser.add_argument('--nclusters', type=int, default=20)
    parser.add_argument('--entropy_scale', type=float, default=1.0)
    parser.add_argument('--softmax_temp', type=float, default=1.0)
    parser.add_argument('--instance_scale', type=float, default=1.0)

    ## A quick test option
    parser.add_argument('--sharpened_cluster_loss', type=int, choices=[0,1], default=0)
    
    ## Projection head
    parser.add_argument('--proj_arch', type=str, default="two")
    parser.add_argument('--proj_init_bn', type=int, choices=[0,1], default=0)
    parser.add_argument('--proj_final_bn', type=int, choices=[0,1], default=0)
    parser.add_argument('--proj_temp', type=float, default=0.5)
    parser.add_argument('--latent', type=int, default=128)
    parser.add_argument('--nhidden', type=int, default=512)

    ## kNN monitoring options
    parser.add_argument('--knn_nbank',    type=int, default=50000)
    parser.add_argument('--knn_nquery',   type=int, default=10000)
    parser.add_argument('--knn_every', type=int, default=1)
    parser.add_argument('--knn_k',     type=int, default=20)
    parser.add_argument('--knn_T',     type=float, default=0.1)
    
    ## Restart option
    parser.add_argument('--restart', action='store_true')

    ## Optional profiler
    parser.add_argument('--run_profiler', type=int, choices=[0,1], default=0)
    parser.add_argument('--extra_log_rate', type=int, default=10000000)
    # Parse arguments from command line
    args = parser.parse_args()

    ## Note global and local ranks to allow multi-node training
    rank       = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    
    ## Report arguments (but only rank 0)
    if rank==0:
        for arg in vars(args): print(arg, getattr(args, arg))

    ## Removed mp.spawn, now requires srun
    run_training(rank, local_rank, world_size, args)
