## This file trains a simple encoder with a contrastive loss. It should work with any number of GPUs distributed across nodes
import numpy as np
import argparse
from torch import optim
import sys
import MinkowskiEngine as ME
import torch
import time
import math
from collections import defaultdict

## The parallelisation libraries
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.profiler import profile, record_function, ProfilerActivity

## Includes from my libraries for this project
from core.losses.ntxent import NTXentMerged, NTXentMergedMultiGPU
from core.losses.clustering import ClusteringLossMerged, ClusteringLossMergedMultiGPU, SharpenedClusterLoss
# from core.models.encoder import get_encoder
from datasets.nularbox.encoder import get_encoder
from core.models.projection_head import get_projhead
from core.models.clustering_head import get_clusthead
from core.analysis.metrics import argmax_consistency, uniformity, alignment
from core.training.lars import LARS, LARS_LRScheduler
from core.losses.gather import GatherLayer

from core.training.system_monitoring_utils import log_memory, log_gpu, log_vmstat
import psutil, os

## For logging
from torch.utils.tensorboard import SummaryWriter

## Seeding
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

torch.set_num_threads(8)
torch.set_num_interop_threads(8)

## Import transformations
from core.data.augmentations_2d import DoNothing
from datasets.nularbox.augmentations_2d import get_transform

## Import dataset
from core.data.datasets import paired_2d_dataset_ME, cat_ME_collate_fn

## For parallelising things
def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def print_model_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")
            total_params += param.numel()
    print("Total parameters =", total_params)

def get_dataloader(rank, world_size, train_dataset, batch_size, num_workers=8):
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             collate_fn=cat_ME_collate_fn,
                                             batch_size=batch_size,
                                             shuffle=False,  # Set to False, as DistributedSampler handles shuffling
                                             num_workers=num_workers,
                                             drop_last=True,
                                             persistent_workers=False,
                                             prefetch_factor=2,
                                             sampler=sampler)
    return dataloader

## A simply logging utility
def log_scalar(writer, metrics, name, value, step):
    if writer is not None:
        writer.add_scalar(name, value, step)
    metrics[name].append(value)
    
def manage_cuda_memory(rank, gpu_threshold):
    """Check and clear GPU memory if it exceeds the threshold."""
    if torch.cuda.memory_allocated(rank) > gpu_threshold:
        torch.cuda.empty_cache()

def load_pretrained(encoder, heads, file_name):
    checkpoint = torch.load(file_name, map_location='cpu')
    encoder.module.load_state_dict(checkpoint['encoder_state_dict'])

    ## Load heads as requested
    for name, head in heads.items():
        key = f'{name}_head_state_dict'
        if key in checkpoint:
            head.module.load_state_dict(checkpoint[key])
    return

def load_checkpoint(encoder, heads, optimizer, state_file_name):
    checkpoint = torch.load(state_file_name, map_location='cpu')
    encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'].cpu())
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

    ## Load heads as requested
    for	name, head in heads.items():
        key = f'{name}_head_state_dict'
        if key in checkpoint:
            head.module.load_state_dict(checkpoint[key])

    ## Load metrics
    metrics = defaultdict(list, checkpoint.get("metrics", {}))
    
    start_epoch = checkpoint['epoch'] + 1

    return start_epoch, metrics

def save_checkpoint(encoder, heads, optimizer, state_file_name, iteration, metrics, args):

    state_dict = {
        'epoch': iteration,
        'encoder_state_dict': encoder.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
        'metrics': dict(metrics),
        'args':vars(args)
    }

    ## Save heads as needed:
    for name, head in heads.items():
        state_dict[f'{name}_head_state_dict'] = head.module.state_dict()

    torch.save(state_dict, state_file_name)


## Function to deal with all of the dataset handling
def get_dataset(args, rank=0):

    ## Get the augmentation from the argument name
    aug_transform = get_transform('256x256', args.aug_type, args.aug_prob)
    
    ## Get the concrete dataset
    data_dataset = paired_2d_dataset_ME(args.data_dir, \
                                        nom_transform=DoNothing(), \
                                        aug_transform=aug_transform, \
                                        max_events=args.nevents)
    if rank==0: print("Training with", args.nevents, "data events!")
    
    return data_dataset


def get_opt_and_sched(args, encoder, heads, total_steps):

    scheduler = None
    optimizer = None

    ## Sort out the optimizer (one for each GPU...)
    if args.optimizer == 'lars':
        param_groups = build_param_groups_LARS(encoder, heads, args.weight_decay)
        
        corr_lr = args.lr * (args.batch_size*args.world_size / 256)
        optimizer = LARS(
            param_groups,
            lr=corr_lr,
            momentum=args.lars_momentum,
            trust_coef=args.lars_trust_coeff,
        )

        warmup_steps = int(0.05 * total_steps)
        scheduler = LARS_LRScheduler(optimizer, warmup_steps, total_steps, lr_max=args.lr, lr_min=0.0)

    ## Default to adam
    else:
        param_groups = build_param_groups_ADAM(encoder, heads, args.weight_decay)

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
        )
        if args.scheduler == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr*500, total_steps=total_steps, cycle_momentum=False)
        if args.scheduler == "step":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=[150,300,450],
                                                       gamma=0.1,
                                                       last_epoch=-1,
                                                       verbose=False)
    return optimizer, scheduler

## Some diagnostic functions
@torch.no_grad()
def log_grad_norm(module, tag, writer, iteration):
    if writer is None: return
    total = 0.0
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        total += p.grad.pow(2).sum()

    grad_norm = total.sqrt().item()
    writer.add_scalar(f'grads/{tag}/sqrt_sum_grad_l2', grad_norm, iteration)
    return

@torch.no_grad()
def log_grad_rms(module, tag, writer, iteration):
    if writer is None: return

    total = 0.0
    count = 0
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        total += p.grad.pow(2).mean()
        count += 1

    mean_rms = (total / count).sqrt()
    writer.add_scalar(f'grads/{tag}/mean_grad_rms', mean_rms, iteration)
    return

@torch.no_grad()
def log_grad_over_wgt(module, tag, writer, iteration, eps=1e-12):
    if writer is None: return
    g2 = 0.0
    w2 = 0.0
    
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        g2 += p.grad.pow(2).sum()
        w2 += p.data.pow(2).sum()
        

    ratio = (g2.sqrt() / (w2.sqrt() + eps)).item()
    writer.add_scalar(f'grads/{tag}/sum_grad_over_wgt', ratio, iteration)
    return


@torch.no_grad()
def simclr_geometry_metrics(buffer, device):

    '''
    Each element in buffer is the concatenation of the two views in a batch
    Loop over the buffer and calculate values for each batch and then average
    '''

    dim = buffer[0].shape[1]
    cov = torch.zeros(dim, dim, device=device)
    n = 0
    
    ## Keep track of values
    pos_buffer = []
    neg_buffer = []
    
    ## loop over buffer
    for emb_cat_cpu in buffer:

        emb_cat = emb_cat_cpu.to(device, non_blocking=True)
        
        batch_size = emb_cat.shape[0]//2
        z_cat = emb_cat / (emb_cat.norm(dim=1, keepdim=True) + 1e-8)
        z_i, z_j = z_cat[:batch_size], z_cat[batch_size:]

        z_i_all = torch.cat(GatherLayer.apply(z_i), dim=0)
        z_j_all = torch.cat(GatherLayer.apply(z_j), dim=0)
        total_batch = z_i_all.shape[0]
        z_all = torch.cat([z_i_all, z_j_all], dim=0)
        
        #######################
        ### Geometry metrics ##
        #######################

        sim = torch.mm(z_all, z_all.t())
        mask = torch.eye(2*total_batch, device=z_all.device, dtype=torch.bool)
        sim.masked_fill_(mask, -float("inf"))

        idx = torch.arange(2*total_batch, device=z_all.device)
        pos_idx = (idx + total_batch) % (2*total_batch)
        pos_buffer .append(sim[idx, pos_idx])

        ## Now modify sim for calculating hard negatives
        sim[idx, pos_idx] = -float("inf")
        neg_buffer .append(sim.max(dim=1).values)

        #######################
        # Effective dimension #
        #######################
        
        z_all = z_all - z_all.mean(dim=0, keepdim=True)
        cov += z_all.T @ z_all
        n += z_all.shape[0]

    ## Now calculate the covariance info
    cov = cov / (n - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    deff = (eigvals.sum() ** 2) / (eigvals.pow(2).sum())
    lambda1_ratio = eigvals.max() / eigvals.sum()

    ## Calculate the SimCLR geometry values
    all_pos = torch.cat(pos_buffer, dim=0)
    all_neg = torch.cat(neg_buffer, dim=0)        
    gap = all_pos - all_neg
    pos_mean = all_pos.mean()
    neg_mean = all_neg.mean()
    gap_mean = gap.mean()
    gap_std  = gap.std(unbiased=False)

    return {"pos": pos_mean.item(),
            "hard_neg": neg_mean.item(),
            "gap": gap_mean.item(),
            "gap_std": gap_std.item(),
            "deff": deff.item(),
            "l1_ratio": lambda1_ratio.item(),
            "eigvals": eigvals.flip(0)[:10].cpu()
            }


def build_param_groups_ADAM(encoder, heads, weight_decay):

    decay = []
    no_decay = []

    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            continue
        
        if (param.ndim == 1
            or name.endswith(".bias")):
            no_decay.append(param)
        else:
            decay.append(param)
        
    head_pars = []
    for module in list(heads.values()):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            head_pars.append(param)
                
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
	{"params": head_pars, "weight_decay": 0.0},
    ]

def build_param_groups_LARS(encoder, heads, weight_decay):

    lars_params = []
    no_lars_params = []

    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            continue
        
        if (param.ndim == 1
            or name.endswith(".bias")):
            no_lars_params.append(param)
        else:
            lars_params.append(param)

    head_pars = []
    for module in list(heads.values()):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            head_pars.append(param)
            
    return [
        {"params": lars_params, "weight_decay": weight_decay},
        {"params": no_lars_params, "weight_decay": 0.0, "lars_exclude": True},
        {"params": head_pars, "weight_decay": 0.0},
    ]


## Wrapped training function
def run_training(rank, world_size, args):

    torch.autograd.set_detect_anomaly(False)
    ## For timing
    tstart = time.time()

    ## For parallel work
    setup(rank, world_size)
    ## Need a local device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    ## Setup the encoder
    encoder = get_encoder(args)
    encoder = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(encoder)
    encoder_nchan_instance = encoder.get_nchan_instance()
    encoder_nchan_cluster = encoder.get_nchan_cluster()
    encoder .to(device)
    encoder = DDP(encoder, device_ids=[rank])  ## Sort out parallel models (e.g., one is sent to each GPU)

    ## Dictionary of heads
    heads = {}
    
    ## Dictionary of loss functions
    loss_fns = {}

    ## Set up head and loss for projection space
    proj_head = get_projhead(encoder_nchan_instance, args)
    # proj_head = nn.SyncBatchNorm.convert_sync_batchnorm(proj_head)
    proj_head.to(device)
    proj_head = DDP(proj_head, device_ids=[rank])
    heads["proj"] = proj_head
    loss_fns["proj"] = NTXentMergedMultiGPU(args.proj_temp)
    
    ## Optionally include the head and loss for the clustering space
    if args.clust_arch != "none":
        clust_head = get_clusthead(encoder_nchan_cluster, args)
        clust_head .to(device)
        clust_head = DDP(clust_head, device_ids=[rank])
        heads["clust"] = clust_head
    
        if args.sharpened_cluster_loss == 0:
            loss_fns["clust"] = ClusteringLossMergedMultiGPU(args.clust_temp, args.entropy_scale)
        else:
            loss_fns["clust"] = SharpenedClusterLoss(args.clust_temp, 0.05, args.entropy_scale)
        
    ## Set up the distributed dataset
    train_dataset = get_dataset(args, rank)
    train_loader = get_dataloader(rank, world_size, train_dataset, args.batch_size, 6)
    nbatches   = len(train_loader)
    
    ## So we don't constantly ask args
    num_iterations = args.nstep
    log_dir = args.log
    instance_scale = args.instance_scale
    clip_gradients = bool(args.clip_gradients)
    norm_encoder = bool(args.norm_encoder)
    
    writer = None
    if rank==0 and log_dir is not None:
        print("Training with", num_iterations, "iterations")
        writer = SummaryWriter(log_dir=log_dir)

    ## Sort out the optimizer (one for each GPU...)
    optimizer, scheduler = get_opt_and_sched(args, encoder, heads, nbatches*args.nstep)
    
    ## Set up metrics
    metrics = defaultdict(list)
    
    ## Load the checkpoint if one has been given
    start_iteration = 0
    if args.restart:
        if not args.state_file:
            if rank==0: print("Restart requested, but no state file provided, aborting")
            sys.exit()
        start_iteration, metrics = load_checkpoint(encoder, heads, optimizer, args.state_file)
        if rank==0: print("Restarting from iteration", start_iteration)

    ## Load the pretrained model if given
    if args.pretrained:
        if args.restart:
            if rank==0: print("Restart requested along with a pretraining file, abort!")
            sys.exit()
        load_pretrained(encoder, heads, args.pretrained)

    ## Stuff in a profiler
    ## if rank==0:
    ##     
    ##     prof = torch.profiler.profile(
    ##         activities=[
    ##             torch.profiler.ProfilerActivity.CPU,
    ##             torch.profiler.ProfilerActivity.CUDA,
    ##         ],
    ##         record_shapes=True,
    ##         profile_memory=True,
    ##         with_stack=True,
    ##     )
    ##     prof.__enter__()
        
    ## Loop over the desired iterations
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

            if first_batch_latency == None:
                first_batch_latency = time.time() - t0
            
            ## Send to the device, then make the sparse tensors
            cat_bcoords = cat_bcoords.to(device, non_blocking=True)
            cat_bfeats  = cat_bfeats .to(device)
            cat_batch   = ME.SparseTensor(cat_bfeats, cat_bcoords, device=device)

            ## Now do the forward passes
            encoded_instance_batch, encoded_cluster_batch = encoder(cat_batch, this_batch_size)

            ## This is probably an unnecessary check, but keep it for testing
            if norm_encoder:
                encoded_cluster_batch = torch.nn.functional.normalize(encoded_cluster_batch, p=2, dim=1)
                encoded_instance_batch = torch.nn.functional.normalize(encoded_instance_batch, p=2, dim=1)
                
            ## Keep track of the total loss
            tot_loss = torch.tensor(0.0, device=device)
            # tot_loss = 0
            
            ## Deal with the projection loss
            proj_batch = heads["proj"](encoded_instance_batch)
            proj_loss = loss_fns["proj"](proj_batch)*instance_scale
            tot_loss = proj_loss
            losses_tensor["proj"] += proj_loss.detach()

            ## Add to metrics
            total_enc_align_tensor += alignment(encoded_instance_batch)
            total_enc_unif_tensor += uniformity(encoded_instance_batch)
            total_proj_align_tensor += alignment(proj_batch)
            total_proj_unif_tensor += uniformity(proj_batch)

            ## Get a few batches for calculating the running deff
            if len(buffer_enc) < nbuffer:
                with torch.no_grad():
                    buffer_enc .append(encoded_instance_batch.detach().to("cpu", non_blocking=True))
                    buffer_proj.append(proj_batch.detach().to("cpu", non_blocking=True))
                    
            ## Optionally deal with clustering loss
            if "clust" in heads:
                clust_batch = heads["clust"](encoded_cluster_batch)
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
            
            ## keep track of losses
            tot_loss_tensor += tot_loss.detach()
            
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
        enc_geom = simclr_geometry_metrics(buffer_enc, device)
        proj_geom = simclr_geometry_metrics(buffer_proj, device)
        
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

            log_grad_norm(heads["proj"].module, "proj", writer, iteration)
            log_grad_rms(heads["proj"].module, "proj", writer, iteration)
            log_grad_over_wgt(heads["proj"].module, "proj", writer, iteration)

            ## Eigenvalue debugging
            log_scalar(writer, metrics, "eigen/enc_deff", enc_geom["deff"], iteration)
            log_scalar(writer, metrics, "eigen/proj_deff", proj_geom["deff"], iteration)
            
            log_scalar(writer, metrics, "eigen/enc_l1_ratio", enc_geom["l1_ratio"], iteration)
            log_scalar(writer, metrics, "eigen/proj_l1_ratio", proj_geom["l1_ratio"], iteration)
            
            for i,val in enumerate(enc_geom["eigvals"]):
                log_scalar(writer, metrics, f"eigen/enc_lambda{i}", val, iteration)
            for i,val in enumerate(proj_geom["eigvals"]):
                log_scalar(writer, metrics, f"eigen/proj_lambda{i}", val, iteration)

            log_scalar(writer, metrics, 'monitor/enc_pos', enc_geom["pos"], iteration)
            log_scalar(writer, metrics, 'monitor/enc_hard_neg', enc_geom["hard_neg"], iteration)
            log_scalar(writer, metrics, 'monitor/enc_gap', enc_geom["gap"], iteration)
            log_scalar(writer, metrics, 'monitor/enc_gap_std', enc_geom["gap_std"], iteration)
            
            log_scalar(writer, metrics, 'monitor/proj_pos', proj_geom["pos"], iteration)
            log_scalar(writer, metrics, 'monitor/proj_hard_neg', proj_geom["hard_neg"], iteration)
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

                
            if scheduler: 
                log_scalar(writer, metrics, 'lr/train', scheduler.get_last_lr()[0], iteration)

            ## Build a string to report the outcome
            iter_string = f"Processed {iteration} / {start_iteration + num_iterations}; loss = {av_tot_loss:.4f}"
            
            if "clust" in heads:
                iter_string += f" ({av_losses['proj']:.4f} + {av_losses['clust']:.4f} + {av_entropy:.4f}); acc = {av_acc:.4f}"
            print(iter_string)
            print(f"Time taken: {(time.time()-tstart):.2f}")
            
        ## For checkpointing
        if rank==0 and iteration%25 == 0 and iteration != 0:
            save_checkpoint(encoder, heads, optimizer, args.state_file+".check"+str(iteration), iteration, metrics, args)

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

            
    ## Final version of the model
    if rank==0:
        save_checkpoint(encoder, heads, optimizer, args.state_file, iteration, metrics, args)
        if log_dir: writer.close()

    ## Report profiler if requested
    ## if rank == 0:
    ##     prof.__exit__(None, None, None)
    ##     
    ##     print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    ##     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
    ## Clear things up
    dist.destroy_process_group()

    
## Do the business
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("NN training module")

    # Basic job setup
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--nevents', type=int)
    parser.add_argument('--log', type=str, default=None)    
    parser.add_argument('--state_file', type=str)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--nstep', type=int, default=200)
    
    ## World size is the number of GPUs
    parser.add_argument('--world_size', type=int)

    ## Training dynamics
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--lars_trust_coeff', type=float, default=0.01)
    parser.add_argument('--lars_momentum', type=float, default=0.9)
    parser.add_argument('--enc_act', type=str, default="silu")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--aug_type', type=str, default=None)
    parser.add_argument('--aug_prob', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--clip_gradients', type=int, choices=[0,1], default=0)
    parser.add_argument('--norm_encoder', type=int, choices=[0,1], default=0)
    
    ## Encoder architecture choices
    parser.add_argument('--enc_arch', type=str, default=None)
    parser.add_argument('--enc_res_pool', type=int, choices=[0,1], default=0)
    parser.add_argument('--enc_stem_norm', type=int, choices=[0,1], default=0)
    parser.add_argument('--enc_stem_pool', type=int, choices=[0,1], default=0)
    parser.add_argument('--enc_stem_deep', type=int, choices=[0,1], default=1)
    parser.add_argument('--enc_layer1_norm', type=int, choices=[0,1], default=1)
    # parser.add_argument('--enc_arch_final_linear', type=int, default=512)

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
    parser.add_argument('--proj_temp', type=float, default=0.5)
    parser.add_argument('--latent', type=int, default=128)
    parser.add_argument('--nhidden', type=int, default=512)

    ## Restart option
    parser.add_argument('--restart', action='store_true')

    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))
    
    mp.spawn(run_training,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)
