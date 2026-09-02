import numpy as np
import argparse
import sys
import MinkowskiEngine as ME
import torch
import time
import math
import random
from collections import defaultdict
from functools import partial

## The parallelisation libraries
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.profiler import profile, record_function, ProfilerActivity

## Includes from my libraries for this project
from datasets.nularbox.encoder import get_encoder
from core.models.projection_head import get_projhead
from core.models.clustering_head import get_clusthead
from core.analysis.metrics import uniformity, alignment, basic_geometry_metrics
from core.training.logging import log_scalar, log_grad_norm, log_grad_rms, log_grad_over_wgt
from core.training.scheduling import get_opt_and_sched, cosine_scheduler, update_weight_decay

from core.training.system_monitoring_utils import log_memory, log_gpu, log_vmstat
import psutil, os
from threadpoolctl import threadpool_limits

## For logging
from torch.utils.tensorboard import SummaryWriter

## Import transformations
from datasets.nularbox.augmentations_2d import get_transform

## Import dataset
from core.data.datasets import solo_labelled_collate_fn
from core.data.dataloaders import build_supervised_dataloaders

## Supervised learning specific
from core.supervised import LABEL_CLAMP, DERIVED_LABELS, DEFAULT_CLASSIFIER_CONFIG
from core.supervised import SupervisedHead, supervised_loss, ClassificationMetrics

## Utilities for multi-rank training
from core.dist_utils import setup_distributed_runtime, print0

## Checkpointing
from core.training.checkpointing import load_pretrained, load_checkpoint, save_checkpoint

## Wrapped training function
def run_training(rank, local_rank, world_size, args):

    ## For parallel work
    device = setup_distributed_runtime(
        rank,
        local_rank,
        world_size,
        seed=args.seed,
        num_workers=args.num_workers,
        print_cpu_affinity=True,
    )

    if bool(args.run_profiler) and rank==0:
        torch.cuda.set_sync_debug_mode("warn")

    torch.autograd.set_detect_anomaly(False)
    
    ## For timing
    tstart = time.time()
    
    ## Setup the encoder
    encoder = get_encoder(args)
    encoder = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(encoder)
    encoder_nchan = encoder.get_nchan()
    encoder .to(device)
    encoder = DDP(encoder, device_ids=[local_rank])  ## Sort out parallel models (e.g., one is sent to each GPU)

    ## Dictionary of heads
    heads = {}
    
    ## Dictionary of loss functions
    loss_fns = {}

    ## Set up head and loss for projection space
    sup_head = SupervisedHead(encoder_nchan,
                              classifier_config=DEFAULT_CLASSIFIER_CONFIG)
    sup_head .to(device)
    sup_head = DDP(sup_head, device_ids=[local_rank])
    heads["sup"] = sup_head
    loss_fns["sup"] = supervised_loss
    
    ## Set up the distributed dataset
    train_transform = get_transform(
        args.out_image_size,
        args.aug_type,
        args.aug_prob,
        args.aug_val,
    )
    
    val_transform = get_transform(
        args.out_image_size,
        "no_aug",
    )
    
    labelled_collate = partial(
        solo_labelled_collate_fn,
        label_clamp=LABEL_CLAMP,
        derived_labels=DERIVED_LABELS,
    )
    
    train_dataset, train_loader, val_dataset, val_loader = build_supervised_dataloaders(
        data_dir=args.data_dir,
        ntrain=args.nevents,
        nval=args.nval,
        train_transform=train_transform,
        val_transform=val_transform,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=labelled_collate,
        seed=args.seed,
    )
    nbatches   = len(train_loader)
    
    ## So we don't constantly ask args
    num_iterations = args.nepoch
    log_dir = args.log
    clip_gradients = bool(args.clip_gradients)
    norm_encoder = bool(args.norm_encoder)
    weight_decay = args.weight_decay
    weight_decay_final = args.weight_decay_final

    print0("Training with", num_iterations, "iterations")
    
    writer = None
    if rank==0 and log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)

    ## Sort out the optimizer (one for each GPU...)
    nstep_total = nbatches*args.nepoch
    optimizer, scheduler = get_opt_and_sched(args, encoder, heads, nstep_total, world_size)
    
    ## Set up metrics
    metrics = defaultdict(list)
    
    ## Load the checkpoint if one has been given
    start_iteration = 0
    if args.restart:
        if not args.state_file:
            print0("Restart requested, but no state file provided, aborting")
            sys.exit()
        start_iteration, metrics = load_checkpoint(encoder, heads, optimizer, scheduler, args.state_file)
        print0("Restarting from iteration", start_iteration)

    ## Load the pretrained model if given
    if args.pretrained:
        if args.restart:
            print0("Restart requested along with a pretraining file, abort!")
            sys.exit()
        load_pretrained(encoder, heads, args.pretrained)

    ## Stuff in a profiler
    if bool(args.run_profiler) and rank==0:
        
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA
                        ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )        
        prof.__enter__()

    ## Set up metrics:
    clf_metrics = ClassificationMetrics(DEFAULT_CLASSIFIER_CONFIG, device=device)
    val_metrics = ClassificationMetrics(DEFAULT_CLASSIFIER_CONFIG, device=device)

    ## Loop over the desired iterations
    global_iter = 0
    for iteration in range(start_iteration, start_iteration+num_iterations):

        # Ensure shuffling with the sampler each epoch
        train_loader.sampler.set_epoch(iteration)
        
        tot_loss_tensor = torch.zeros((), device=device)
        losses_tensor = {name: torch.zeros((), device=device) for name in DEFAULT_CLASSIFIER_CONFIG}
        
        ## For monitoring
        clf_metrics.reset()
        total_enc_align_tensor = torch.zeros((), device=device)
        total_enc_unif_tensor = torch.zeros((), device=device)

        ## Add more monitoring tools
        nbuffer = 5
        buffer_enc = []

        # Set train mode for the encoder and any heads
        encoder.train()
        for h in heads.values(): h.train()
        
        # Iterate over batches of images with the dataloader
        t0 = time.time()
        first_batch_latency = None
        step = 0
        for bcoords, bfeats, blabels, this_batch_size in train_loader:
            
            if first_batch_latency is None:
                first_batch_latency = time.time() - t0
                
            ## Update weight decay to allow for scheduling
            this_wd = update_weight_decay(optimizer,
	        	                  weight_decay,
                                          weight_decay_final,
                                          global_iter,
                                          nstep_total)
            
            ## Send to the device, then make the sparse tensors
            blabels = {name: val.to(device, non_blocking=True) for name, val in blabels.items()}
            bcoords = bcoords.to(device, non_blocking=True)
            bfeats  = bfeats .to(device, non_blocking=True)
            batch   = ME.SparseTensor(bfeats, bcoords, device=device)
            
            ## Now do the forward passes
            encoded_batch = encoder(batch, this_batch_size)
            
            ## L2 norm the encoder
            if norm_encoder: encoded_batch = torch.nn.functional.normalize(encoded_batch, p=2, dim=1)

            ## Deal with the projection loss
            sup_batch = heads["sup"](encoded_batch)
            sup_loss, sup_loss_dict = loss_fns["sup"](sup_batch, blabels, DEFAULT_CLASSIFIER_CONFIG)
            for name, loss_val in sup_loss_dict.items():
                losses_tensor[name] += loss_val.detach()

            ## Add to metrics
            total_enc_align_tensor += alignment(encoded_batch)
            total_enc_unif_tensor += uniformity(encoded_batch)
            
            ## Get a few batches for cealculating the running deff
            ## If the number of batches is large w.r.t. the total number (e.g., for testing), non_blocking will cause an issue here
            if len(buffer_enc) < nbuffer:
                with torch.no_grad():
                    buffer_enc .append(encoded_batch.detach().to("cpu"))
            
            ## Supervision specific metrics:
            with torch.no_grad(): clf_metrics.update(sup_batch, blabels)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            sup_loss .backward()
            
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                for h in heads.values(): torch.nn.utils.clip_grad_norm_(h.parameters(), max_norm=1.0)

            ## Update optimizer and scheduler
            optimizer.step()
            if scheduler: scheduler.step()
            
            ## Increment global_iter
            global_iter += 1
            step += 1
            
            ## keep track of losses
            tot_loss_tensor += sup_loss.detach()

        # torch.cuda.empty_cache()
            
        ## Validation pass
        encoder.eval()
        for h in heads.values(): h.eval()

        val_tot_loss_tensor = torch.zeros((), device=device)
        val_losses_tensor = {name: torch.zeros((), device=device) for name in DEFAULT_CLASSIFIER_CONFIG}
        val_metrics.reset()
        val_nbatches = 0
        step = 0

        with torch.no_grad():
            for bcoords, bfeats, blabels, this_batch_size in val_loader:
        
                blabels = {name: val.to(device, non_blocking=True) for name, val in blabels.items()}
                bcoords = bcoords.to(device, non_blocking=True)
                bfeats  = bfeats .to(device, non_blocking=True)
                batch   = ME.SparseTensor(bfeats, bcoords, device=device)
        
                encoded_batch = encoder(batch, this_batch_size)
                if norm_encoder: encoded_batch = torch.nn.functional.normalize(encoded_batch, p=2, dim=1)
        
                sup_batch = heads["sup"](encoded_batch)
                sup_loss, sup_loss_dict = loss_fns["sup"](sup_batch, blabels, DEFAULT_CLASSIFIER_CONFIG)
        
                for name, loss_val in sup_loss_dict.items():
                    val_losses_tensor[name] += loss_val
                val_tot_loss_tensor += sup_loss
        
                val_metrics.update(sup_batch, blabels)
                val_nbatches += 1
                step += 1
                
        torch.cuda.empty_cache()

        # Resume train mode for next iteration
        encoder.train()
        for h in heads.values(): h.train()
        
        ## Although the gradients are handled correctly by GatherLayer, the losses are global
        ## Strictly speaking this step isn't necessary as each mini-batch gives the same loss value
        ## But I kept it in to avoid my own headaches...
        dist.all_reduce(tot_loss_tensor, op=dist.ReduceOp.SUM)
        for name in losses_tensor.keys(): dist.all_reduce(losses_tensor[name], op=dist.ReduceOp.SUM)
        dist.all_reduce(total_enc_align_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_enc_unif_tensor, op=dist.ReduceOp.SUM)

        av_tot_loss = tot_loss_tensor.item() / (nbatches * world_size)
        av_losses = {
            name: losses_tensor[name].item() / (nbatches * world_size)
            for name in losses_tensor.keys()
        }
        av_enc_align = total_enc_align_tensor.item() / (nbatches * world_size)
        av_enc_unif  = total_enc_unif_tensor.item() / (nbatches * world_size)

        ## Other geometry calculations
        with torch.no_grad():
            enc_geom = basic_geometry_metrics(buffer_enc, device, norm_encoder)

        ## Sort out the metrics (needs to be on all ranks due to collective ops)
        clf_metrics.reduce()
        metric_results = clf_metrics.compute()

        ## Also deal with validation metrics
        dist.all_reduce(val_tot_loss_tensor, op=dist.ReduceOp.SUM)
        for name in val_losses_tensor.keys():
            dist.all_reduce(val_losses_tensor[name], op=dist.ReduceOp.SUM)
        av_val_tot_loss = val_tot_loss_tensor.item() / (val_nbatches * world_size)
        av_val_losses = {
            name: val_losses_tensor[name].item() / (val_nbatches * world_size)
            for name in val_losses_tensor.keys()
        }
        val_metrics.reduce()
        val_metric_results = val_metrics.compute()
        
        ## Reporting, but only for rank 0
        if rank==0:
            metrics["iteration"].append(iteration)
            metrics['time/total_seconds'].append(time.time()-tstart)
            log_scalar(writer, metrics, 'loss/total', av_tot_loss, iteration)
            for name in losses_tensor.keys():
                log_scalar(writer, metrics, 'loss/'+name, av_losses[name], iteration)

            ## Supervised training metrics
            for name, m in metric_results.items():
                log_scalar(writer, metrics, f'acc/{name}_accuracy',           m['accuracy'],           iteration)
                log_scalar(writer, metrics, f'acc/{name}_mean_per_class_acc', m['mean_per_class_acc'], iteration)
                log_scalar(writer, metrics, f'acc/{name}_mae',                m['mae'],                iteration)
                log_scalar(writer, metrics, f'acc/{name}_recall_nonzero',     m['recall_nonzero'],     iteration)
                ## for c, val in enumerate(m['per_class_acc']):
                ##     if not math.isnan(val):
                ##         log_scalar(writer, metrics, f'acc/{name}_class{c}_acc', val, iteration)
                
            ## Add metrics for debugging/training diagnostics
            log_scalar(writer, metrics, 'monitor/enc_alignment', av_enc_align, iteration)
            log_scalar(writer, metrics, 'monitor/enc_uniformity', av_enc_unif, iteration)
                
            ## Extensive logging for gradient debugging
            log_grad_norm(encoder.module, "encoder", writer, iteration)
            log_grad_rms(encoder.module, "encoder", writer, iteration)
            log_grad_over_wgt(encoder.module, "encoder", writer, iteration)

            log_grad_norm(heads["sup"].module, "sup", writer, iteration)
            log_grad_rms(heads["sup"].module, "sup", writer, iteration)
            log_grad_over_wgt(heads["sup"].module, "sup", writer, iteration)

            ## Eigenvalue debugging
            log_scalar(writer, metrics, "eigen/enc_deff", enc_geom["deff"], iteration)
            log_scalar(writer, metrics, "eigen/rankme_deff", enc_geom["rankme"], iteration)
            log_scalar(writer, metrics, "eigen/enc_l1_ratio", enc_geom["l1_ratio"], iteration)
            
            for i,val in enumerate(enc_geom["eigvals"]):
                log_scalar(writer, metrics, f"eigen/enc_lambda{i}", val, iteration)
                
            if scheduler: 
                log_scalar(writer, metrics, 'train/lr', scheduler.get_last_lr()[0], iteration)
            log_scalar(writer, metrics, 'train/weight_decay', this_wd, iteration)

            ## Build a string to report the outcome
            iter_string = f"Processed {iteration} / {start_iteration + num_iterations}; loss = {av_tot_loss:.4f} (val loss = {av_val_tot_loss:.4f})"
            print0(iter_string)
            print0(f"Time taken: {(time.time()-tstart):.2f}")

        ## Log validation now:
        if rank == 0:
            log_scalar(writer, metrics, 'loss/val_total', av_val_tot_loss, iteration)
            for name in val_losses_tensor.keys():
                log_scalar(writer, metrics, 'loss/val_'+name, av_val_losses[name], iteration)
                
            for name, m in val_metric_results.items():
                log_scalar(writer, metrics, f'acc/val_{name}_accuracy',           m['accuracy'],           iteration)
                log_scalar(writer, metrics, f'acc/val_{name}_mean_per_class_acc', m['mean_per_class_acc'], iteration)
                log_scalar(writer, metrics, f'acc/val_{name}_mae',                m['mae'],                iteration)
                log_scalar(writer, metrics, f'acc/val_{name}_recall_nonzero',     m['recall_nonzero'],     iteration)
                for c, val in enumerate(m['per_class_acc']):
                    if not math.isnan(val):
                        log_scalar(writer, metrics, f'acc/val_{name}_class{c}_acc', val, iteration)
            
        ## For checkpointing

        #if rank==0 and iteration%25 == 0 and iteration != 0:
        #    save_checkpoint(encoder, heads, optimizer, scheduler, args.state_file+".check"+str(iteration), iteration, metrics, args)

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
    dist.destroy_process_group()

    
## Do the business
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("SimCLR training module")

    # Basic job setup
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--nevents', type=int, required=True)
    parser.add_argument('--nval', type=int, required=True)
    parser.add_argument('--log', type=str, default=None)    
    parser.add_argument('--state_file', type=str)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--nepoch', type=int, default=200, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    
    ## Training dynamics
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--lars_trust_coeff', type=float, default=0.01)
    parser.add_argument('--lars_momentum', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--weight_decay_final', type=float, default=-1)
    parser.add_argument('--weight_decay_head', type=int, choices=[0,1], default=0)
    parser.add_argument('--clip_gradients', type=int, choices=[0,1], default=0)
    parser.add_argument('--norm_encoder', type=int, choices=[0,1], default=0)
    parser.add_argument('--non_lars_lr_scale', type=float, default=1.0)

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

    ## Restart option
    parser.add_argument('--restart', action='store_true')

    ## Optional profiler
    parser.add_argument('--run_profiler', type=int, choices=[0,1], default=0)

    # Parse arguments from command line
    args = parser.parse_args()

    ## Note global and local ranks to allow multi-node training
    rank       = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    ## Report arguments (but only rank 0)
    for arg in vars(args): print0(arg, getattr(args, arg))
    
    ## Removed mp.spawn, now requires srun
    run_training(rank, local_rank, world_size, args)
