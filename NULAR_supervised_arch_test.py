import numpy as np
import argparse
import sys
import MinkowskiEngine as ME
import torch
import time
import math
from collections import defaultdict
from functools import partial

## The parallelisation libraries
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
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
from core.data.datasets import single_2d_dataset_ME, solo_labelled_collate_fn

## Supervised learning specific
from core.supervised import LABEL_CLAMP, DERIVED_LABELS, DEFAULT_CLASSIFIER_CONFIG
from core.supervised import SupervisedHead, supervised_loss, ClassificationMetrics

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
    
def get_supervised_dataloaders(args, rank, world_size, num_workers=8):

    ## Get the augmentation from the argument name
    aug_transform = get_transform('256x256', args.aug_type, args.aug_prob)
    
    ## Get the concrete dataset
    full_dataset = single_2d_dataset_ME(args.data_dir, \
                                   transform=aug_transform, \
                                   max_events=args.nevents)
    if rank==0:
        print(f"Loaded {len(full_dataset)} events, "
              f"training with {args.nevents}, validating with {args.nval}")

    ## Split indices -- no shuffle needed since data is already randomly ordered
    train_indices = list(range(args.nevents))
    val_indices = list(range(args.nevents, args.nevents + args.nval))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    ## Slightly hacky way to manipulate the labels
    this_collate_fn = partial(solo_labelled_collate_fn,
                              label_clamp=LABEL_CLAMP,
                              derived_labels=DERIVED_LABELS)
    
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   collate_fn=this_collate_fn,
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=num_workers,
                                                   drop_last=True,
                                                   persistent_workers=False,
                                                   prefetch_factor=2,
                                                   sampler=sampler)
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 collate_fn=this_collate_fn,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 drop_last=True,
                                                 persistent_workers=False,
                                                 prefetch_factor=2,
                                                 sampler=val_sampler)
    return train_dataset, train_dataloader, val_dataset, val_dataloader

    
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


## Wrapped training function
def run_training(rank, world_size, args):

    ME.set_sparse_tensor_operation_mode(
        ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
    
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
    encoder .to(device)
    encoder = DDP(encoder, device_ids=[rank])  ## Sort out parallel models (e.g., one is sent to each GPU)

    ## Dictionary of heads
    heads = {}
    
    ## Dictionary of loss functions
    loss_fns = {}

    ## Set up head and loss for projection space
    sup_head = SupervisedHead(encoder_nchan_instance,
                              classifier_config=DEFAULT_CLASSIFIER_CONFIG)
    sup_head .to(device)
    sup_head = DDP(sup_head, device_ids=[rank])
    heads["sup"] = sup_head
    loss_fns["sup"] = supervised_loss
    
    ## Set up the distributed dataset
    train_dataset, train_dataloader, val_dataset, val_dataloader = get_supervised_dataloaders(args, rank, world_size, 6)
    nbatches   = len(train_loader)
    
    ## So we don't constantly ask args
    num_iterations = args.nstep
    log_dir = args.log
    clip_gradients = bool(args.clip_gradients)
    norm_encoder = bool(args.norm_encoder)
    weight_decay = args.weight_decay
    weight_decay_final = args.weight_decay_final
    
    writer = None
    if rank==0 and log_dir is not None:
        print("Training with", num_iterations, "iterations")
        writer = SummaryWriter(log_dir=log_dir)

    ## Sort out the optimizer (one for each GPU...)
    nstep_total = nbatches*args.nstep
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

    ## Set up metrics:
    clf_metrics = ClassificationMetrics(DEFAULT_CLASSIFIER_CONFIG, device=device)
        
    ## Loop over the desired iterations
    global_iter = 0
    for iteration in range(start_iteration, start_iteration+num_iterations):

        # Ensure shuffling with the sampler each epoch
        train_loader.sampler.set_epoch(iteration)
        
        tot_loss_tensor = torch.tensor(0.0, device=device)
        losses_tensor = {name: torch.tensor(0.0, device=device) for name in DEFAULT_CLASSIFIER_CONFIG}
        
        ## For monitoring
        clf_metrics.reset()
        total_enc_align_tensor = torch.tensor(0.0, device=device)
        total_enc_unif_tensor = torch.tensor(0.0, device=device)

        ## Add more monitoring tools
        nbuffer = 5
        buffer_enc = []

        # Set train mode for the encoder and any heads
        encoder.train()
        for h in heads.values(): h.train()
        
        # Iterate over batches of images with the dataloader
        t0 = time.time()
        first_batch_latency = None
        for bcoords, bfeats, blabels, this_batch_size in train_loader:
            
            if first_batch_latency == None:
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
            bfeats  = bfeats .to(device)
            batch   = ME.SparseTensor(bfeats, bcoords, device=device)

            ## Now do the forward passes
            encoded_batch = encoder(batch, this_batch_size)

            ## L2 norm the encoder
            if norm_encoder: encoded_batch = torch.nn.functional.normalize(encoded_batch, p=2, dim=1)
                           
            ## Deal with the projection loss
            sup_batch = heads["sup"](encoded_batch)
            sup_loss, sup_loss_dict = loss_fns["sup"](sup_batch, blabels, DEFAULT_CLASSIFIER_CONFIG)
            tot_loss = sup_loss
            for name, loss_val in sup_loss_dict.items():
                losses_tensor[name] += loss_val

            ## Add to metrics
            total_enc_align_tensor += alignment(encoded_batch)
            total_enc_unif_tensor += uniformity(encoded_batch)

            ## Get a few batches for calculating the running deff
            if len(buffer_enc) < nbuffer:
                with torch.no_grad():
                    buffer_enc .append(encoded_batch.detach().to("cpu", non_blocking=False))

            ## Supervision specific metrics:
            with torch.no_grad(): clf_metrics.update(sup_batch, blabels)

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

            ME.clear_global_coordinate_manager()

        # Manage CUDA memory for ME
        torch.cuda.empty_cache()

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
        with torch.no_grad(): enc_geom = basic_geometry_metrics(buffer_enc, device)

        ## Sort out the metrics (needs to be on all ranks due to collective ops)
        clf_metrics.reduce()
        metric_results = clf_metrics.compute()

        ## Reporting, but only for rank 0
        if rank==0:
            metrics["iteration"].append(iteration)
            log_scalar(writer, metrics, 'loss/total', av_tot_loss, iteration)
            for name in losses_tensor.keys():
                log_scalar(writer, metrics, 'loss/'+name, av_losses[name], iteration)

            ## Supervised training metrics
            for name, m in metric_results.items():
                log_scalar(writer, metrics, f'acc/{name}_accuracy',           m['accuracy'],           iteration)
                log_scalar(writer, metrics, f'acc/{name}_mean_per_class_acc', m['mean_per_class_acc'], iteration)
                log_scalar(writer, metrics, f'acc/{name}_mae',                m['mae'],                iteration)
                log_scalar(writer, metrics, f'acc/{name}_recall_nonzero',     m['recall_nonzero'],     iteration)
                for c, val in enumerate(m['per_class_acc']):
                    if not math.isnan(val):
                        log_scalar(writer, metrics, f'acc/{name}_class{c}_acc', val, iteration)
                
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
            log_scalar(writer, metrics, "eigen/enc_l1_ratio", enc_geom["l1_ratio"], iteration)
            
            for i,val in enumerate(enc_geom["eigvals"]):
                log_scalar(writer, metrics, f"eigen/enc_lambda{i}", val, iteration)
                
            if scheduler: 
                log_scalar(writer, metrics, 'train/lr', scheduler.get_last_lr()[0], iteration)
            log_scalar(writer, metrics, 'train/weight_decay', this_wd, iteration)

            ## Build a string to report the outcome
            iter_string = f"Processed {iteration} / {start_iteration + num_iterations}; loss = {av_tot_loss:.4f}"
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
    if bool(args.run_profiler) and rank == 0:
        prof.__exit__(None, None, None)
        
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
    ## Clear things up
    dist.destroy_process_group()

    
## Do the business
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("SimCLR training module")

    # Basic job setup
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--nevents', type=int)
    parser.add_argument('--nval', type=int)
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
    parser.add_argument('--weight_decay_final', type=float, default=-1)
    parser.add_argument('--weight_decay_head', type=int, choices=[0,1], default=0)
    parser.add_argument('--clip_gradients', type=int, choices=[0,1], default=0)
    parser.add_argument('--norm_encoder', type=int, choices=[0,1], default=0)
    
    ## Encoder architecture choices
    parser.add_argument('--enc_arch', type=str, default=None)
    parser.add_argument('--enc_arch_pool', type=str, default="avg")
    parser.add_argument('--enc_res_pool', type=int, choices=[0,1], default=0)
    parser.add_argument('--enc_stem_norm', type=int, choices=[0,1], default=0)
    parser.add_argument('--enc_init_stem_stride', type=int, default=2)
    parser.add_argument('--enc_stem_pool', type=int, choices=[0,1], default=0)
    parser.add_argument('--enc_stem_deep', type=int, choices=[0,1], default=1)
    parser.add_argument('--enc_layer1_norm', type=int, choices=[0,1], default=1)
    # parser.add_argument('--enc_arch_final_linear', type=int, default=512)

    ## Restart option
    parser.add_argument('--restart', action='store_true')

    ## Optional profiler
    parser.add_argument('--run_profiler', type=int, choices=[0,1], default=0)

    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))
    
    mp.spawn(run_training,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)
