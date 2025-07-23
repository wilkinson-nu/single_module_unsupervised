## This file trains a simple encoder with a contrastive loss. It should work with any number of GPUs distributed across nodes
import numpy as np
import argparse
from torch import optim
import sys
import torchvision.transforms.v2 as transforms
import MinkowskiEngine as ME
import torch
import time

## The parallelisation libraries
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import ConcatDataset
from torch import nn

## Includes from my libraries for this project
from ME_NN_libs import NTXentMerged, NTXentMergedTopTenNeg, ClusteringLossMerged
from ME_NN_libs import ContrastiveEncoderFSD, ContrastiveEncoderShallowFSD
from ME_NN_libs import CCEncoderFSD, ProjectionHead, ClusteringHead

## For logging
from torch.utils.tensorboard import SummaryWriter

## Seeding
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

## Import transformations
from ME_dataset_libs import DoNothing, get_transform

## Import dataset
from ME_dataset_libs import SingleModuleImage2D_MultiHDF5_ME, cat_ME_collate_fn

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

def get_dataloader(rank, world_size, train_dataset, batch_size, num_workers=16):
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             collate_fn=cat_ME_collate_fn,
                                             batch_size=batch_size,
                                             shuffle=False,  # Set to False, as DistributedSampler handles shuffling
                                             num_workers=num_workers,
                                             drop_last=True,
                                             prefetch_factor=1,
                                             sampler=sampler)
    return dataloader

def manage_cuda_memory(rank, gpu_threshold):
    """Check and clear GPU memory if it exceeds the threshold."""
    if torch.cuda.memory_allocated(rank) > gpu_threshold:
        torch.cuda.empty_cache()

def load_pretrained(encoder, proj_head, clust_head, file_name):
    checkpoint = torch.load(file_name, map_location='cpu')
    encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
    proj_head.module.load_state_dict(checkpoint['proj_head_state_dict'])
    clust_head.module.load_state_dict(checkpoint['clust_head_state_dict'])    
    return

def load_checkpoint(encoder, proj_head, clust_head, optimizer, state_file_name):
    checkpoint = torch.load(state_file_name, map_location='cpu')
    encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
    proj_head.module.load_state_dict(checkpoint['proj_head_state_dict'])
    clust_head.module.load_state_dict(checkpoint['clust_head_state_dict']) 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'].cpu())
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    return checkpoint['epoch'] + 1

def save_checkpoint(encoder, proj_head, clust_head, optimizer, state_file_name, iteration, loss):
    torch.save({
        'epoch': iteration,
        'encoder_state_dict': encoder.module.state_dict(),
        'proj_head_state_dict': proj_head.module.state_dict(),
        'clust_head_state_dict': clust_head.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
        'loss': loss
    }, state_file_name)

def get_act_from_string(act_name):

    ## For the hidden layers
    if act_name == "relu":
        return ME.MinkowskiReLU
    if act_name == "leakyrelu":
        return ME.MinkowskiLeakyReLU
    if act_name == "gelu":
        return ME.MinkowskiGELU
    if act_name in ["silu", "swish"]:
        return ME.MinkowskiSiLU
    if act_name == "selu":
        return ME.MinkowskiSELU

    ## For the bottleneck
    if act_name == "tanh":
        return ME.MinkowskiTanh
    if act_name == "softsign":
        return ME.MinkowskiSoftsign

    return None

    
## Wrapped training function
def run_training(rank, world_size, num_iterations, log_dir, enc, enc_act_name, \
                 nchan, nlatent, nclusters, lr, weight_decay, dropout, proj_loss, proj_temp, clust_loss, clust_temp, train_dataset, \
                 batch_size, sched, state_file=None, pretrained=None, restart=False):
    torch.autograd.set_detect_anomaly(True)
    ## For timing
    tstart = time.time()

    ## For parallel work
    setup(rank, world_size)
    ## Need a local device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    ## Set up the heads
    hidden_act_fn = nn.SiLU
    latent_act_fn=nn.Tanh
    proj_head = ProjectionHead(nchan, nlatent, hidden_act_fn, latent_act_fn)
    clust_head = ClusteringHead(nchan, nclusters, hidden_act_fn)

    ## Setup the encoder
    enc_act_fn=get_act_from_string(enc_act_name)
    encoder = enc(nchan, enc_act_fn, dropout)
    encoder = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(encoder)
    
    ## Set up the distributed dataset
    train_loader = get_dataloader(rank, world_size, train_dataset, batch_size, 16)
    
    if rank==0:
        print("Training with", num_iterations, "iterations")
        if log_dir: writer = SummaryWriter(log_dir=log_dir)

    ## Set up the loss functions
    proj_loss_fn = proj_loss(proj_temp)
    clust_loss_fn = clust_loss(clust_temp, 0.1)
    
    encoder.to(device)
    proj_head.to(device)
    clust_head.to(device)
    
    ## Sort out parallel models (e.g., one is sent to each GPU)
    encoder = DDP(encoder, device_ids=[rank])
    proj_head = DDP(proj_head, device_ids=[rank])
    clust_head = DDP(clust_head, device_ids=[rank])

    ## Sort out the optimizer (one for each GPU...)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': proj_head.parameters()},
        {'params': clust_head.parameters()},
    ]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)

    ## Deal with a scheduler
    scheduler = None
    if sched == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*1000, total_steps=num_iterations, cycle_momentum=False)
    if sched == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[150,300,450],
                                                   gamma=0.1,
                                                   last_epoch=-1,
                                                   verbose=False)
    if sched == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=0,
                                                         cooldown=2,
                                                         threshold=5e-3,
                                                         threshold_mode='rel')

    
    ## Load the checkpoint if one has been given
    start_iteration = 0
    if restart:
        if not state_file:
            if rank==0: print("Restart requested, but no state file provided, aborting")
            sys.exit()
        start_iteration = load_checkpoint(encoder, proj_head, clust_head, optimizer, state_file)
        if rank==0: print("Restarting from iteration", start_iteration)

    ## Load the pretrained autoencoder if given
    if pretrained:
        if restart:
            print("Restart requested along with a pretraining file, abort!")
            sys.exit()
        load_pretrained(encoder, pretrained)
        
    if rank==0 and log_dir: writer = SummaryWriter(log_dir=log_dir)

    ## Loop over the desired iterations
    for iteration in range(start_iteration, start_iteration+num_iterations):

        # Ensure shuffling with the sampler each epoch
        train_loader.sampler.set_epoch(iteration)
        
        tot_loss_tensor = torch.tensor(0.0, device=device)  
        proj_loss_tensor = torch.tensor(0.0, device=device)        
        clust_loss_tensor = torch.tensor(0.0, device=device)        

        nbatches   = len(train_loader)
        
        # Set train mode for both the encoder and the decoder
        encoder.train()
        
        # Iterate over batches of images with the dataloader
        for cat_bcoords, cat_bfeats, this_batch_size in train_loader:

            ## Send to the device, then make the sparse tensors
            cat_bcoords = cat_bcoords.to(device, non_blocking=True)
            cat_bfeats  = cat_bfeats .to(device)
            cat_batch   = ME.SparseTensor(cat_bfeats, cat_bcoords, device=device)

            ## Now do the forward passes
            encoded_batch = encoder(cat_batch, this_batch_size)
            proj_batch = proj_head(encoded_batch.F)
            clust_batch = clust_head(encoded_batch.F)

            proj_loss = proj_loss_fn(proj_batch)
            clust_loss = clust_loss_fn(clust_batch)
            tot_loss = proj_loss + clust_loss

            # Backward pass
            optimizer.zero_grad()
            tot_loss .backward()
            optimizer.step()            

            ## keep track of losses
            tot_loss_tensor += tot_loss.detach()
            proj_loss_tensor += proj_loss.detach()
            clust_loss_tensor += clust_loss.detach()
            
            # Manage CUDA memory for ME
            torch.cuda.empty_cache()
            
        dist.all_reduce(tot_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(proj_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(clust_loss_tensor, op=dist.ReduceOp.SUM)
        
        av_tot_loss = tot_loss_tensor.item() / (nbatches * world_size) 
        av_proj_loss = proj_loss_tensor.item() / (nbatches * world_size) 
        av_clust_loss = clust_loss_tensor.item() / (nbatches * world_size) 

        ## See if we have an LR scheduler...
        if scheduler:
            if sched == "plateau": scheduler.step(av_tot_loss)
            else: scheduler.step()

        ## Reporting, but only for rank 0
        if rank==0:
            if log_dir: 
                writer.add_scalar('loss/total', av_tot_loss, iteration)              
                writer.add_scalar('loss/proj', av_proj_loss, iteration)              
                writer.add_scalar('loss/clust', av_clust_loss, iteration)              
                if scheduler: writer.add_scalar('lr/train', scheduler.get_last_lr()[0], iteration)
            
            print("Processed", iteration, "/", start_iteration + num_iterations, "; loss =", av_tot_loss, "(",av_proj_loss,"+", av_clust_loss,")")
            print("Time taken:", time.time() - tstart)

        ## For checkpointing
        if rank==0 and iteration%10 == 0 and iteration != 0:
            save_checkpoint(encoder, proj_head, clust_head, optimizer, state_file+".check"+str(iteration), iteration, av_tot_loss)
        
    ## Final version of the model
    if rank==0:
        save_checkpoint(encoder, proj_head, clust_head, optimizer, state_file, iteration, av_tot_loss)
        if log_dir: writer.close()

    ## Clear things up
    dist.destroy_process_group()

    
## Do the business
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("NN training module")

    # Add arguments
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--nevents', type=int)
    parser.add_argument('--log', type=str)    
    parser.add_argument('--lr', type=float)
    parser.add_argument('--state_file', type=str)
    
    ## World size is the number of GPUs
    parser.add_argument('--world_size', type=int)
    
    ## Optional
    parser.add_argument('--pretrained', type=str, default=None, nargs='?')
    parser.add_argument('--latent', type=int, default=128, nargs='?')
    parser.add_argument('--nclusters', type=int, default=20, nargs='?')
    parser.add_argument('--nstep', type=int, default=200, nargs='?')    
    parser.add_argument('--nchan', type=int, default=16, nargs='?')
    parser.add_argument('--scheduler', type=str, default=None, nargs='?')
    parser.add_argument('--enc_act', type=str, default="silu", nargs='?')
    parser.add_argument('--dropout', type=float, default=0, nargs='?')
    parser.add_argument('--proj_loss', type=str, default=None, nargs='?')
    parser.add_argument('--proj_temp', type=float, default=0.5, nargs='?')
    parser.add_argument('--aug_type', type=str, default=None, nargs='?')
    parser.add_argument('--batch_size', type=int, default=512, nargs='?')
    parser.add_argument('--weight_decay', type=float, default=0, nargs='?')

    ## With the new clustering loss
    parser.add_argument('--clust_temp', type=float, default=0.5, nargs='?')    
    
    ## This changes the architecture
    parser.add_argument('--arch', type=str, default=None, nargs='?')

    ## For adding simulation files, frac_data is the fraction of nevents which should be simulation
    parser.add_argument('--sim_dir', type=str, default=None, nargs='?')    
    parser.add_argument('--frac_data', type=float, default=1.0, nargs='?')
    
    ## Restart option
    parser.add_argument('--restart', action='store_true')

    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))

    ## Get the augmentation from the argument name
    aug_transform = get_transform('fsd', args.aug_type)
    
    ## Get the concrete dataset
    ## train_dataset now has a mix of simulation and data, with a controllable fraction
    nsim = 0
    ndata = int(args.nevents*args.frac_data)
    nsim = args.nevents - ndata

    data_dataset = SingleModuleImage2D_MultiHDF5_ME(args.data_dir, \
                                                     nom_transform=DoNothing(), \
                                                     aug_transform=aug_transform, \
                                                     max_events=ndata)
    if nsim > 0:
        print("Training with", ndata, "data and", nsim, "simulation events!")
        sim_dataset = SingleModuleImage2D_MultiHDF5_ME(args.sim_dir, \
                                                     nom_transform=DoNothing(), \
                                                     aug_transform=aug_transform, \
                                                     max_events=nsim)
        train_dataset = ConcatDataset([data_dataset, sim_dataset])
    else:
        print("Training with", ndata, "data events!")
        train_dataset = data_dataset

    ## Only one architecture for now
    enc = CCEncoderFSD

    #if args.arch == "shallow":
    #    enc = ContrastiveEncoderShallowFSD

    ## Sort out the loss functions
    proj_loss = NTXentMerged
    if args.proj_loss == 'NTXentMergedTopTenNeg':
        proj_loss = NTXentMergedTopTenNeg

    clust_loss = ClusteringLossMerged
        
    mp.spawn(run_training,
             args=(args.world_size,
                   args.nstep,
                   args.log,
                   enc,
                   args.enc_act,
                   args.nchan,
                   args.latent,
                   args.nclusters,
                   args.lr,
                   args.weight_decay,
                   args.dropout,
                   proj_loss,
                   args.proj_temp,
                   clust_loss,
                   args.clust_temp,
                   train_dataset,
                   args.batch_size,
                   args.scheduler,
                   args.state_file,
                   args.pretrained,
                   args.restart),
             nprocs=args.world_size,
             join=True)
