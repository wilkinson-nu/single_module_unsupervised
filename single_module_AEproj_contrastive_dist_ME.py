## This file trains the autoencoder + a projective head with a contrastive loss. It should work with any number of GPUs distributed across nodes
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

## Includes from my libraries for this project
from ME_NN_libs import AsymmetricL2LossME, NTXent
from ME_NN_libs import EncoderME, DecoderME, DeepEncoderME, DeepDecoderME, DeeperEncoderME, DeeperDecoderME
from ME_NN_libs import ProjectionHead

## For logging
from torch.utils.tensorboard import SummaryWriter

## Seeding
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

## Import transformations
from ME_dataset_libs import CenterCrop, RandomCrop, RandomHorizontalFlip, RandomRotation2D, RandomShear2D, \
    RandomBlockZero, RandomJitterCharge, RandomScaleCharge, RandomElasticDistortion2D, RandomGridDistortion2D

## Import dataset
from ME_dataset_libs import SingleModuleImage2D_MultiHDF5_ME, triple_ME_collate_fn

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
                                             collate_fn=triple_ME_collate_fn,
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

def load_pretrained(encoder, decoder, file_name):
    checkpoint = torch.load(file_name, map_location='cpu')
    encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
    return

def load_checkpoint(encoder, decoder, project, optimizer, state_file_name):
    checkpoint = torch.load(state_file_name, map_location='cpu')
    encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
    project.module.load_state_dict(checkpoint['project._state_dict'])    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'].cpu())
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    return checkpoint['epoch'] + 1

def save_checkpoint(encoder, decoder, project, optimizer, state_file_name, iteration, loss):
    torch.save({
        'epoch': iteration,
        'encoder_state_dict': encoder.module.state_dict(),
        'decoder_state_dict': decoder.module.state_dict(),
        'project_state_dict': project.module.state_dict(),
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
def run_training(rank, world_size, num_iterations, log_dir, enc, dec, hidden_act_name, latent_act_name, \
                 nchan, latent, lr, dropout, ntx_temp, train_dataset, batch_size, sched, state_file=None, pretrained=None, restart=False):

    ## For timing
    tstart = time.process_time()

    ## For parallel work
    setup(rank, world_size)
    ## Need a local device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    ## Setup the models
    hidden_act_fn=get_act_from_string(hidden_act_name)
    latent_act_fn=get_act_from_string(latent_act_name)

    encoder=enc(nchan, latent, hidden_act_fn, latent_act_fn, dropout)
    decoder=dec(nchan, latent, hidden_act_fn)
    project=ProjectionHead([latent, latent, latent, latent], latent_act_fn)
    
    ## Set up the distributed dataset
    train_loader = get_dataloader(rank, world_size, train_dataset, batch_size, 16)
    
    if rank==0:
        print("Training with", num_iterations, "iterations")
        if log_dir: writer = SummaryWriter(log_dir=log_dir)

    ## Set up the loss functions
    reco_loss_fn = AsymmetricL2LossME(10, 1, batch_size)
    latent_loss_fn = NTXent(ntx_temp)
    
    encoder.to(device, non_blocking=True)
    decoder.to(device, non_blocking=True)
    project.to(device)
    
    ## Sort out parallel models (e.g., one is sent to each GPU)
    encoder = DDP(encoder, device_ids=[rank])
    decoder = DDP(decoder, device_ids=[rank])
    project = DDP(project, device_ids=[rank])

    ## Sort out the optimizer (one for each GPU...)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()},
        {'params': project.parameters()},
    ]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=0)

    ## Deal with a scheduler
    scheduler = None
    if sched == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*100, total_steps=num_iterations, cycle_momentum=False)
    if sched == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[150,300,450],
                                                   gamma=0.1,
                                                   last_epoch=-1,
                                                   verbose=False)
    if sched == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    
    ## Load the checkpoint if one has been given
    start_iteration = 0
    if restart:
        if not state_file:
            if rank==0: print("Restart requested, but no state file provided, aborting")
            sys.exit()
        start_iteration = load_checkpoint(encoder, decoder, project, optimizer, state_file)
        if rank==0: print("Restarting from iteration", start_iteration)

    ## Load the pretrained autoencoder if given
    if pretrained:
        if restart:
            print("Restart requested along with a pretraining file, abort!")
            sys.exit()
        load_pretrained(encoder, decoder, pretrained)
        
    if rank==0 and log_dir: writer = SummaryWriter(log_dir=log_dir)

    ## Loop over the desired iterations
    for iteration in range(start_iteration, start_iteration+num_iterations):

        # Ensure shuffling with the sampler each epoch
        train_loader.sampler.set_epoch(iteration)
        
        tot_loss_tensor = torch.tensor(0.0, device=device)
        rec_loss_tensor = torch.tensor(0.0, device=device)
        lat_loss_tensor = torch.tensor(0.0, device=device)
        
        nbatches   = len(train_loader)
        
        # Set train mode for both the encoder and the decoder
        encoder.train()
        decoder.train()
        project.train()
        
        # Iterate over batches of images with the dataloader
        for aug1_bcoords, aug1_bfeats, aug2_bcoords, aug2_bfeats, orig_bcoords, orig_bfeats in train_loader:

            ## Send to the device, then make the sparse tensors
            aug1_bcoords = aug1_bcoords.to(device, non_blocking=True)
            aug1_bfeats  = aug1_bfeats .to(device, non_blocking=True)
            aug2_bcoords = aug2_bcoords.to(device, non_blocking=True)
            aug2_bfeats  = aug2_bfeats .to(device)
            aug1_batch   = ME.SparseTensor(aug1_bfeats, aug1_bcoords, device=device)
            aug2_batch   = ME.SparseTensor(aug2_bfeats, aug2_bcoords, device=device)
                                    
            ## Now do the forward passes
            encoded_batch1 = encoder(aug1_batch)
            decoded_batch1 = decoder(encoded_batch1)
            encoded_batch2 = encoder(aug2_batch)
            decoded_batch2 = decoder(encoded_batch2)
            project_batch1 = project(encoded_batch1)
            project_batch2 = project(encoded_batch2)
            
            # Evaluate losses
            aug1_loss = reco_loss_fn(decoded_batch1, aug1_batch)
            aug2_loss = reco_loss_fn(decoded_batch2, aug2_batch)
            lat_loss  = latent_loss_fn(project_batch1.F, project_batch2.F)
            tot_loss  = aug1_loss + aug2_loss + lat_loss
            
            # Backward pass
            optimizer.zero_grad()
            tot_loss .backward()
            optimizer.step()            

            ## keep track of losses
            tot_loss_tensor += tot_loss.detach()
            rec_loss_tensor += (aug1_loss+aug2_loss).detach()
            lat_loss_tensor += lat_loss.detach()
            
            # Manage CUDA memory for ME
            torch.cuda.empty_cache()
            
        ## See if we have an LR scheduler...
        if scheduler: scheduler.step()
        dist.all_reduce(tot_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(rec_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(lat_loss_tensor, op=dist.ReduceOp.SUM)
        
        av_tot_loss = tot_loss_tensor.item() / (nbatches * world_size) 
        av_rec_loss = rec_loss_tensor.item() / (nbatches * world_size) / 2
        av_lat_loss = lat_loss_tensor.item() / (nbatches * world_size)
        
        ## Reporting, but only for rank 0
        if rank==0:
            if log_dir: 
                writer.add_scalar('loss/total', av_tot_loss, iteration)
                writer.add_scalar('loss/latent', av_lat_loss, iteration)
                writer.add_scalar('loss/reco', av_rec_loss, iteration)
              
                if scheduler: writer.add_scalar('lr/train', scheduler.get_last_lr()[0], iteration)
            
            print("Processed", iteration, "/", start_iteration + num_iterations, "; loss =", av_tot_loss, \
                  "("+av_rec_loss, "+", av_lat_loss+")")
            print("Time taken:", time.process_time() - tstart)

        ## For checkpointing
        if rank==0 and iteration%50 == 0 and iteration != 0:
            save_checkpoint(encoder, decoder, project, optimizer, state_file+".check"+str(iteration), iteration, av_loss)
        
    ## Final version of the model
    if rank==0:
        save_checkpoint(encoder, decoder, project, optimizer, state_file, iteration, av_loss)
        if log_dir: writer.close()

    ## Clear things up
    dist.destroy_process_group()

    
## Do the business
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("NN training module")

    # Add arguments
    parser.add_argument('--indir', type=str)
    parser.add_argument('--nevents', type=int)
    parser.add_argument('--log', type=str)    
    parser.add_argument('--lr', type=float)
    parser.add_argument('--state_file', type=str)
    
    ## World size is the number of GPUs
    parser.add_argument('--world_size', type=int)
    
    ## Optional
    parser.add_argument('--pretrained', type=str, default=None, nargs='?')
    parser.add_argument('--latent', type=int, default=8, nargs='?')
    parser.add_argument('--arch', type=str, default=None, nargs='?')
    parser.add_argument('--nstep', type=int, default=200, nargs='?')    
    parser.add_argument('--nchan', type=int, default=16, nargs='?')
    parser.add_argument('--scheduler', type=str, default=None, nargs='?')
    parser.add_argument('--latent_act', type=str, default="relu", nargs='?')
    parser.add_argument('--hidden_act', type=str, default="tanh", nargs='?')
    parser.add_argument('--dropout', type=float, default=0, nargs='?')
    parser.add_argument('--ntx_temp', type=float, default=0.5, nargs='?')
    
    ## Restart option
    parser.add_argument('--restart', action='store_true')

    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))
    
    ## Other hard-coded values
    batch_size=1024

    ## Hard code the transform for now...    
    aug_transform = transforms.Compose([
        RandomGridDistortion2D(5,5),
        RandomShear2D(0.1, 0.1),
        RandomHorizontalFlip(),
        RandomRotation2D(-10,10),
        RandomBlockZero(5, 6),
        RandomScaleCharge(0.02),
        RandomJitterCharge(0.02),
        RandomCrop()
    ])

    ## Get the concrete dataset
    train_dataset = SingleModuleImage2D_MultiHDF5_ME(args.indir, \
                                                     nom_transform=CenterCrop(), \
                                                     aug_transform=aug_transform, \
                                                     max_events=args.nevents)
    ## Dropout is 0 for now...
    enc, dec = None, None

    if args.arch == None or args.arch == "simple":
        print("Using the simple architecture")
        enc = EncoderME
        dec = DecoderME
    if args.arch == "deep":
        print("Using the deep architecture")
        enc = DeepEncoderME
        dec = DeepDecoderME
    if args.arch == "deeper":
        print("Using the deeper architecture")
        enc = DeeperEncoderME
        dec = DeeperDecoderME        

    mp.spawn(run_training,
             args=(args.world_size,
                   args.nstep,
                   args.log,
                   enc,
                   dec,
                   args.hidden_act,
                   args.latent_act,
                   args.nchan,
                   args.latent,
                   args.lr,
                   args.dropout,
                   args.ntx_temp,
                   train_dataset,
                   batch_size,
                   args.scheduler,
                   args.state_file,
                   args.pretrained,
                   args.restart),
             nprocs=args.world_size,
             join=True)
