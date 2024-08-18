## This file pretrains the autoencoder alone, without any contrastive loss at all. It should work with any number of GPUs distributed across nodes
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
from ME_NN_libs import AsymmetricL2LossME, EuclideanDistLoss
from ME_NN_libs import EncoderME, DecoderME, DeepEncoderME, DeepDecoderME, DeeperEncoderME, DeeperDecoderME

## For logging
from torch.utils.tensorboard import SummaryWriter

## Device and seeding
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Running with device:", device)
# torch.device(device)
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

## Import transformations
from ME_dataset_libs import CenterCrop, RandomCrop, RandomHorizontalFlip, RandomRotation2D, RandomShear2D, RandomBlockZero

## Import dataset
from ME_dataset_libs import SingleModuleImage2D_solo_ME, solo_ME_collate_fn

## For parallelising things
def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def get_dataloader(rank, world_size, train_dataset, batch_size):
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             collate_fn=solo_ME_collate_fn,
                                             batch_size=batch_size,
                                             shuffle=False,  # Set to False, as DistributedSampler handles shuffling
                                             num_workers=16,
                                             drop_last=True,
                                             prefetch_factor=2,
                                             sampler=sampler)
    return dataloader

def manage_cuda_memory(rank, gpu_threshold):
    """Check and clear GPU memory if it exceeds the threshold."""
    if torch.cuda.memory_allocated() > gpu_threshold:
        torch.cuda.empty_cache()

## Wrapped training function
def run_training(rank, world_size, num_iterations, log_dir, encoder, decoder, lr, train_dataset, batch_size, state_file=None, restart=False):

    ## For timing
    tstart = time.process_time()

    ## For parallel work
    setup(rank, world_size)
    ## Need a local device
    device = torch.device(f'cuda:{rank}')

    ## Set up the distributed dataset
    train_loader = get_dataloader(rank, world_size, train_dataset, batch_size)
    
    if rank==0: print("Training with", num_iterations, "iterations")
    start_iteration = 0
    
    ## Load the checkpoint if one has been given
    ## if restart:
    ##     if state_file:
    ##         checkpoint = torch.load(state_file, map_location=device)
    ##         encoder.load_state_dict(checkpoint['encoder_state_dict'])
    ##         decoder.load_state_dict(checkpoint['decoder_state_dict'])
    ##         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ##         start_iteration = checkpoint['epoch'] + 1
    ##         print("Restarting from iteration", start_iteration)
    ##     else:
    ##         print("Restart requested, but no state file provided, aborting")
    ##         sys.exit()
    
    if rank==0 and log_dir: writer = SummaryWriter(log_dir=log_dir)

    ## Set up the loss functions
    reco_loss_fn = AsymmetricL2LossME(10, 1)

    encoder.to(device, non_blocking=True)
    decoder.to(device)

    ## Sort out parallel models (e.g., one is sent to each GPU)
    encoder = DDP(encoder, device_ids=[rank])
    decoder = DDP(decoder, device_ids=[rank])

    if rank==0:
        print(encoder)
        print(decoder)
    
    ## Sort out the optimizer (one for each GPU...)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()},
    ]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=0)
    
    ## Set a maximum for thresholding
    total_memory = torch.cuda.get_device_properties(rank).total_memory
    gpu_threshold = 0.8 * total_memory
    
    ## Loop over the desired iterations
    for iteration in range(start_iteration, start_iteration+num_iterations):

        # Ensure shuffling with the sampler each epoch
        train_loader.sampler.set_epoch(iteration)
        
        total_loss = 0
        nbatches   = len(train_loader)
        
        # Set train mode for both the encoder and the decoder
        encoder.train()
        decoder.train()
        
        # Iterate over batches of images with the dataloader
        for orig_bcoords, orig_bfeats in train_loader:
            
            ## Send to the device, then make the sparse tensors
            orig_bcoords = orig_bcoords.to(device, non_blocking=True)
            orig_bfeats = orig_bfeats.to(device)
            orig_batch = ME.SparseTensor(orig_bfeats, orig_bcoords, device=device)            
                                    
            ## Now do the forward passes
            encoded_orig   = encoder(orig_batch)
            decoded_orig   = decoder(encoded_orig)
     
            # Evaluate loss
            loss = reco_loss_fn(decoded_orig, orig_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            total_loss += loss.item()
            
            # Manage CUDA memory for ME
            manage_cuda_memory(rank, gpu_threshold)
        
        ## See if we have an LR scheduler...
        # if scheduler: scheduler.step()
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        av_loss = total_loss / (nbatches * world_size) 

        ## Reporting, but only for rank 0
        if rank==0:
            if log_dir: 
                writer.add_scalar('loss/total', av_loss, iteration)
                #if scheduler: writer.add_scalar('lr/train', scheduler.get_last_lr()[0], iteration)
            
            print("Processed", iteration, "/", num_iterations, "; loss =", av_loss)
            print("Time taken:", time.process_time() - start)

        ## For checkpointing
        ## if iteration%25 == 0 and iteration != 0:
        ##     torch.save({
        ##         'epoch': iteration,
        ##         'encoder_state_dict': encoder.state_dict(),
        ##         'decoder_state_dict': decoder.state_dict(),
        ##         'optimizer_state_dict': optimizer.state_dict(),
        ##         'loss': loss
        ##     }, state_file+".check"+str(iteration))
        
    ## Final version of the model
    ## torch.save({
    ##     'epoch': iteration,
    ##     'encoder_state_dict': encoder.state_dict(),
    ##     'decoder_state_dict': decoder.state_dict(),
    ##     'optimizer_state_dict': optimizer.state_dict(),
    ##     'loss': loss
    ## }, state_file)    

    ## Close logging
    if rank==0 and log_dir: writer.close()

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

    ## Optional
    parser.add_argument('--latent', type=int, default=8, nargs='?')
    parser.add_argument('--arch', type=str, default=None, nargs='?')
    parser.add_argument('--nstep', type=int, default=200, nargs='?')    
    parser.add_argument('--nchan', type=int, default=16, nargs='?')
    parser.add_argument('--scheduler', type=str, default=None, nargs='?')

    ## Restart option
    parser.add_argument('--restart', action='store_true')

    ## Add augment option
    parser.add_argument('--augment', action='store_true')
    
    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))
    
    ## Other hard-coded values
    batch_size=512
    act_fn=ME.MinkowskiReLU

    ## Hard code the transform for now...    
    aug_transform = transforms.Compose([
        RandomShear2D(0.1, 0.1),
        RandomHorizontalFlip(),
        RandomRotation2D(-10,10),
        RandomBlockZero(5, 6),
        RandomCrop()
    ])

    ## Should I augment?
    transform=CenterCrop()
    if args.augment:
        print("Augmenting the data")
        transform = aug_transform
    
    ## Get the concrete dataset
    train_dataset = SingleModuleImage2D_solo_ME(args.indir, transform=transform, max_events=args.nevents)

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
    encoder=enc(args.nchan, args.latent, act_fn, 0)
    decoder=dec(args.nchan, args.latent, act_fn)

    scheduler = None

    if args.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, total_steps=num_iterations, cycle_momentum=False)
    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[150,300,450],
                                                   gamma=0.1,
                                                   last_epoch=-1,
                                                   verbose=False)
    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Number of GPUs
    world_size = 4

    mp.spawn(run_training,
             args=(world_size,
                   args.nstep,
                   args.log,
                   encoder,
                   decoder,
                   args.lr,
                   train_dataset,
                   batch_size,
                   args.state_file,
                   args.restart),
             nprocs=world_size,
             join=True)
