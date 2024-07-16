import numpy as np
import argparse
import os
from torch import nn, optim
# from torchvision import transforms
import sys
import torchvision.transforms.v2 as transforms
import random
from glob import glob
from bisect import bisect
import h5py
import MinkowskiEngine as ME

## Includes from my libraries for this project
from ME_NN_libs import AsymmetricL2LossME, EuclideanDistLoss
from ME_NN_libs import EncoderME, DecoderME

## For logging
from torch.utils.tensorboard import SummaryWriter

import torch
from torch.utils.data import Dataset
import time

## Device and seeding
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running with device:", device)
torch.device(device)
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

## This is a transformation for the nominal image
class CenterCrop:
    def __init__(self):
        self.orig_y = 280
        self.orig_x = 140
        self.new_y = 256
        self.new_x = 128
        self.pad_y = (self.orig_y - self.new_y)/2
        self.pad_x = (self.orig_x - self.new_x)/2
        
    def __call__(self, coords, feats):
        
        coords = coords - np.array([self.pad_y, self.pad_x])
        mask = (coords[:,0] > 0) & (coords[:,0] < (self.new_y)) \
             & (coords[:,1] > 0) & (coords[:,1] < (self.new_x))
        
        return coords[mask], feats[mask]

    
## This just takes a 256x128 subimage from the original 280x140 block
class RandomCrop:
    def __init__(self):
        self.orig_y = 280
        self.orig_x = 140
        self.new_y = 256
        self.new_x = 128       

    def __call__(self, coords, feats):
        ## Need to copy the array
        new_coords = coords.copy()
        new_feats = feats.copy()
        
        shift_y = random.randint(0, self.orig_y - self.new_y)
        shift_x = random.randint(0, self.orig_x - self.new_x)
        
        new_coords = new_coords - np.array([shift_x, shift_y])
        mask = (new_coords[:,0] > 0) & (new_coords[:,0] < (self.new_y)) \
             & (new_coords[:,1] > 0) & (new_coords[:,1] < (self.new_x))
        
        return new_coords[mask], new_feats[mask]
    
    
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
        self.ncols = 128
        
    def __call__(self, coords, feats):
        
        ## Need to copy the array
        new_coords = coords.copy()
        
        if torch.rand(1) < self.p:
            new_coords[:,1] = self.ncols - 1 - new_coords[:,1]
        return new_coords,feats
    
    
## Need to define a fairly standard functions that work for ME tensors
class RandomRotation2D:
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def _M(self, theta):
        """Generate a 2D rotation matrix for a given angle theta."""
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

    def __call__(self, coords, feats):
        """Apply a random rotation to 2D coordinates and return the rotated coordinates with features."""
        # Generate a random rotation angle
        angle = np.deg2rad(torch.FloatTensor(1).uniform_(self.min_angle, self.max_angle).item())

        # Get the 2D rotation matrix
        R = self._M(angle)
        # Apply the rotation
        rotated_coords = coords @ R
        return rotated_coords, feats
    
class RandomShear2D:
    def __init__(self, max_shear_x, max_shear_y):
        self.max_shear_x = max_shear_x
        self.max_shear_y = max_shear_y

    def __call__(self, coords, feats):
        """Apply a random rotation to 2D coordinates and return the rotated coordinates with features."""
        # Generate a random rotation angle
        shear_x = np.random.uniform(-self.max_shear_x, self.max_shear_x)
        shear_y = np.random.uniform(-self.max_shear_y, self.max_shear_y)

        shear_matrix = np.array([
            [1, shear_x],
            [shear_y, 1]
        ])
        
        rotated_coords = coords @ shear_matrix
        return rotated_coords, feats
    
    
## A function to randomly remove some number of blocks of size
## This has to be called before the cropping as it uses the original image size
class RandomBlockZero:
    def __init__(self, max_blocks=4, block_size=6):
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.xmax = 140
        self.ymax = 280

    def __call__(self,  coords, feats):

        combined_mask = np.full(feats.size, True, dtype=bool)
        
        num_blocks_removed = random.randint(0, self.max_blocks)
        for _ in range(num_blocks_removed):
            this_size = self.block_size
            block_x = random.randint(0, self.xmax - this_size - 1)
            block_y = random.randint(0, self.ymax - this_size - 1)
            
            mask = ~((coords[:,0] > block_y) & (coords[:,0] < (block_y+this_size)) \
                   & (coords[:,1] > block_x) & (coords[:,1] < (block_x+this_size)))
            combined_mask = np.logical_and(combined_mask, mask)
            
        ## Need to copy the array
        new_coords = coords.copy()
        new_feats = feats.copy()
        
        return new_coords[combined_mask], new_feats[combined_mask]


class SingleModuleImage2D_MultiHDF5_ME(Dataset):

    def __init__(self, infile_dir, nom_transform, aug_transform=None, max_events=None):
        self.hdf5_files = sorted(glob(os.path.join(infile_dir, '*.h5')))
        self.file_indices = []
        self.nom_transform = nom_transform
        self.aug_transform = aug_transform        
        self.max_events = max_events
        
        ## Sort out the file map
        self.create_file_indices()

        ## Apply some limitation to the size
        if self.max_events and max_events < self.length:
            self.length = self.max_events
         
    def create_file_indices(self):
        cumulative_size = 0
        
        for file in self.hdf5_files:
            self.file_indices.append(cumulative_size)
            f = h5py.File(file, 'r', libver='latest')
            cumulative_size += f.attrs['N']
            f .close()
        self.file_indices.append(cumulative_size)
        self.length = cumulative_size
        
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):

        file_index = bisect(self.file_indices, idx)-1
        this_idx = idx - self.file_indices[file_index]
        
        f = h5py.File(self.hdf5_files[file_index], 'r') 
        group = f[str(this_idx)]
        data = group['data'][:]
        row = group['row'][:]
        col = group['col'][:]

        ## Use the format that ME requires
        ## Note that we can't build the sparse tensor here because ME uses some sort of global indexing
        ## And this function is replicated * num_workers
        raw_coords = np.vstack((row, col)).T #.copy()
        raw_feats = data.reshape(-1, 1)  # Reshape data to be of shape (N, 1)
            
        ## Apply transforms to augment the data
        if not self.aug_transform:
            raw_coords, raw_feats = self.nom_transform(raw_coords, raw_feats)
            aug1_coords,aug1_feats = raw_coords,raw_feats
            aug2_coords,aug2_feats = raw_coords,raw_feats
        else:
            aug1_coords, aug1_feats = self.aug_transform(raw_coords, raw_feats)
            aug2_coords, aug2_feats = self.aug_transform(raw_coords, raw_feats)

            ## Make sure the images aren't empty...
            while aug1_feats.size == 0: aug1_coords, aug1_feats = self.aug_transform(raw_coords, raw_feats)
            while aug2_feats.size == 0: aug2_coords, aug2_feats = self.aug_transform(raw_coords, raw_feats)

            raw_coords, raw_feats   = self.nom_transform(raw_coords, raw_feats)

        return aug1_coords, aug1_feats, aug2_coords, aug2_feats, raw_coords, raw_feats
    
def triple_ME_collate_fn(batch):
    aug1_coords, aug1_feats, aug2_coords, aug2_feats, raw_coords, raw_feats = zip(*batch)
    
    # Create batched coordinates for the SparseTensor input
    aug1_bcoords = ME.utils.batched_coordinates(aug1_coords)
    aug2_bcoords = ME.utils.batched_coordinates(aug2_coords)
    raw_bcoords  = ME.utils.batched_coordinates(raw_coords)
    
    # Concatenate all lists
    aug1_bfeats = torch.from_numpy(np.concatenate(aug1_feats, 0)).float()
    aug2_bfeats = torch.from_numpy(np.concatenate(aug2_feats, 0)).float()
    raw_bfeats  = torch.from_numpy(np.concatenate(raw_feats, 0)).float()
    
    return aug1_bcoords, aug1_bfeats, aug2_bcoords, aug2_bfeats, raw_bcoords, raw_bfeats



## Wrap the training in a nicer function...
def run_training(num_iterations, log_dir, encoder, decoder, dataloader, optimizer, latent_scale=1.0, scheduler=None, state_file=None, restart=False):

    print("Training with", num_iterations, "iterations")
    tstart = time.time()
    start_iteration = 0
    
    ## Load the checkpoint if one has been given
    if restart:
        if state_file:
            checkpoint = torch.load(state_file, map_location=device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iteration = checkpoint['epoch'] + 1
            print("Restarting from iteration", start_iteration)
        else:
            print("Restart requested, but no state file provided, aborting")
            sys.exit()
    
    if log_dir: writer = SummaryWriter(log_dir=log_dir)

    ## Set up the loss functions
    reco_loss_fn = AsymmetricL2LossME(10, 1)
    latent_loss_fn = EuclideanDistLoss(latent_scale)

    encoder.to(device, non_blocking=True)
    decoder.to(device)
    
    ## Set a maximum for thresholding
    total_memory = torch.cuda.get_device_properties(0).total_memory
    gpu_threshold = 0.8 * total_memory
    
    ## Loop over the desired iterations
    for iteration in range(start_iteration, start_iteration+num_iterations):
        
        total_loss = 0
        total_aug1_loss = 0
        total_aug2_loss = 0
        total_orig_loss = 0
        total_latent_loss = 0
        nbatches   = 0
        
        # Set train mode for both the encoder and the decoder
        encoder.train()
        decoder.train()
        
        # Iterate over batches of images with the dataloader
        for aug1_bcoords, aug1_bfeats, aug2_bcoords, aug2_bfeats, orig_bcoords, orig_bfeats in dataloader:
            
            ## Send to the device, then make the sparse tensors
            aug1_bcoords = aug1_bcoords.to(device, non_blocking=True)
            aug1_bfeats = aug1_bfeats.to(device, non_blocking=True)
            aug2_bcoords = aug2_bcoords.to(device, non_blocking=True)
            aug2_bfeats = aug2_bfeats.to(device) ## This one has to block or it can try to run the forward function without the data on the GPU...
            #orig_bcoords = orig_bcoords.to(device, non_blocking=True)
            #orig_bfeats = orig_bfeats.to(device)
            
            aug1_batch = ME.SparseTensor(aug1_bfeats, aug1_bcoords, device=device)
            aug2_batch = ME.SparseTensor(aug2_bfeats, aug2_bcoords, device=device)
            #orig_batch = ME.SparseTensor(orig_bfeats, orig_bcoords, device=device)            
                                    
            ## Now do the forward passes
            encoded_batch1 = encoder(aug1_batch)
            decoded_batch1 = decoder(encoded_batch1, aug1_batch.coordinate_map_key)
            encoded_batch2 = encoder(aug2_batch)
            decoded_batch2 = decoder(encoded_batch2, aug2_batch.coordinate_map_key)
            #encoded_orig   = encoder(orig_batch)
            #decoded_orig   = decoder(encoded_orig)
     
            # Evaluate loss
            aug1_loss = reco_loss_fn(decoded_batch1, aug1_batch)
            aug2_loss = reco_loss_fn(decoded_batch2, aug2_batch)
            # orig_loss = reco_loss_fn(decoded_orig, orig_batch)
            # print(encoded_batch1.C, encoded_batch2.C)
            latent_loss = latent_loss_fn(encoded_batch1.F, encoded_batch2.F)
            
            loss = aug1_loss + aug2_loss + latent_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            
            total_loss += loss.item()
            total_aug1_loss += aug1_loss.item()
            total_aug2_loss += aug2_loss.item()
            # total_orig_loss += orig_loss.item()
            total_latent_loss += latent_loss.item()
            nbatches += 1
            
            if torch.cuda.memory_allocated() > gpu_threshold:
                torch.cuda.empty_cache()
        
        ## See if we have an LR scheduler...
        if scheduler: scheduler.step()
        
        av_loss = total_loss/nbatches
        av_aug1_loss = total_aug1_loss/nbatches
        av_aug2_loss = total_aug2_loss/nbatches
        av_orig_loss = total_orig_loss/nbatches
        av_latent_loss = total_latent_loss/nbatches
        
        if log_dir: 
            writer.add_scalar('loss/total', av_loss, iteration)
            writer.add_scalar('loss/aug1', av_aug1_loss, iteration)
            writer.add_scalar('loss/aug2', av_aug2_loss, iteration)
            writer.add_scalar('loss/orig', av_orig_loss, iteration)
            writer.add_scalar('loss/latent', av_latent_loss, iteration)
           
            if scheduler: writer.add_scalar('lr/train', scheduler.get_last_lr()[0], iteration)
            
        #if iteration%10 == 0:
        print("Processed", iteration, "/", num_iterations, "; loss =", av_loss, "(", av_aug1_loss, "+", av_aug2_loss, "+", av_latent_loss, ";", av_orig_loss, ")")
        print("Time taken:", time.process_time() - start)

        ## For checkpointing
        if iteration%25 == 0 and iteration != 0:
            torch.save({
                'epoch': iteration,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, state_file+".check"+str(iteration))
        
    ## Final version of the model
    torch.save({
        'epoch': iteration,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, state_file)    

    ## Close logging
    if log_dir: writer.close()
    
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
    parser.add_argument('--latent_scale', type=float, default=1, nargs='?')
    parser.add_argument('--nstep', type=int, default=200, nargs='?')    
    parser.add_argument('--nchan', type=int, default=16, nargs='?')
    parser.add_argument('--scheduler', type=str, default=None, nargs='?')

    ## Restart option
    parser.add_argument('--restart', action='store_true')

    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))
    
    ## Other hard-coded values
    batch_size=512
    weight_decay=0
    act_fn=ME.MinkowskiReLU

    ## Hard code the transform for now...    
    aug_transform = transforms.Compose([
        RandomShear2D(0.1, 0.1),
        RandomHorizontalFlip(),
        RandomRotation2D(-10,10),
        RandomBlockZero(5, 6),
        RandomCrop()
    ])

    ## Get a concrete dataset and data loader
    start = time.process_time() 
    train_dataset = SingleModuleImage2D_MultiHDF5_ME(args.indir, nom_transform=CenterCrop(), aug_transform=aug_transform, max_events=args.nevents)
    print("Time taken to load", train_dataset.__len__(),"images:", time.process_time() - start)
    
    ## Randomly chosen batching
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           collate_fn=triple_ME_collate_fn,
                                           batch_size=512,
                                           shuffle=True, 
                                           num_workers=16,
                                           drop_last=True,
                                           pin_memory=True,
                                           prefetch_factor=2)

    ## Dropout is 0 for now...
    encoder=EncoderME(args.nchan, args.latent, act_fn, 0)
    decoder=DecoderME(args.nchan, args.latent, act_fn)
    encoder.to(device, non_blocking=True)
    decoder.to(device)

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=weight_decay)
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

    run_training(args.nstep,
                 args.log,
                 encoder,
                 decoder,
                 train_loader,
                 optimizer,
                 args.latent_scale,
                 scheduler,
                 args.state_file,
                 args.restart)
