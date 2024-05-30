import numpy as np
import joblib
import argparse
from torch import nn, optim
from torchvision import transforms
import sys
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F
import random

## Includes from my libraries for this project
from NN_libs import AsymmetricL2Loss, AsymmetricL1Loss, EuclideanDistLoss
from NN_libs import get_model

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

## Need to define a RandomRotation function that works for Tensors
class RandomTensorRotation:
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, img):
        angle = torch.FloatTensor(1).uniform_(self.min_angle, self.max_angle).item()
        return F.rotate(img.unsqueeze(0), angle).squeeze()

## A function to randomly remove some number of blocks of size
class RandomBlockZero:
    def __init__(self, max_blocks=5, block_size=4):
        self.max_blocks = max_blocks
        self.block_size = block_size

    def __call__(self, img):
        # Randomly zero out blocks of 4x4 pixels
        num_blocks_removed = random.randint(0, self.max_blocks)
        for _ in range(num_blocks_removed):
            this_size = self.block_size
            block_x = random.randint(0, img.size(1) // this_size - 1) * this_size
            block_y = random.randint(0, img.size(0) // this_size - 1) * this_size
            img[block_y:block_y+4, block_x:block_x+4] = 0
        return img    

## A function to randomly shift the image by some number of pixels and crop it
class RandomShiftTensor:
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, img):
        
        height, width = img.shape

        shift_x = random.randint(-self.max_shift, self.max_shift)
        shift_y = random.randint(-self.max_shift, self.max_shift)

        new_img = torch.zeros_like(img)

        src_x1 = max(0, -shift_x)
        src_y1 = max(0, -shift_y)
        src_x2 = min(width, width - shift_x)
        src_y2 = min(height, height - shift_y)

        tgt_x1 = max(0, shift_x)
        tgt_y1 = max(0, shift_y)
        tgt_x2 = tgt_x1 + (src_x2 - src_x1)
        tgt_y2 = tgt_y1 + (src_y2 - src_y1)

        new_img[tgt_y1:tgt_y2, tgt_x1:tgt_x2] = img[src_y1:src_y2, src_x1:src_x2]

        return new_img
    

class SingleModuleImage2D_augpair(Dataset):

    def __init__(self, infilename, transform=None):
        self._data = joblib.load(infilename)
        self._length = len(self._data)
        self._transform = transform

    def __len__(self):
        return self._length
    
    def __getitem__(self,idx):

        ## Convert the raw data to a dense pytorch tensor...
        raw_data = torch.Tensor(self._data[idx].toarray())
        
        ## Apply transforms to augment the data
        if not self._transform:
            img1 = raw_data.unsqueeze(0)
            img2 = raw_data.unsqueeze(0)
        else:
            img1 = self._transform(raw_data).unsqueeze(0)
            img2 = self._transform(raw_data).unsqueeze(0)
        
        return img1, img2, raw_data.unsqueeze(0)

def collate_triplet(batch):
    img1_batch = torch.stack([item[0] for item in batch])
    img2_batch = torch.stack([item[1] for item in batch])
    img3_batch = torch.stack([item[2] for item in batch])
    return img1_batch, img2_batch, img3_batch



## Wrap the training in a nicer function...
def run_training(num_iterations, log_dir, encoder, decoder, dataloader, optimizer, scheduler=None, state_file=None, restart=False):

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

    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()

    ## Set up the loss functions
    reco_loss_fn = AsymmetricL2Loss(10, 1)
    latent_loss_fn = EuclideanDistLoss()
    
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
        for aug_batch1, aug_batch2, orig_batch in dataloader:
                        
            # Evaluate loss for each of the batches
            aug_batch1 = aug_batch1.to(device)
            enc_aug_batch1 = encoder(aug_batch1)
            dec_aug_batch1 = decoder(enc_aug_batch1)
            aug1_loss = reco_loss_fn(dec_aug_batch1, aug_batch1, encoder, decoder)

            aug_batch2 = aug_batch2.to(device)
            enc_aug_batch2 = encoder(aug_batch2)
            dec_aug_batch2 = decoder(enc_aug_batch2)
            aug2_loss = reco_loss_fn(dec_aug_batch2, aug_batch2, encoder, decoder)

            orig_batch = orig_batch.to(device)
            enc_orig_batch = encoder(orig_batch)
            dec_orig_batch = decoder(enc_orig_batch)
            aug1_loss = reco_loss_fn(dec_orig_batch, orig_batch, encoder, decoder)

            ## Get the final component to the loss (using the two augmented batches)
            latent_loss = latent_loss_fn(enc_aug_batch1, enc_aug_batch2)
            
            loss = aug1_loss + aug2_loss + latent_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
            
            total_loss += loss.item()
            total_aug1_loss += aug1_loss.item()
            total_aug2_loss += aug2_loss.item()
            total_orig_loss += orig_loss.item()
            total_latent_loss += latent_loss.item()
            nbatches += 1
        
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
        print("Processed", iteration, "/", num_iterations, "; loss =", av_loss, "(", av_aug1_loss, "+", av_aug2_loss, "+", av_latent_loss, ";", av_orig_loss)
        print("Time taken:", time.process_time() - start)

        ## For checkpointing
        if iteration%100 == 0 and iteration != 0:
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
    parser.add_argument('--infile', type=str)
    parser.add_argument('--log', type=str)    
    parser.add_argument('--lr', type=float)
    parser.add_argument('--state_file', type=str)

    ## Optional
    parser.add_argument('--latent', type=int, default=8, nargs='?')
    parser.add_argument('--nstep', type=int, default=200, nargs='?')    
    parser.add_argument('--nchan', type=int, default=16, nargs='?')
    parser.add_argument('--scheduler', type=str, default=None, nargs='?')
    parser.add_argument('--arch_type', type=str, default="None", nargs='?')

    ## Restart option
    parser.add_argument('--restart', action='store_true')

    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))
    
    ## Other hard-coded values
    batch_size=512
    weight_decay=0
    act_fn=nn.LeakyReLU

    ## Hard code the transform for now...
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        RandomBlockZero(),
        RandomTensorRotation(-10, 10),
        RandomShiftTensor()
    ])


    ## Get a concrete dataset and data loader
    start = time.process_time() 
    train_dataset = SingleModuleImage2D_augpair(args.infile, transform=aug_transform)
    print("Time taken to load", train_dataset.__len__(),"images:", time.process_time() - start)
    
    ## Randomly chosen batching
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               collate_fn=collate_triplet,
                                               batch_size=batch_size,
                                               shuffle=True, 
                                               num_workers=16,
                                               drop_last=True)
    
    print("Found arch_type", args.arch_type)
    enc, dec = get_model(args.arch_type)
    encoder=enc(args.nchan, args.latent, act_fn)
    decoder=dec(args.nchan, args.latent, act_fn)

    print(encoder)
    print(decoder)    
    
    encoder.to(device)
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
                 scheduler,
                 args.state_file,
                 args.restart)
