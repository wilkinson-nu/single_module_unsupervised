import numpy as np
import joblib
import argparse
from torch import nn

## Get the autoencoder options I included from elsewhere
from NN_libs import Encoder, Decoder, EncoderSimple, DecoderSimple, EncoderComplex, DecoderComplex

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

class SingleModuleImage2D_sparse_joblib(Dataset):

    def __init__(self, infilename, normalize=None):
        
        self._data = joblib.load(infilename)
        self._length = len(self._data)
        self._normalize = normalize

    def __len__(self):
        return self._length
    
    def __getitem__(self,idx):

        ## Convert to a dense pytorch tensor...
        data = torch.Tensor(self._data[idx].toarray())
        
        ## Normalize entries if necessary
        if self._normalize:
            data = data/np.amax(data.numpy())

        ## By default, this is assumed to be in "Tensor, label" format. The collate function is necessary because this is different...
        return data

def collate(batch):
    batched_data = torch.cat([sample[None][None] for sample in batch],0)
    return batched_data


## Wrap the training in a nicer function...
def run_training(num_iterations, log_dir, encoder, decoder, dataloader, loss_fn, optimizer, scheduler=None):

    print("Training with", num_iterations, "iterations")
    tstart = time.time()

    if log_dir: writer = SummaryWriter(log_dir=log_dir)

    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()

    ## Loop over the desired iterations
    for iteration in range(num_iterations):
        
        total_loss = 0
        nbatches   = 0
        
        # Set train mode for both the encoder and the decoder
        encoder.train()
        decoder.train()
    
        # Iterate over batches of images with the dataloader
        for image_batch in dataloader:
                        
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_batch = encoder(image_batch)
            # Decode data
            decoded_batch = decoder(encoded_batch)
            # Evaluate loss
            loss = loss_fn(decoded_batch, image_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
            
            total_loss += loss.item()
            nbatches += 1
        
        ## See if we have an LR scheduler...
        if scheduler: scheduler.step()
        
        av_loss = total_loss/nbatches
        if log_dir: 
            writer.add_scalar('loss/train', av_loss, iteration)
        #if iteration%10 == 0:
        print("Processed", iteration, "/", num_iterations, "; loss =", av_loss)
        print("Time taken:", time.process_time() - start)

        
## Do the business
if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("NN training module")

    # Add arguments
    parser.add_argument('--infile', type=str)
    parser.add_argument('--log', type=str)    
    parser.add_argument('--lr', type=float)    
    parser.add_argument('--latent', type=int, default=8, nargs='?')
    parser.add_argument('--nstep', type=int, default=200, nargs='?')    
    parser.add_argument('--nchan', type=int, default=16, nargs='?')
    parser.add_argument('--scheduler', type=str, default=None, nargs='?')
    parser.add_argument('--loss_type', type=str, default="L2", nargs='?')
    parser.add_argument('--arch_type', type=str, default="None", nargs='?')
    parser.add_argument('--norm_data', type=int, default=0, nargs='?')

    # Parse arguments from command line
    args = parser.parse_args()

    ## Other hard-coded values
    batch_size=1024
    weight_decay=0
    act_fn=nn.LeakyReLU
    
    ## Get a concrete dataset and data loader
    start = time.process_time() 
    train_dataset = SingleModuleImage2D_sparse_joblib(args.infile, args.norm_data)
    print("Time taken to load", train_dataset.__len__(),"images:", time.process_time() - start)
    
    ## Randomly chosen batching
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               collate_fn=collate,
                                               batch_size=batch_size,
                                               shuffle=True, 
                                               num_workers=4,
                                               drop_last=True)
    
    loss_fn = torch.nn.MSELoss() 
    if args.loss_type == "L1": loss_fn = torch.nn.SmoothL1Loss()

    enc = Encoder
    dec = Decoder

    if args.arch_type == "simple":
        enc = EncoderSimple
        dec = DecoderSimple
    if args.arch_type == "complex":
        enc = EncoderComplex
        dec = DecoderComplex       
    
    ## Get the encoders etc
    encoder=enc(args.nchan, args.latent, act_fn)
    decoder=dec(args.nchan, args.latent, act_fn)
    
    encoder.to(device)
    decoder.to(device)
    
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, total_steps=num_iterations, cycle_momentum=False)
    
    run_training(args.nstep, args.log, encoder, decoder, train_loader, loss_fn, optimizer)#, scheduler)
