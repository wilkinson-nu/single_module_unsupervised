from torch import nn
import torch
import sys
import MinkowskiEngine as ME
import os
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F
import h5py
from glob import glob
from bisect import bisect
import random
import numpy as np

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

    def __call__(self, coords, feats):

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


## Calculate the average distance between pairs in the latent space
class EuclideanDistLoss(torch.nn.Module):
    def __init__(self):
        super(EuclideanDistLoss, self).__init__()
        
    def forward(self, latent1, latent2):
        # Compute the Euclidean distance between each pair of corresponding tensors in the batch
        norm_lat1 = nn.functional.normalize(latent1, p=2, dim=1)
        norm_lat2 = nn.functional.normalize(latent2, p=2, dim=1)
        distances = torch.norm(norm_lat1 - norm_lat2, p=2, dim=1)

        mod_penalty = torch.stack([item**2 for item in distances])
        loss = mod_penalty.mean()
        return loss
    
class CosDistLoss(torch.nn.Module):
    def __init__(self):
        super(CosDistLoss, self).__init__()
        
    def forward(self, latent1, latent2):
        norm_lat1 = nn.functional.normalize(latent1, p=2, dim=1)
        norm_lat2 = nn.functional.normalize(latent2, p=2, dim=1)
                
        sim = nn.functional.cosine_similarity(norm_lat1, norm_lat2, dim=1)
        loss = 1 - sim.mean()
        return loss

class NTXent(torch.nn.Module):
    def __init__(self, batch_size=512, temperature=0.5):
        super(NTXent, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
    
    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
    def forward(self, latent1, latent2):
        batch_size = latent1.shape[0]
        z_i = nn.functional.normalize(latent1, p=2, dim=1)
        z_j = nn.functional.normalize(latent2, p=2, dim=1)
        
        similarity_matrix = self.calc_similarity_batch(z_i, z_j)
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(device)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = mask*torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss

    
## This is a loss function to deweight the penalty for getting blank pixels wrong
class AsymmetricL2LossME(torch.nn.Module):
    def __init__(self, nonzero_cost=2.0, zero_cost=1.0):
        super(AsymmetricL2LossME, self).__init__()
        self.nonzero_cost = nonzero_cost
        self.zero_cost = zero_cost
    
    def forward(self, pred, targ):
        #diff = pred - targ
        #loss = torch.sum(diff.F**2)
        #return loss/512/128/256
        
        # Extract coordinates and features from both sparse tensors
        pred_C = pred.C
        pred_F = pred.F
    
        targ_C = targ.C
        targ_F = targ.F
        
        _, idx, counts = torch.cat([pred_C, targ_C], dim=0).unique(dim=0, return_inverse=True, return_counts=True)
        mask = torch.isin(idx, torch.where(counts.gt(1))[0])
        
        ## This is a safe alternative
        gt_mask = counts.gt(1)
        mask = gt_mask[idx]
        pred_mask = mask[:pred_C.size(0)]
        targ_mask = mask[pred_C.size(0):]
               
        indices_pred = torch.arange(pred_F.size(0), device='cuda')[pred_mask]
        indices_targ = torch.arange(targ_F.size(0), device='cuda')[targ_mask]
        
        common_pred_F = pred_F.index_select(0, indices_pred)
        common_targ_F = targ_F.index_select(0, indices_targ)
        
        ## These cause blocking synchronization calls
        common_pred_F = pred_F[pred_mask]
        common_targ_F = targ_F[targ_mask]
        uncommon_pred_F = pred_F[~pred_mask]
        uncommon_targ_F = targ_F[~targ_mask]
       
        common = torch.sum(self.nonzero_cost*(common_pred_F - common_targ_F)**2)
        only_p = torch.sum(self.zero_cost*(uncommon_pred_F**2))
        only_t = torch.sum(self.nonzero_cost*(uncommon_targ_F**2))
        
        return (common+only_p+only_t)/(512*128*256)


## These are the original encoders
class EncoderME(nn.Module):
    
    def __init__(self, 
                 nchan : int,
                 latent_dim : int,
                 act_fn : object = ME.MinkowskiReLU,
                 drop_fract : float = 0.2):
        """
        Inputs:
            - nchan : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        
        self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32, nchan*64, nchan*128]
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=self.ch[0], kernel_size=(2,1), stride=(2,1), bias=False, dimension=2), ## 256x128 ==> 128x128
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[1], kernel_size=2, stride=2, bias=False, dimension=2), ## 128x128 ==> 64x64
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[2], kernel_size=2, stride=2, bias=False, dimension=2), ## 64x64 ==> 32x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[3], kernel_size=2, stride=2, bias=False, dimension=2), ## 32x32 ==> 16x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[4], kernel_size=2, stride=2, bias=False, dimension=2), ## 16x16 ==> 8x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[5], kernel_size=2, stride=2, bias=False, dimension=2), ## 8x8 ==> 4x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[6], kernel_size=2, stride=2, bias=False, dimension=2), ## 4x4 ==> 2x2
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[7], kernel_size=2, stride=2, bias=False, dimension=2), ## 2x2 ==> 1x1
            ME.MinkowskiBatchNorm(self.ch[7]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
        )
        
        ### Linear section, simple for now
        self.encoder_lin = nn.Sequential(
            ME.MinkowskiLinear(self.ch[7], self.ch[4]),
            act_fn(),      
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiLinear(self.ch[4], latent_dim),
        )
        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution) or \
               isinstance(m, ME.MinkowskiGenerativeConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_lin(x)
        return x

    
class DecoderME(nn.Module):
    
    def __init__(self, 
                 nchan : int,
                 latent_dim : int,
                 act_fn : object):
        """
        Inputs:
            - nchan : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        
        self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32, nchan*64, nchan*128]
        
        self.decoder_lin = nn.Sequential(
            ME.MinkowskiLinear(latent_dim, self.ch[4]),
            act_fn(),      
            ME.MinkowskiLinear(self.ch[4], self.ch[7]),
            act_fn()   
        )

        self.decoder_conv = nn.Sequential(  
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[7], out_channels=self.ch[6], kernel_size=2, stride=2, bias=False, dimension=2), ## 1x1 ==> 2x2
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, stride=1, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[6], out_channels=self.ch[5], kernel_size=2, stride=2, bias=False, dimension=2), ## 2x2 ==> 4x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, stride=1, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[5], out_channels=self.ch[4], kernel_size=2, stride=2, bias=False, dimension=2), ## 4x4 ==> 8x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[4], out_channels=self.ch[3], kernel_size=2, stride=2, bias=False, dimension=2), ## 8x8 ==> 16x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[3], out_channels=self.ch[2], kernel_size=2, stride=2, bias=False, dimension=2), ## 16x16 ==> 32x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[2], out_channels=self.ch[1], kernel_size=2, stride=2, bias=False, dimension=2), ## 32x32 ==> 64x64
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[1], out_channels=self.ch[0], kernel_size=2, stride=2, bias=False, dimension=2), ## 64x64 ==> 128x128
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),          
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[0], out_channels=1, kernel_size=(2,1), stride=(2,1), bias=True, dimension=2), ## 128x128 ==> 256x128
            act_fn()
        )
        
        # Initialize weights using Xavier initialization
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution) or \
               isinstance(m, ME.MinkowskiGenerativeConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)
             
    def forward(self, x, target_key=None):
        x = self.decoder_lin(x)
        x = self.decoder_conv(x)
        return x

## Modified for testing
class DeepEncoderME(nn.Module):
    
    def __init__(self, 
                 nchan : int,
                 latent_dim : int,
                 act_fn : object = ME.MinkowskiReLU,
                 drop_fract : float = 0.2):
        """
        Inputs:
            - nchan : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        
        self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32, nchan*64, nchan*128]
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=self.ch[0], kernel_size=(2,1), stride=(2,1), bias=False, dimension=2), ## 256x128 ==> 128x128
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[1], kernel_size=2, stride=2, bias=False, dimension=2), ## 128x128 ==> 64x64
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[2], kernel_size=2, stride=2, bias=False, dimension=2), ## 64x64 ==> 32x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[3], kernel_size=2, stride=2, bias=False, dimension=2), ## 32x32 ==> 16x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[4], kernel_size=2, stride=2, bias=False, dimension=2), ## 16x16 ==> 8x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[5], kernel_size=2, stride=2, bias=False, dimension=2), ## 8x8 ==> 4x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[6], kernel_size=2, stride=2, bias=False, dimension=2), ## 4x4 ==> 2x2
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[7], kernel_size=2, stride=2, bias=False, dimension=2), ## 2x2 ==> 1x1
            ME.MinkowskiBatchNorm(self.ch[7]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
        )
        
        ### Linear section, simple for now
        self.encoder_lin = nn.Sequential(
            ME.MinkowskiLinear(self.ch[7], self.ch[4]),
            act_fn(),      
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiLinear(self.ch[4], self.ch[3]),
            act_fn(),      
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiLinear(self.ch[3], self.ch[3]),
            act_fn(),      
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiLinear(self.ch[3], latent_dim),
        )
        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution) or \
               isinstance(m, ME.MinkowskiGenerativeConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_lin(x)
        return x

    
class DeepDecoderME(nn.Module):
    
    def __init__(self, 
                 nchan : int,
                 latent_dim : int,
                 act_fn : object):
        """
        Inputs:
            - nchan : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        
        self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32, nchan*64, nchan*128]
        
        self.decoder_lin = nn.Sequential(
            ME.MinkowskiLinear(latent_dim, self.ch[4]),
            act_fn(),      
            ME.MinkowskiLinear(self.ch[4], self.ch[7]),
            act_fn()   
        )

        self.decoder_conv = nn.Sequential(  
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[7], out_channels=self.ch[6], kernel_size=2, stride=2, bias=False, dimension=2), ## 1x1 ==> 2x2
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, stride=1, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, stride=1, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[6], out_channels=self.ch[5], kernel_size=2, stride=2, bias=False, dimension=2), ## 2x2 ==> 4x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, stride=1, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, stride=1, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[5], out_channels=self.ch[4], kernel_size=2, stride=2, bias=False, dimension=2), ## 4x4 ==> 8x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[4], out_channels=self.ch[3], kernel_size=2, stride=2, bias=False, dimension=2), ## 8x8 ==> 16x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[3], out_channels=self.ch[2], kernel_size=2, stride=2, bias=False, dimension=2), ## 16x16 ==> 32x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[2], out_channels=self.ch[1], kernel_size=2, stride=2, bias=False, dimension=2), ## 32x32 ==> 64x64
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[1], out_channels=self.ch[0], kernel_size=2, stride=2, bias=False, dimension=2), ## 64x64 ==> 128x128
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[0], out_channels=1, kernel_size=(2,1), stride=(2,1), bias=True, dimension=2), ## 128x128 ==> 256x128
            act_fn()
        )
        
        # Initialize weights using Xavier initialization
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution) or \
               isinstance(m, ME.MinkowskiGenerativeConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)
             
    def forward(self, x, target_key=None):
        x = self.decoder_lin(x)
        x = self.decoder_conv(x)
        return x

    
## Modified for testing
class DeeperEncoderME(nn.Module):
    
    def __init__(self, 
                 nchan : int,
                 latent_dim : int,
                 act_fn : object = ME.MinkowskiReLU,
                 drop_fract : float = 0.2):
        """
        Inputs:
            - nchan : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        
        self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32, nchan*64, nchan*128]
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=self.ch[0], kernel_size=(2,1), stride=(2,1), bias=False, dimension=2), ## 256x128 ==> 128x128
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[1], kernel_size=2, stride=2, bias=False, dimension=2), ## 128x128 ==> 64x64
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[2], kernel_size=2, stride=2, bias=False, dimension=2), ## 64x64 ==> 32x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[3], kernel_size=2, stride=2, bias=False, dimension=2), ## 32x32 ==> 16x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[4], kernel_size=2, stride=2, bias=False, dimension=2), ## 16x16 ==> 8x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[5], kernel_size=2, stride=2, bias=False, dimension=2), ## 8x8 ==> 4x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[6], kernel_size=2, stride=2, bias=False, dimension=2), ## 4x4 ==> 2x2
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[7], kernel_size=2, stride=2, bias=False, dimension=2), ## 2x2 ==> 1x1
            ME.MinkowskiBatchNorm(self.ch[7]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
        )
        
        ### Linear section, simple for now
        self.encoder_lin = nn.Sequential(
            ME.MinkowskiLinear(self.ch[7], self.ch[4]),
            act_fn(),      
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiLinear(self.ch[4], self.ch[3]),
            act_fn(),      
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiLinear(self.ch[3], self.ch[3]),
            act_fn(),      
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiLinear(self.ch[3], latent_dim),
        )
        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution) or \
               isinstance(m, ME.MinkowskiGenerativeConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_lin(x)
        return x

    
class DeeperDecoderME(nn.Module):
    
    def __init__(self, 
                 nchan : int,
                 latent_dim : int,
                 act_fn : object):
        """
        Inputs:
            - nchan : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        
        self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32, nchan*64, nchan*128]
        
        self.decoder_lin = nn.Sequential(
            ME.MinkowskiLinear(latent_dim, self.ch[4]),
            act_fn(),      
            ME.MinkowskiLinear(self.ch[4], self.ch[7]),
            act_fn()   
        )

        self.decoder_conv = nn.Sequential(  
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[7], out_channels=self.ch[6], kernel_size=2, stride=2, bias=False, dimension=2), ## 1x1 ==> 2x2
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, stride=1, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, stride=1, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, stride=1, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            act_fn(),
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[6], out_channels=self.ch[5], kernel_size=2, stride=2, bias=False, dimension=2), ## 2x2 ==> 4x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, stride=1, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, stride=1, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, stride=1, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[5], out_channels=self.ch[4], kernel_size=2, stride=2, bias=False, dimension=2), ## 4x4 ==> 8x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[4], out_channels=self.ch[3], kernel_size=2, stride=2, bias=False, dimension=2), ## 8x8 ==> 16x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[3], out_channels=self.ch[2], kernel_size=2, stride=2, bias=False, dimension=2), ## 16x16 ==> 32x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[2], out_channels=self.ch[1], kernel_size=2, stride=2, bias=False, dimension=2), ## 32x32 ==> 64x64
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(), 
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[1], out_channels=self.ch[0], kernel_size=2, stride=2, bias=False, dimension=2), ## 64x64 ==> 128x128
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),  
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[0], out_channels=1, kernel_size=(2,1), stride=(2,1), bias=True, dimension=2), ## 128x128 ==> 256x128
            act_fn()
        )
        
        # Initialize weights using Xavier initialization
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution) or \
               isinstance(m, ME.MinkowskiGenerativeConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)
             
    def forward(self, x, target_key=None):
        x = self.decoder_lin(x)
        x = self.decoder_conv(x)
        return x
