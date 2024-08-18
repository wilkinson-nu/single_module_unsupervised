import torch
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


class SingleModuleImage2D_solo_ME(Dataset):

    def __init__(self, infile_dir, transform, max_events=None):
        self.hdf5_files = sorted(glob(os.path.join(infile_dir, '*.h5')))
        self.file_indices = []
        self.transform = transform
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
        coords = np.vstack((row, col)).T #.copy()
        feats = data.reshape(-1, 1)  # Reshape data to be of shape (N, 1)            
        coords, feats = self.transform(coords, feats)

        return coords, feats
    
def solo_ME_collate_fn(batch):
    coords, feats = zip(*batch)
    
    # Create batched coordinates for the SparseTensor input
    bcoords  = ME.utils.batched_coordinates(coords)
    
    # Concatenate all lists
    bfeats  = torch.from_numpy(np.concatenate(feats, 0)).float()
    
    return bcoords, bfeats

