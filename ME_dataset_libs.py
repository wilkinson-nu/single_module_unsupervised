import torch
import MinkowskiEngine as ME
import os
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F
from scipy.ndimage import gaussian_filter, map_coordinates
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

class MaxNonZeroCrop:
    def __init__(self):
        self.orig_y = 280
        self.orig_x = 140
        self.new_y = 256
        self.new_x = 128

    def __call__(self, coords, feats):
        """
        Crop to the region with the most non-zero values.
        """
        # Create a 2D histogram to count nonzeros in each pixel
        hist, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1],
                                              bins=(self.orig_y, self.orig_x),
                                              range=[[0, self.orig_y], [0, self.orig_x]])
        # Compute the sliding window sum
        max_count = 0
        best_start_y = 0
        best_start_x = 0

        for start_y in range(self.orig_y - self.new_y + 1):
            for start_x in range(self.orig_x - self.new_x + 1):
                window_sum = np.sum(hist[start_y:start_y + self.new_y, start_x:start_x + self.new_x])
                if window_sum > max_count:
                    max_count = window_sum
                    best_start_y = start_y
                    best_start_x = start_x

        # Define the selected crop area
        crop_min_y = best_start_y
        crop_max_y = best_start_y + self.new_y
        crop_min_x = best_start_x
        crop_max_x = best_start_x + self.new_x

        # Apply the crop mask to sparse coordinates
        mask = (coords[:, 0] >= crop_min_y) & (coords[:, 0] < crop_max_y) & \
               (coords[:, 1] >= crop_min_x) & (coords[:, 1] < crop_max_x)

        # Adjust coordinates to the cropped region
        cropped_coords = coords[mask] - np.array([crop_min_y, crop_min_x])
        cropped_feats = feats[mask]

        return cropped_coords, cropped_feats

class MaxRegionCrop:
    def __init__(self):
        self.orig_y = 280
        self.orig_x = 140
        self.new_y = 256
        self.new_x = 128

        # Predefined regions (center, corners)
        self.regions = {
            "center": ((self.orig_y - self.new_y) // 2, (self.orig_x - self.new_x) // 2),
            "top_left": (0, 0),
            "top_right": (0, self.orig_x - self.new_x),
            "bottom_left": (self.orig_y - self.new_y, 0),
            "bottom_right": (self.orig_y - self.new_y, self.orig_x - self.new_x),
        }

    def __call__(self, coords, feats):
        """
        Crop the sparse coordinates and features to the region with the most non-zero hits.
        """
        max_count = 0
        best_region = None

        # Iterate through predefined regions
        for region, (start_y, start_x) in self.regions.items():
            crop_min_y, crop_max_y = start_y, start_y + self.new_y
            crop_min_x, crop_max_x = start_x, start_x + self.new_x

            # Mask for the region
            mask = (coords[:, 0] >= crop_min_y) & (coords[:, 0] < crop_max_y) & \
                   (coords[:, 1] >= crop_min_x) & (coords[:, 1] < crop_max_x)

            # Count the number of non-zero elements in this region
            count = np.sum(mask)

            # Update the best region if this one has more hits
            if count > max_count:
                max_count = count
                best_region = (start_y, start_x)

        # Apply the best region crop
        crop_min_y, crop_max_y = best_region[0], best_region[0] + self.new_y
        crop_min_x, crop_max_x = best_region[1], best_region[1] + self.new_x

        # Mask for the best region
        mask = (coords[:, 0] >= crop_min_y) & (coords[:, 0] < crop_max_y) & \
               (coords[:, 1] >= crop_min_x) & (coords[:, 1] < crop_max_x)

        # Adjust the coordinates and filter the features
        cropped_coords = coords[mask] - np.array([crop_min_y, crop_min_x])
        cropped_feats = feats[mask]

        return cropped_coords, cropped_feats

## This just takes a 256x128 subimage from the transformed block
class RandomCrop:
    def __init__(self, clip=5):
        self.new_y = 256
        self.new_x = 128  
        self.clip = clip

    def __call__(self, coords, feats):
        new_feats = feats.copy()
        y_round, x_round = np.round(coords[:, 0]).astype(int), np.round(coords[:, 1]).astype(int)
        new_coords = np.stack([y_round, x_round], axis=-1)
                
        y_max = np.max(y_round)
        y_min = np.min(y_round)
        x_max = np.max(x_round)
        x_min = np.min(x_round)
        
        shift_x, shift_y = 0, 0
        
        if x_max - x_min >= self.new_x:
            ## If the transformed image is wider than the cropped image, ensure it's "through-going"
            shift_x = random.randint(self.new_x-x_max, -1*x_min)
        else: 
            ## If not, randomly place it in the image, with a small region "clipped"
            shift_x = random.randint(-x_min -self.clip, self.new_x-x_max + self.clip)
            
        if y_max - y_min >= self.new_y:
            shift_y = random.randint(self.new_y-y_max, -1*y_min)
        else: 
            shift_y = random.randint(-y_min -self.clip, self.new_y-y_max + self.clip)

        new_coords = new_coords + np.array([shift_y, shift_x])        
        mask = (new_coords[:,0] > 0) & (new_coords[:,0] < (self.new_y)) \
             & (new_coords[:,1] > 0) & (new_coords[:,1] < (self.new_x))
                
        return new_coords[mask], new_feats[mask]


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        ## This is the probability to flip the image
        self.p = p

    def __call__(self, coords, feats):
        new_coords = np.round(coords).astype(np.int32)

        if torch.rand(1) < self.p:
            min_col = new_coords[:,1].min()
            max_col = new_coords[:,1].max()
            new_coords[:,1] = max_col - (new_coords[:,1] - min_col)
        return new_coords,feats

## Need to define a fairly standard functions that work for ME tensors
class RandomRotation2D:
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def _M(self, theta):
        # Generate a 2D rotation matrix for a given angle theta
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
	])

    def __call__(self, coords, feats):
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

## Updated function to remove random blocks
class RandomBlockZeroImproved:
    def __init__(self, nblocks=[0,6], block_range=[0,10], xrange=[0,140], yrange=[0,280]):
        self.nblocks = nblocks
        self.rblocks = block_range
        self.xrange = xrange
        self.yrange = yrange

    def __call__(self, coords, feats):

        coords = np.round(coords).astype(np.int32)
        combined_mask = np.full(feats.size, True, dtype=bool)

        # Dynamically determine extent
        y_min, y_max = min(self.yrange[0], coords[:, 0].min()), max(self.yrange[1], coords[:, 0].max())
        x_min, x_max = min(self.xrange[0], coords[:, 1].min()), max(self.xrange[1], coords[:, 1].max())
        
        num_blocks_removed = random.randint(*self.nblocks)
        for _ in range(num_blocks_removed):
            
            block_size = random.randint(*self.rblocks)
            block_x = random.randint(x_min, x_max - block_size)
            block_y = random.randint(y_min, y_max - block_size)

            mask = ~((coords[:,0] > block_y) & (coords[:,0] < (block_y+block_size)) \
                     & (coords[:,1] > block_x) & (coords[:,1] < (block_x+block_size)))
            combined_mask &= mask
            
        return coords[combined_mask].copy(), feats[combined_mask].copy()
        
    
## Apply a Gaussian jitter to all values
class RandomJitterCharge:
    def __init__(self, width=0.1):
        self.width = width

    def __call__(self,  coords, feats):
        scale_factors = np.random.normal(loc=1.0, scale=self.width, size=feats.shape)
        new_feats = feats*scale_factors
        return coords, new_feats
    
# Scale the entire feature vector by a single scaling factor
class RandomScaleCharge:
    def __init__(self, width=0.1):
        self.width = width

    def __call__(self,  coords, feats):
        scale_factor = np.random.normal(loc=1.0, scale=self.width)
        new_feats = feats*scale_factor
        return coords, new_feats

## This is just used to check the performance when the charge scale is artificially removed
class ConstantCharge:
    
    def __init__(self, value=1.0):
        self.value = value

    def __call__(self,  coords, feats):
        new_feats = np.ones_like(feats)
        return coords, new_feats
    
    
class RandomElasticDistortion2D:
    def __init__(self, alpha_range, sigma):
        self.alpha_range = alpha_range
        self.sigma = sigma
        self.height = 280
        self.width = 140
        
    def __call__(self, coords, feats):
        """       
       # Arguments
       image: Numpy array with shape (height, width, channels). 
       alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
           Controls intensity of deformation.
       sigma: Float, sigma of gaussian filter that smooths the displacement fields.
       random_state: `numpy.random.RandomState` object for generating displacement fields.
        """
    
        alpha = np.random.uniform(low=self.alpha_range[0], high=self.alpha_range[1])

        # Create random displacement fields
        displacement_shape = (self.height, self.width)
        dx = gaussian_filter((np.random.rand(*displacement_shape) * 2 - 1), self.sigma) * alpha
        dy = gaussian_filter((np.random.rand(*displacement_shape) * 2 - 1), self.sigma) * alpha

        # Normalize coords to the grid size
        norm_x = coords[:, 0] / self.width * (displacement_shape[1] - 1)
        norm_y = coords[:, 1] / self.height * (displacement_shape[0] - 1)

        # Interpolate displacement fields at coordinate positions
        distorted_x = norm_x + map_coordinates(dx, [norm_y, norm_x], order=1, mode='reflect')
        distorted_y = norm_y + map_coordinates(dy, [norm_y, norm_x], order=1, mode='reflect')

        # Denormalize back to original coordinate scale
        new_coords = np.stack((distorted_x * self.width / (displacement_shape[1] - 1),
                            distorted_y * self.height / (displacement_shape[0] - 1)), axis=-1)
        return new_coords, feats

    
## Apply distortions in a regular grid, with random strength at each point up to some maximum, smoothed by some amount
class RandomGridDistortion2D:
    def __init__(self, grid_size, distortion_strength):
        """
        Initializes the GridDistortion2D transformation.

        :param grid_size: Size of the grid (number of grid points along each axis).
        :param distortion_strength: Maximum displacement for the grid points.
        """
        self.grid_size = grid_size
        self.distortion_strength = distortion_strength
        self.height = 280
        self.width = 140

    def __call__(self, coords, feats):

        # Create a grid of points
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, self.width, self.grid_size),
            np.linspace(0, self.height, self.grid_size)
        )
                     
        # Create random displacements for the grid points
        displacement_x = np.random.uniform(-self.distortion_strength, self.distortion_strength, grid_x.shape)
        displacement_y = np.random.uniform(-self.distortion_strength, self.distortion_strength, grid_y.shape)
        
        # Apply the displacements to the grid points
        distorted_map_x = (grid_x + displacement_x)/ self.width * (self.grid_size - 1)
        distorted_map_y = (grid_y + displacement_y)/ self.height * (self.grid_size - 1)
        
        # Normalize coords to the grid size
        norm_y = coords[:, 0] / self.height * (self.grid_size - 1)
        norm_x = coords[:, 1] / self.width * (self.grid_size - 1)
                
        # Interpolate the distorted coordinates using map_coordinates
        distorted_x = map_coordinates(distorted_map_x, [norm_y, norm_x], order=1, mode='reflect')
        distorted_y = map_coordinates(distorted_map_y, [norm_y, norm_x], order=1, mode='reflect')
        
        # Combine distorted coordinates
        distorted_coords = np.stack((distorted_y * self.height / (self.grid_size - 1),
                                    distorted_x * self.width / (self.grid_size - 1)), axis=-1)
        
        return distorted_coords, feats
    
class BilinearInterpolation:
    def __init__(self, threshold=0.04):
        self.height=280
        self.width=140
        self.threshold=threshold
        
    def __call__(self, coords, feats):
        """
        Apply bilinear interpolation to sparse image data represented by coordinates and features.
    
        Arguments:
        coords: Numpy array of shape (N, 2), where each row is (x, y) coordinate.
        feats: Numpy array of shape (N,), containing feature values for each coordinate.
        height: Integer, maximum height of the output grid.
        width: Integer, maximum width of the output grid.

        Returns:
        interpolated_coords: Numpy array of shape (M, 2), with interpolated integer coordinates.
        interpolated_feats: Numpy array of shape (M,), containing interpolated feature values.
        """
        
        feats = np.squeeze(feats)  # Remove single-dimensional entries from shape
        
        # Floor and ceil coordinates for each point
        x0, y0 = np.floor(coords[:, 0]).astype(int), np.floor(coords[:, 1]).astype(int)
        x1, y1 = np.ceil(coords[:, 0]).astype(int), np.ceil(coords[:, 1]).astype(int)
    
        # Calculate the weights for bilinear interpolation
        wx1 = coords[:, 0] - x0
        wx0 = 1 - wx1
        wy1 = coords[:, 1] - y0
        wy0 = 1 - wy1
    
        # Coordinates for the four corners
        coords00 = np.stack([x0, y0], axis=-1)
        coords01 = np.stack([x0, y1], axis=-1)
        coords10 = np.stack([x1, y0], axis=-1)
        coords11 = np.stack([x1, y1], axis=-1)
    
        # Calculate interpolated feature values for each of the four corners
        f00 = feats * (wx0 * wy0)
        f01 = feats * (wx0 * wy1)
        f10 = feats * (wx1 * wy0)
        f11 = feats * (wx1 * wy1)
    
        # Combine coordinates and features
        coords_combined = np.vstack([coords00,coords01,coords10,coords11])
        features_combined = np.concatenate([f00, f01, f10, f11])
    
        # Round coordinates to nearest integers and clip them
        coords_combined = np.round(coords_combined).astype(int)
        # coords_combined = np.clip(coords_combined, [0, 0], [self.height-1, self.width-1])

        ## The clipping is wrong...
        mask = (coords_combined[:,0] > 0) \
             & (coords_combined[:,0] < (self.height-1)) \
             & (coords_combined[:,1] > 0) \
             & (coords_combined[:,1] < (self.width-1))
        coords_combined = coords_combined[mask]
        features_combined = features_combined[mask]
        
        # Consolidate features at unique coordinates
        unique_coords, indices = np.unique(coords_combined, axis=0, return_inverse=True)
        summed_feats = np.zeros(len(unique_coords))    
        np.add.at(summed_feats, indices, features_combined)

        # Create a mask for values above the threshold
        mask = summed_feats >= self.threshold
    
        # Apply the mask to filter features and coordinates
        unique_coords = unique_coords[mask]
        summed_feats = summed_feats[mask]        
        
        # Reshape summed_feats to (N, 1)
        summed_feats = summed_feats.reshape(-1, 1)
        
        return unique_coords, summed_feats
    

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


def cat_ME_collate_fn(batch):
    aug1_coords, aug1_feats, aug2_coords, aug2_feats, raw_coords, raw_feats = zip(*batch)

    # Create batched coordinates for the SparseTensor input
    cat_bcoords = ME.utils.batched_coordinates(aug1_coords+aug2_coords)

    # Concatenate all lists
    cat_bfeats = torch.from_numpy(np.concatenate(aug1_feats+aug2_feats, 0)).float()

    return cat_bcoords, cat_bfeats


class SingleModuleImage2D_solo_ME(Dataset):

    def __init__(self, infile_dir, transform, max_events=None, return_metadata=False):
        self.hdf5_files = sorted(glob(os.path.join(infile_dir, '*.h5')))
        self.file_indices = []
        self.transform = transform
        self.max_events = max_events
        self.return_metadata = return_metadata
        
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

        if self.return_metadata:
            event_id = group.attrs.get("event_id", this_idx)
            filename = os.path.basename(self.hdf5_files[file_index])
            return coords, feats, filename, event_id        
        return coords, feats
    
def solo_ME_collate_fn(batch):
    coords, feats = zip(*batch)
    
    # Create batched coordinates for the SparseTensor input
    bcoords  = ME.utils.batched_coordinates(coords)
    
    # Concatenate all lists
    bfeats  = torch.from_numpy(np.concatenate(feats, 0)).float()
    
    return bcoords, bfeats


def solo_ME_collate_fn_with_meta(batch):
    coords, feats, filenames, event_ids = zip(*batch)
    bcoords = ME.utils.batched_coordinates(coords)
    bfeats = torch.from_numpy(np.concatenate(feats, 0)).float()
    return bcoords, bfeats, filenames, event_ids


## Utility functions to make a dense image
def make_dense(coords_batch, feats_batch, device, index=0, max_i=256, max_j=128):
    img = ME.SparseTensor(feats_batch.float(), coords_batch.int(), device=device)
    coords, feats = img.decomposed_coordinates_and_features
    batch_size = len(coords)
    img_dense,_,_ = img.dense(torch.Size([batch_size, 1, max_i, max_j]))
    return img_dense[index].squeeze().numpy()

def make_dense_from_tensor(sparse_batch, index=0, max_i=256, max_j=128):
    coords, feats = sparse_batch.decomposed_coordinates_and_features
    batch_size = len(coords)
    img_dense,_,_ = sparse_batch.dense(torch.Size([batch_size, 1, max_i, max_j]), min_coordinate=torch.IntTensor([0,0]))
    return img_dense[index]

def make_dense_array(coords, feats, max_i=256, max_j=128):
    img_dense = np.zeros((max_i, max_j))
    i_coords, j_coords = coords[:, 0], coords[:, 1]
    img_dense[i_coords, j_coords] = feats
    return img_dense



## Just a big function to return a set of transforms, to be returned by name
def get_transform(aug_type=None):

    if aug_type == "block10x10":
        return transforms.Compose([
            RandomGridDistortion2D(5,5),
            RandomShear2D(0.1, 0.1),
	    RandomHorizontalFlip(),
	    RandomRotation2D(-10,10),
	    RandomBlockZeroImproved([0,10], [5,10]),
            RandomScaleCharge(0.02),
            RandomJitterCharge(0.02),
	    RandomCrop()
        ])
    if aug_type == "bigmodblock10x10":
        return transforms.Compose([
	    RandomGridDistortion2D(5,5),
            RandomShear2D(0.2, 0.2),
            RandomHorizontalFlip(),
            RandomRotation2D(-30,30),
            RandomBlockZeroImproved([0,10], [5,10]),
            RandomScaleCharge(0.1),
            RandomJitterCharge(0.1),
            RandomCrop()
        ])
    if aug_type == "unitcharge":
        return transforms.Compose([
            RandomGridDistortion2D(5,5),
            RandomShear2D(0.1, 0.1),
            RandomHorizontalFlip(),
            RandomRotation2D(-10,10),
            RandomBlockZero(5, 6),
            RandomCrop(),
            ConstantCharge()
        ])
    if aug_type == "bigmod":
        return transforms.Compose([
            RandomGridDistortion2D(5,5),
            RandomShear2D(0.2, 0.2),
            RandomHorizontalFlip(),
            RandomRotation2D(-30,30),
            RandomBlockZero(5, 6),
            RandomScaleCharge(0.1),
            RandomJitterCharge(0.1),
            RandomCrop()
        ])
    if aug_type == "bigbilin":
        return transforms.Compose([
            RandomGridDistortion2D(5,5),
            RandomShear2D(0.2, 0.2),
            RandomHorizontalFlip(),
            RandomRotation2D(-30,30),
            RandomBlockZero(5, 6),
            BilinearInterpolation(0.05),
            RandomScaleCharge(0.1),
            RandomJitterCharge(0.1),
            RandomCrop()
        ])

    ## If not, return the default
    return transforms.Compose([
	RandomGridDistortion2D(5,5),
	RandomShear2D(0.1, 0.1),
	RandomHorizontalFlip(),
	RandomRotation2D(-10,10),
	RandomBlockZero(5, 6),
	RandomScaleCharge(0.02),
	RandomJitterCharge(0.02),
	RandomCrop()
    ])

    
