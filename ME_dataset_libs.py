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
from enum import Enum, auto
from scipy.stats import truncnorm

## Class to define the truth labels, with some helper functions
class Label(Enum):

    ## Default
    DATA = -1
    
    ## e+/- or photon induced
    EM = auto()

    ## Neutron induced
    NEUTRON = auto()

    ## Proton induced
    PROTON = auto()

    ## Charged pion induced
    PION = auto()
    
    ## Multiple muons deposit energy into the active volume
    MULTIMUON = auto()

    ## The muon interacted outside the active volume
    EXTERNAL = auto()

    ## Stopping muon which is captured by a nucleus and then decays not through a Michel
    STOPPINGCAPTURE = auto()

    ## Muon which decays into a Michel inside the volume
    STOPPINGMICHEL = auto()

    ## This is a catch-all category for stopping events that aren't the other two...
    STOPPINGOTHER = auto()
    
    ## This is meant to be for through-going muons without much colinear activity
    THROUGHCLEAN = auto()

    ## This is to try and get at through-going muons with colinear showers
    THROUGHMESSY = auto()

    ## A method to dump the list
    @classmethod
    def print_members(cls):
        for member in cls:
            print(f"{member.name}: {member.value}")

    @classmethod
    def name_from_index(cls, index):
        for member in cls:
            if member.value == index:
                return member.name
        return f"Unknown label for index {index}"
        

## This is a transformation for the nominal image
class CenterCrop:
    def __init__(self, orig_size=(280, 140), new_size=(256, 128)):
        self.orig_y = orig_size[0]
        self.orig_x = orig_size[1]
        self.new_y = new_size[0]
        self.new_x = new_size[1]
        self.pad_y = (self.orig_y - self.new_y)/2
        self.pad_x = (self.orig_x - self.new_x)/2

    def __call__(self, coords, feats):

        coords = coords - np.array([self.pad_y, self.pad_x])
        mask = (coords[:,0] > 0) & (coords[:,0] < (self.new_y)) \
             & (coords[:,1] > 0) & (coords[:,1] < (self.new_x))

        return coords[mask], feats[mask]

class MaxNonZeroCrop:
    def __init__(self, orig_size=(280, 140), new_size=(256, 128)):
        self.orig_y = orig_size[0]
        self.orig_x = orig_size[1]
        self.new_y = new_size[0]
        self.new_x = new_size[1]

    def __call__(self, coords, feats):
        # Create a 2D histogram to count nonzeros in each pixel
        hist, xedges, yedges = np.histogram2d(coords[:, 1], coords[:, 0],
                                              bins=(self.orig_x, self.orig_y),
                                              range=[[0, self.orig_x], [0, self.orig_y]])
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
    def __init__(self, orig_size=(280, 140), new_size=(256, 128)):
        self.orig_y = orig_size[0]
        self.orig_x = orig_size[1]
        self.new_y = new_size[0]
        self.new_x = new_size[1]

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

class FirstRegionCrop:
    def __init__(self, orig_size=(280, 140), new_size=(256, 128)):
        self.orig_y = orig_size[0]
        self.orig_x = orig_size[1]
        self.new_y = new_size[0]
        self.new_x = new_size[1]

        # Predefined regions (center, corners)
        self.regions = {
            "center": ((self.orig_y - self.new_y) // 2, (self.orig_x - self.new_x) // 2),
            "top_left": (0, 0),
            "top_right": (0, self.orig_x - self.new_x),
            "bottom_left": (self.orig_y - self.new_y, 0),
            "bottom_right": (self.orig_y - self.new_y, self.orig_x - self.new_x),
        }

    def __call__(self, coords, feats):

        use_region = None

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
            if count > 0: use_region = (start_y, start_x)

        # Apply the best region crop
        crop_min_y, crop_max_y = use_region[0], use_region[0] + self.new_y
        crop_min_x, crop_max_x = use_region[1], use_region[1] + self.new_x

        # Mask for the best region
        mask = (coords[:, 0] >= crop_min_y) & (coords[:, 0] < crop_max_y) & \
               (coords[:, 1] >= crop_min_x) & (coords[:, 1] < crop_max_x)

        # Adjust the coordinates and filter the features
        cropped_coords = coords[mask] - np.array([crop_min_y, crop_min_x])
        cropped_feats = feats[mask]

        return cropped_coords, cropped_feats


## This just takes a transformed image from the transformed block
class RandomCrop:
    def __init__(self, new_x, new_y, clip=10):
        self.new_y = new_y
        self.new_x = new_x
        self.clip = clip

    def __call__(self, coords, feats):
        new_feats = feats.copy()
        y_round = np.round(coords[:, 0]).astype(np.int32)
        x_round = np.round(coords[:, 1]).astype(np.int32)
        new_coords = np.stack([y_round, x_round], axis=-1)
        
        y_max = np.maximum(np.max(y_round), self.new_y)
        y_min = np.minimum(np.min(y_round), 0)
        x_max = np.maximum(np.max(x_round), self.new_x)
        x_min = np.minimum(np.min(x_round), 0)

        shift_x = random.randint(x_min-self.clip, (x_max+self.clip) - self.new_x)
        shift_y = random.randint(y_min-self.clip, (y_max+self.clip) - self.new_y)

        new_coords = new_coords + np.array([shift_y, shift_x])        
        mask = (new_coords[:,0] > 0) & (new_coords[:,0] < (self.new_y)) \
             & (new_coords[:,1] > 0) & (new_coords[:,1] < (self.new_x))
                
        return new_coords[mask], new_feats[mask]

class SimpleCrop:
    def __init__(self, max_y, max_x):
        self.max_y = max_y
        self.max_x = max_x

    def __call__(self, coords, feats):
        new_feats = feats.copy()
        y_round = np.round(coords[:, 0]).astype(np.int32)
        x_round = np.round(coords[:, 1]).astype(np.int32)
        new_coords = np.stack([y_round, x_round], axis=-1)

        mask = (new_coords[:,0] > 0) & (new_coords[:,0] < (self.max_y)) \
	     & (new_coords[:,1] > 0) & (new_coords[:,1] < (self.max_x))

        return new_coords[mask], new_feats[mask]

    
    
class RandomInPlaceHorizontalFlip:
    def __init__(self, p=0.5):
        ## This is the probability to flip the image
        self.p = p

    def __call__(self, coords, feats):
        new_coords = coords.copy()
        
        if np.random.rand() < self.p:
            min_h = new_coords[:,1].min()
            max_h = new_coords[:,1].max()
            new_coords[:,1] = max_h - (new_coords[:,1] - min_h)

        return new_coords,feats

class RandomInPlaceVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, coords, feats):
        new_coords = coords.copy()

        if np.random.rand() < self.p:
            min_v = new_coords[:,0].min()
            max_v = new_coords[:,0].max()
            new_coords[:,0] = max_v - (new_coords[:,0] - min_v)

        return new_coords,feats

class RandomHorizontalFlip:
    def __init__(self, p=0.5, x_max=256):
        self.p = p
        self.x_max = x_max

    def __call__(self, coords, feats):
        new_coords = coords.copy()

        if np.random.rand() < self.p:
            new_coords[:,1] = self.x_max - new_coords[:,1]

        return new_coords,feats

    
class RandomVerticalFlip:
    def __init__(self, p=0.5, y_max=512):
        self.p = p
        self.y_max = y_max

    def __call__(self, coords, feats):
        new_coords = coords.copy()
        
        if np.random.rand() < self.p:
            new_coords[:,0] = self.y_max - new_coords[:,0]

        return new_coords,feats

    
    
## Need to define a fairly standard functions that work for ME tensors
class RandomRotation2D:
    def __init__(self, angle, max_y=None, max_x=None):
        self.angle = angle
        self.max_y = max_y
        self.max_x = max_x

    def _M(self, theta):
        # Generate a 2D rotation matrix for a given angle theta
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
	])

    def __call__(self, coords, feats):
    	# Generate a random rotation angle
        angle = np.deg2rad(np.random.normal(loc=0, scale=self.angle))
        fcoords = coords.astype(float)
        
        ## Keep old default behaviour
        if self.max_y is not None and self.max_x is not None:
            # Pick a random central-ish point
            cx = np.random.uniform(self.max_x/4, 3*self.max_x/4)
            cy = np.random.uniform(self.max_y/4, 3*self.max_y/4)
            center = np.array([cy, cx])
        ## Add a new option to take a point within the image
        else:
            # centroid = coords.mean(axis=0)
            # std = coords.std(axis=0)
            # center = centroid + truncnorm.rvs(-2, 2, loc=0, scale=std)
            temp_idx = np.random.randint(len(fcoords))
            center = fcoords[temp_idx]

        # Get the 2D rotation matrix
        R = self._M(angle)
        
        # Shift and apply the rotation
        shifted = fcoords - center
        rotated = shifted @ R
        rotated_coords = rotated + center        
        return rotated_coords, feats


class RandomShear2D:
    def __init__(self, sigma_y, sigma_x, max_y=None, max_x=None):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.max_x = max_x
        self.max_y = max_y

    def __call__(self, coords, feats):
        fcoords = coords.astype(float)

        shear_x = np.random.normal(loc=0, scale=self.sigma_x)
        shear_y = np.random.normal(loc=0, scale=self.sigma_y)

        shear_matrix = np.array([
            [1, shear_x],
            [shear_y, 1]
        ])
        
        ## Keep old default behaviour
        if self.max_y is not None and self.max_x is not None:
            # Pick a random central-ish point
            cx = np.random.uniform(self.max_x/4, 3*self.max_x/4)
            cy = np.random.uniform(self.max_y/4, 3*self.max_y/4)
            center = np.array([cy, cx])
        ## Add a new option to take a point within the image
        else:
            #centroid = coords.mean(axis=0)
            #std = coords.std(axis=0)
            #center = centroid + truncnorm.rvs(-2, 2, loc=0, scale=std)
            temp_idx = np.random.randint(len(fcoords))
            center = fcoords[temp_idx]
            
        shifted = fcoords - center
        rotated = shifted @ shear_matrix
        rotated_coords = rotated + center
        return rotated_coords, feats

class RandomPixelNoise2D:
    def __init__(self, poisson_mean=10, image_bounds=(0, 256, 0, 512)):
        self.poisson_mean = poisson_mean
        self.noise_value = 1
        self.image_bounds = image_bounds

    def __call__(self, coords, feats):
        coords = np.round(coords).astype(np.int32)

        # Determine bounds from input data or fixed override
        x_min, x_max, y_min, y_max = self.image_bounds

        # Sample number of noise pixels to add
        n_noise = np.random.poisson(self.poisson_mean)
        if n_noise == 0:
            return coords, feats

        # Generate unique random (y, x) pairs inside the image bounds
        y_noise = np.random.randint(y_min, y_max, size=n_noise)
        x_noise = np.random.randint(x_min, x_max, size=n_noise)
        noise_coords = np.stack([y_noise, x_noise], axis=1)

        # If this is applied when constant charge isn't used, will have to be careful about overwriting real pixels with noise

        # Generate corresponding features
        noise_feats = np.full((len(noise_coords), feats.shape[1]), self.noise_value, dtype=feats.dtype)

        # Append to input
        new_coords = np.concatenate([coords, noise_coords], axis=0)
        new_feats = np.concatenate([feats, noise_feats], axis=0)

        return new_coords, new_feats


## A function to randomly remove some number of blocks of size
## This has to be called before the cropping as it uses the original image size
## Updated function to remove random blocks
## The x and y ranges can be extended if the other augmentations have extended the image
## These set limits prevent a small image from being entirely blocked out
## A cheaper alternative might just be to drop some fraction of the hits?
class RandomBlockZeroImproved:
    def __init__(self, nblocks=[0,6], block_range=[0,10], xrange=[0,140], yrange=[0,280]):
        self.nblocks = nblocks
        self.rblocks = block_range
        self.xrange = xrange
        self.yrange = yrange

    def __call__(self, coords, feats):

        # Dynamically determine extent
        y_min = min(self.yrange[0], coords[:, 0].min())
        y_max = max(self.yrange[1], coords[:, 0].max())
        x_min = min(self.xrange[0], coords[:, 1].min())
        x_max = max(self.xrange[1], coords[:, 1].max())

        num_blocks_removed = random.randint(*self.nblocks)
        if num_blocks_removed == 0:
            return coords.copy(), feats.copy()

        # Sample all blocks at once
        block_sizes = np.random.randint(self.rblocks[0], self.rblocks[1]+1, size=num_blocks_removed)
        block_xs = np.random.randint(x_min, x_max - block_sizes, size=num_blocks_removed)
        block_ys = np.random.randint(y_min, y_max - block_sizes, size=num_blocks_removed)

        # For each point, check whether it lies inside any block
        cx, cy = coords[:,1][:,None], coords[:,0][:,None]  # shape (N,1)
        inside_x = (cx > block_xs) & (cx < (block_xs + block_sizes))
        inside_y = (cy > block_ys) & (cy < (block_ys + block_sizes))
        inside_any = np.any(inside_x & inside_y, axis=1)

        keep_mask = ~inside_any
        return coords[keep_mask].copy(), feats[keep_mask].copy()        

class RandomDropout:
    def __init__(self, max_frac=0.1):
        self.max_frac = max_frac

    def __call__(self, coords, feats):
        N = coords.shape[0]
        if N == 0:
            # Return copies to avoid modifying original
            return coords.copy(), feats.copy()

        # Determine how many points to keep
        frac = np.random.uniform(0.0, self.max_frac)
        k = int(N * (1.0 - frac))

        # Randomly select indices to keep
        idx = np.random.choice(N, size=k, replace=False)

        # Return copies of selected coords and feats
        return coords[idx].copy(), feats[idx].copy()
    
    
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
## Cell size is the size of the distortion grid (in pixels, assumed square)
## Distortion strength is the same
class RandomGridDistortion2D:
    def __init__(self, cell_size=50, distortion=5, padding_cells=2):
        self.cell_size = cell_size
        self.distortion = distortion
        self.padding_cells = padding_cells

    def __call__(self, coords, feats):
        y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
        x_min, x_max = coords[:, 1].min(), coords[:, 1].max()
        height = y_max - y_min + 1
        width = x_max - x_min + 1

        # Control grid size covering image + padding
        grid_h_img = max(2, int(np.ceil(height / self.cell_size)))
        grid_w_img = max(2, int(np.ceil(width / self.cell_size)))

        grid_h = grid_h_img + 2 * self.padding_cells
        grid_w = grid_w_img + 2 * self.padding_cells

        # Create control grid pixel coords
        control_y = np.linspace(
            y_min - self.padding_cells * self.cell_size,
            y_max + self.padding_cells * self.cell_size,
            grid_h
        )
        control_x = np.linspace(
            x_min - self.padding_cells * self.cell_size,
            x_max + self.padding_cells * self.cell_size,
            grid_w
        )

        grid_x, grid_y = np.meshgrid(control_x, control_y)

        # Random displacement per control point
        displacement_x = np.random.uniform(-self.distortion, self.distortion, (grid_h, grid_w))
        displacement_y = np.random.uniform(-self.distortion, self.distortion, (grid_h, grid_w))

        distorted_x = grid_x + displacement_x
        distorted_y = grid_y + displacement_y

        # Normalize coords into control grid index space (before random shift)
        coords_norm_x = (coords[:, 1] - control_x[0]) / (control_x[-1] - control_x[0]) * (grid_w - 1)
        coords_norm_y = (coords[:, 0] - control_y[0]) / (control_y[-1] - control_y[0]) * (grid_h - 1)

        # Interpolate distorted control points at shifted coords
        interp_x = map_coordinates(distorted_x, [coords_norm_y, coords_norm_x], order=1, mode='reflect')
        interp_y = map_coordinates(distorted_y, [coords_norm_y, coords_norm_x], order=1, mode='reflect')

        distorted_coords = np.stack((interp_y, interp_x), axis=-1)

        return distorted_coords, feats

    
class DoNothing:
    def __call__(self, coords, feats):
        return coords, feats

    
class SemiRandomCrop:
    def __init__(self, new_x, new_y, clip_x=10, clip_y=10):
        self.new_y = new_y
        self.new_x = new_x
        self.clip_x = clip_x
        self.clip_y = clip_y

    def __call__(self, coords, feats):
        new_feats = feats.copy()
        y_round = np.round(coords[:, 0]).astype(np.int32)
        x_round = np.round(coords[:, 1]).astype(np.int32)
        new_coords = np.stack([y_round, x_round], axis=-1)
        
        y_max = np.ceil(np.percentile(y_round,95))
        y_min = np.floor(np.percentile(y_round,5))
        x_max = np.max(x_round)
        x_min = np.min(x_round)
        
        shift_x, shift_y = 0, 0
        
        if x_max - x_min >= self.new_x:
            shift_x = random.randint(self.new_x-x_max, -1*x_min)
        else:
            shift_x = random.randint(-x_min -self.clip_x, self.new_x-x_max + self.clip_x)
        if y_max - y_min >= self.new_y:
            shift_y = random.randint(-y_min-self.clip_y, -y_min+self.clip_y)
        else:
            shift_y = random.randint(-y_min-self.clip_y, self.new_y-y_max+self.clip_y)
            
        new_coords = new_coords + np.array([shift_y, shift_x])
        mask = (new_coords[:,0] > 0) & (new_coords[:,0] < (self.new_y)) \
        & (new_coords[:,1] > 0) & (new_coords[:,1] < (self.new_x))
        
        return new_coords[mask], new_feats[mask]
    
        

class BilinearSplat:
    def __init__(self, threshold=0.04):
        self.threshold=threshold
        
    def __call__(self, coords, feats):
        
        feats = np.squeeze(feats)  # Remove single-dimensional entries from shape
        
        # Floor and ceil coordinates for each point
        x0, y0 = np.floor(coords[:, 1]).astype(int), np.floor(coords[:, 0]).astype(int)
        x1, y1 = x0 + 1, y0 + 1
    
        # Calculate the weights for bilinear interpolation
        wx1 = coords[:, 1] - x0
        wx0 = 1 - wx1
        wy1 = coords[:, 0] - y0
        wy0 = 1 - wy1
    
        # Coordinates for the four corners
        coords00 = np.stack([y0, x0], axis=-1)
        coords10 = np.stack([y1, x0], axis=-1)
        coords01 = np.stack([y0, x1], axis=-1)
        coords11 = np.stack([y1, x1], axis=-1)
        
        # Calculate interpolated feature values for each of the four corners
        f00 = feats * (wx0 * wy0)
        f10 = feats * (wx0 * wy1)
        f01 = feats * (wx1 * wy0)
        f11 = feats * (wx1 * wy1)
        
        # Combine coordinates and features
        coords_combined = np.vstack([coords00,coords01,coords10,coords11])
        features_combined = np.concatenate([f00, f01, f10, f11])
    
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


## Warning: entirely broken, do not use    
class GaussianSmear:
    def __init__(self, sigma=0.8, threshold=0.04, y_max=280, x_max=140):
        self.sigma = sigma
        self.threshold = threshold
        self.y_max = y_max
        self.x_max = x_max

    @staticmethod
    def _safe_coords_feats(coords, feats):
        coords = np.array(coords, dtype=float)
        feats = np.array(feats, dtype=float)

        # Ensure coords is (N,2)
        if coords.ndim == 1 and coords.size == 2:
            coords = coords.reshape(1, 2)
        elif coords.ndim == 0:
            coords = np.zeros((0, 2), dtype=float)

        # Ensure feats is (N,)
        if feats.ndim == 0:
            feats = np.zeros((0,), dtype=float)
        elif feats.ndim > 1:
            feats = feats.ravel()

        return coords, feats

        
    def __call__(self, coords, feats):

        coords, feats = self._safe_coords_feats(coords, feats)
        
        # Early exit if empty
        if coords.shape[0] == 0 or feats.shape[0] == 0:
            return np.zeros((0, 2), dtype=int), np.zeros((0, 1))        

        N = coords.shape[0]

        # Full kernel offsets (truncated at 2*sigma)
        radius = int(np.ceil(2 * self.sigma))
        y_offsets, x_offsets = np.meshgrid(
            np.arange(-radius, radius + 1),
            np.arange(-radius, radius + 1),
            indexing='ij'
        )
        kernel = np.exp(-(x_offsets**2 + y_offsets**2) / (2 * self.sigma**2))
        kernel /= kernel.sum()  # normalize

        kernel_offsets = np.stack([y_offsets.ravel(), x_offsets.ravel()], axis=1)
        kernel_values = kernel.ravel()

        K = kernel_offsets.shape[0]  # total points in kernel

        # Repeat for all points
        coords_expanded = np.repeat(coords, K, axis=0) + np.tile(kernel_offsets, (N, 1))
        feats_expanded = np.repeat(feats, K) * np.tile(kernel_values, N)

        # Threshold contributions
        mask = feats_expanded >= self.threshold
        coords_expanded = coords_expanded[mask]
        feats_expanded = feats_expanded[mask]

        if coords_expanded.shape[0] == 0:
            return np.zeros((0, 2), dtype=int), np.zeros((0, 1))

        # Clip coordinates
        mask_grid = (coords_expanded[:, 0] >= 0) & (coords_expanded[:, 0] < self.y_max) & \
                    (coords_expanded[:, 1] >= 0) & (coords_expanded[:, 1] < self.x_max)
        coords_expanded = coords_expanded[mask_grid].astype(int)
        feats_expanded = feats_expanded[mask_grid]

        if coords_expanded.shape[0] == 0:
            return np.zeros((0, 2), dtype=int), np.zeros((0, 1))

        # Consolidate features at unique coordinates
        unique_coords, indices = np.unique(coords_expanded, axis=0, return_inverse=True)
        summed_feats = np.zeros(len(unique_coords))
        np.add.at(summed_feats, indices, feats_expanded)

        # Apply threshold again
        mask_final = summed_feats >= self.threshold
        unique_coords = unique_coords[mask_final]
        summed_feats = summed_feats[mask_final].reshape(-1, 1)

        return unique_coords, summed_feats

class RandomStretch2D:
    def __init__(self, stretch_y=0.06, stretch_x=0.06, max_y=None, max_x=None):
        self.stretch_y = stretch_y
        self.stretch_x = stretch_x
        self.max_y = max_y
        self.max_x = max_x

    def __call__(self, coords, feats):
        # Random scale factors
        fcoords = coords.astype(float)
        scale_y = np.random.normal(loc=1.0, scale=self.stretch_y)
        scale_x = np.random.normal(loc=1.0, scale=self.stretch_x)

        scale_matrix = np.array([
            [scale_y, 0.0],
            [0.0, scale_x]
        ])

        ## Keep old default behaviour
        if self.max_y is not None and self.max_x is not None:
            # Pick a random central-ish point
            cx = np.random.uniform(self.max_x/4, 3*self.max_x/4)
            cy = np.random.uniform(self.max_y/4, 3*self.max_y/4)
            center = np.array([cy, cx])
        ## Add a new option to take a point within the image
        else:
            #centroid = coords.mean(axis=0)
            #std = coords.std(axis=0)
            #truncnorm.rvs(-2, 2, loc=0, scale=std)
            temp_idx = np.random.randint(len(fcoords))
            center = fcoords[temp_idx]
            
        shifted = fcoords - center
        stretched = shifted @ scale_matrix
        stretched_coords = stretched + center

        return stretched_coords, feats
    
    
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
        raw_coords = np.vstack((row, col)).T
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

    return cat_bcoords, cat_bfeats, len(raw_feats)*2


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
        # Check for 'label' dataset and fall back if missing
        label = -1
        if 'label' in group: label = group['label'][()]

        ## Use the format that ME requires
        ## Note that we can't build the sparse tensor here because ME uses some sort of global indexing
        ## And this function is replicated * num_workers
        coords = np.vstack((row, col)).T 
        feats = data.reshape(-1, 1)  # Reshape data to be of shape (N, 1)            
        coords, feats = self.transform(coords, feats)

        if self.return_metadata:
            event_id = group.attrs.get("event_id", this_idx)
            filename = os.path.basename(self.hdf5_files[file_index])
            return coords, feats, label, filename, event_id        
        return coords, feats, label
    
def solo_ME_collate_fn(batch):
    coords, feats, labels = zip(*batch)
    
    # Create batched coordinates for the SparseTensor input
    bcoords  = ME.utils.batched_coordinates(coords)
    
    # Concatenate all lists
    bfeats  = torch.from_numpy(np.concatenate(feats, 0)).float()
    
    return bcoords, bfeats, labels


def solo_ME_collate_fn_with_meta(batch):
    coords, feats, labels, filenames, event_ids = zip(*batch)
    bcoords = ME.utils.batched_coordinates(coords)
    bfeats = torch.from_numpy(np.concatenate(feats, 0)).float()
    return bcoords, bfeats, labels, filenames, event_ids


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
## Have to very careful in the order. Some require integer points and so force everything on a grid
## This introduces aliasing effects
def get_transform(det="single", aug_type=None):

    ThisCrop = RandomCrop
    x_max = 128
    y_max = 256
    x_orig = 140
    y_orig = 280
    
    if det == "fsd":
        ThisCrop = RandomCrop
        x_max=256
        y_max=768
        x_orig=256
        y_orig=800


    # small
    if aug_type=="smallaug":
        return transforms.Compose([
       	    RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig]),
            RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig]),
    	    RandomInPlaceHorizontalFlip(),
            RandomInPlaceVerticalFlip(),
    	    RandomHorizontalFlip(x_max=x_orig),
            RandomVerticalFlip(y_max=y_orig),    
            RandomShear2D(0.03, 0.03),
            RandomRotation2D(3),
            RandomStretch2D(0.05, 0.05),
    	    RandomGridDistortion2D(100, 5),
    	    RandomScaleCharge(0.05),
    	    RandomJitterCharge(0.05),
            SemiRandomCrop(x_max, y_max, 20),
        ])
        
    # smallbilin
    if aug_type=="smallaugbilin":
        return transforms.Compose([
       	    RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig]),
            RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig]),
    	    RandomInPlaceHorizontalFlip(),
            RandomInPlaceVerticalFlip(),
    	    RandomHorizontalFlip(x_max=x_orig),
            RandomVerticalFlip(y_max=y_orig),    
            RandomShear2D(0.03, 0.03),
            RandomRotation2D(3),
            RandomStretch2D(0.05, 0.05),
    	    RandomGridDistortion2D(100, 5),
    	    RandomScaleCharge(0.05),
    	    RandomJitterCharge(0.05),
    	    BilinearSplat(0.5),
            SemiRandomCrop(x_max, y_max, 20),
        ])
        
    # big
    if aug_type=="bigaug":
        return transforms.Compose([
    	    RandomBlockZeroImproved([50,100], [5,10], [0,x_orig], [0,y_orig]),
            RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig]),
    	    RandomInPlaceHorizontalFlip(),
            RandomInPlaceVerticalFlip(),
    	    RandomHorizontalFlip(x_max=x_orig),
            RandomVerticalFlip(y_max=y_orig),    
            RandomShear2D(0.1, 0.1),
            RandomRotation2D(6),
            RandomStretch2D(0.1, 0.1),
    	    RandomGridDistortion2D(100, 5),
    	    RandomScaleCharge(0.05),
    	    RandomJitterCharge(0.05),
            SemiRandomCrop(x_max, y_max, 20),
            RandomDropout(0.1)
        ])
        
    # bigbilin
    if aug_type=="bigaugbilin":
        return transforms.Compose([
    	    RandomBlockZeroImproved([50,100], [5,10], [0,x_orig], [0,y_orig]),
            RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig]),
    	    RandomInPlaceHorizontalFlip(),
            RandomInPlaceVerticalFlip(),
    	    RandomHorizontalFlip(x_max=x_orig),
            RandomVerticalFlip(y_max=y_orig),    
            RandomShear2D(0.1, 0.1),
            RandomRotation2D(6),
            RandomStretch2D(0.1, 0.1),
    	    RandomGridDistortion2D(100, 5),
    	    RandomScaleCharge(0.05),
    	    RandomJitterCharge(0.05),
    	    BilinearSplat(0.5),
            SemiRandomCrop(x_max, y_max, 20),
            RandomDropout(0.1)
        ])
    
    ## Is the worse performance here due to the dropout, the threshold change, or the smaller stretch?
    if aug_type=="newvbig":
        return transforms.Compose([
            RandomBlockZeroImproved([50,100], [5,10], [0,x_orig], [0,y_orig]),
            RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig]),
            RandomInPlaceHorizontalFlip(),
            RandomInPlaceVerticalFlip(),
            RandomHorizontalFlip(x_max=x_orig),
            RandomVerticalFlip(y_max=y_orig),
            RandomShear2D(0.1, 0.1),
            RandomRotation2D(6),
            RandomStretch2D(0.15, 0.15),
            RandomGridDistortion2D(100, 5),
            RandomScaleCharge(0.1),
            RandomJitterCharge(0.05),
            BilinearSplat(0.5),
            RandomDropout(0.1),
            RandomCrop(x_max, y_max, 20)
	])

    if aug_type=="newvvbig":
        return transforms.Compose([
            RandomBlockZeroImproved([50,100], [5,10], [0,x_orig], [0,y_orig]),
            RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig]),
            RandomInPlaceHorizontalFlip(),
            RandomInPlaceVerticalFlip(),
            RandomHorizontalFlip(x_max=x_orig),
            RandomVerticalFlip(y_max=y_orig),
            RandomShear2D(0.2, 0.2),
            RandomRotation2D(10),
            RandomStretch2D(0.25, 0.25),
            RandomGridDistortion2D(100, 8),
            RandomScaleCharge(0.1),
            RandomJitterCharge(0.05),
            BilinearSplat(0.5),
            RandomDropout(0.1),
            RandomCrop(x_max, y_max, 20)
        ])

    raise ValueError("Unknown augmentation type:", aug_type)

    
