import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates

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
        
        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
            
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

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
            
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
        
        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
            
        new_coords = coords.copy()

        if np.random.rand() < self.p:
            new_coords[:,1] = self.x_max - new_coords[:,1]

        return new_coords,feats

    
class RandomVerticalFlip:
    def __init__(self, p=0.5, y_max=512):
        self.p = p
        self.y_max = y_max

    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
            
        new_coords = coords.copy()
        
        if np.random.rand() < self.p:
            new_coords[:,0] = self.y_max - new_coords[:,0]

        return new_coords,feats

    
    
## Need to define a fairly standard functions that work for ME tensors
class RandomRotation2D:
    def __init__(self, angle, p=1):
        self.p = p
        self.angle = angle

    def _M(self, theta):
        # Generate a 2D rotation matrix for a given angle theta
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
	])

    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
        
        # Add some probability to return immediately
        if np.random.rand() > self.p: return coords, feats
            
        # Generate a random rotation angle
        angle = np.deg2rad(np.random.normal(loc=0, scale=self.angle))
        fcoords = coords.astype(float)

        # Decide on the starting point within the image
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
    def __init__(self, sigma_y, sigma_x, p=1):
        self.p = p
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def __call__(self, coords, feats):

        # Add some probability to return immediately
        if np.random.rand() > self.p: return coords, feats

        fcoords = coords.astype(float)

        shear_x = np.random.normal(loc=0, scale=self.sigma_x)
        shear_y = np.random.normal(loc=0, scale=self.sigma_y)

        shear_matrix = np.array([
            [1, shear_x],
            [shear_y, 1]
        ])

        ## Define the centre for shearing
        temp_idx = np.random.randint(len(fcoords))
        center = fcoords[temp_idx]
            
        shifted = fcoords - center
        rotated = shifted @ shear_matrix
        rotated_coords = rotated + center
        return rotated_coords, feats

class RandomPixelNoise2D:
    def __init__(self, poisson_mean=10, image_bounds=(0, 256, 0, 512)):
        self.poisson_mean = poisson_mean
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

        # Sample noise values from existing features
        idx = np.random.randint(0, len(feats), size=n_noise)
        noise_feats = feats[idx].copy()
        
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
    def __init__(self, nblocks=[0,6], block_range=[0,10], xrange=[0,140], yrange=[0,280], p=1):
        self.p = 1
        self.nblocks = nblocks
        self.rblocks = block_range
        self.xrange = xrange
        self.yrange = yrange

    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
        
        # Add some probability to return immediately
        if np.random.rand() > self.p: return coords, feats
        
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
    def __init__(self, max_frac=0.1, p=1):
        self.p = 1
        self.max_frac = max_frac

    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
        
        # Add some probability to return immediately
        if np.random.rand() > self.p: return coords, feats

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

## This always needs to be done 
class JitterCoords:
    def __init__(self, coord_jitter=0.5):
        self.coord_jitter = coord_jitter

    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
        
        jitter = np.random.uniform(
            low=-self.coord_jitter,
            high=self.coord_jitter,
            size=coords.shape
        )
        coords_new = coords + jitter
        return coords_new, feats

class SplitJitterCoords:
    def __init__(self, n_sub=10, coord_jitter=0.5):
        self.n_sub = n_sub
        self.coord_jitter = coord_jitter

    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
        
        coords_out = np.repeat(coords, self.n_sub, axis=0)
        feats_out = np.repeat(feats/self.n_sub, self.n_sub, axis=0)
        
        jitter = np.random.uniform(
            low=-self.coord_jitter,
            high=self.coord_jitter,
            size=coords_out.shape
        )

        return coords_out+jitter, feats_out


## Move the initial grid within a pixel width in both dimensions
class GridJitter:
    def __init__(self, ndim=2):
        self.ndim = ndim

    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
        
        grid_offset = np.random.uniform(-0.5, 0.5, size=self.ndim)
        return coords + grid_offset, feats



## Apply a Gaussian jitter to all values
class RandomJitterCharge:
    def __init__(self, width=0.1, p=1):
        self.p = p
        self.width = width

    def __call__(self,  coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
            
        # Add some probability to return immediately
        if np.random.rand() > self.p: return coords, feats

        scale_factors = np.random.normal(loc=1.0, scale=self.width, size=feats.shape)
        new_feats = feats*scale_factors
        return coords, new_feats

## Remove the log transform...
class UnlogCharge:
    def __call__(self, coords, feats):
        Q = np.power(10.0, feats) - 1.0
        return coords, Q

## Re-apply the log transform...
class RelogCharge:
    def __call__(self, coords, feats):
        Z = np.log10(1.0 + np.maximum(feats, 0.0))
        return coords, Z

    
# Scale the entire feature vector by a single scaling factor
class RandomScaleCharge:
    def __init__(self, width=0.1, p=1):
        self.p = p
        self.width = width

    def __call__(self,  coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
            
        # Add some probability to return immediately
        if np.random.rand() > self.p: return coords, feats
        
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

    
## Apply distortions in a regular grid, with random strength at each point up to some maximum, smoothed by some amount
## Cell size is the size of the distortion grid (in pixels, assumed square)
## Distortion strength is the same
class RandomGridDistortion2D:
    def __init__(self, cell_size=50, distortion=5, padding_cells=2, cell_size_jitter=10, p=1):
        self.p = p
        self.cell_size = cell_size
        self.distortion = distortion
        self.padding_cells = padding_cells
        self.cell_size_jitter = cell_size_jitter

    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
            
        # Add some probability to return immediately
        if np.random.rand() > self.p: return coords, feats
        
        y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
        x_min, x_max = coords[:, 1].min(), coords[:, 1].max()
        height = y_max - y_min + 1
        width = x_max - x_min + 1

        ## Add some randomness to the grid size
        cell_size_h = np.random.uniform(self.cell_size-self.cell_size_jitter, self.cell_size+self.cell_size_jitter)
        cell_size_w = np.random.uniform(self.cell_size-self.cell_size_jitter, self.cell_size+self.cell_size_jitter)

        # Control grid size covering image + padding
        grid_h_img = max(2, int(np.ceil(height / cell_size_h)))
        grid_w_img = max(2, int(np.ceil(width / cell_size_w)))

        grid_h = grid_h_img + 2 * self.padding_cells
        grid_w = grid_w_img + 2 * self.padding_cells

        # Create control grid pixel coords
        control_y = np.linspace(
            y_min - self.padding_cells * cell_size_h,
            y_max + self.padding_cells * cell_size_h,
            grid_h
        )
        control_x = np.linspace(
            x_min - self.padding_cells * cell_size_w,
            x_max + self.padding_cells * cell_size_w,
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

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
            
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
        W = 10000
        hash_vals = coords_combined[:,0] * W + coords_combined[:,1]
        unique_hashes, inverse = np.unique(hash_vals, return_inverse=True)
        summed_feats = np.zeros(len(unique_hashes), dtype=features_combined.dtype)
        np.add.at(summed_feats, inverse, features_combined)
        unique_coords = np.stack([unique_hashes // W, unique_hashes % W], axis=-1)

        #unique_coords, indices = np.unique(coords_combined, axis=0, return_inverse=True)
        #summed_feats = np.zeros(len(unique_coords))    
        #np.add.at(summed_feats, indices, features_combined)

        # Create a mask for values above the threshold
        mask = summed_feats >= self.threshold
    
        # Apply the mask to filter features and coordinates
        unique_coords = unique_coords[mask]
        summed_feats = summed_feats[mask]        
        
        # Reshape summed_feats to (N, 1)
        summed_feats = summed_feats.reshape(-1, 1)
        
        return unique_coords, summed_feats

## Add an optional threshold
class BilinearSplatMod:
    def __init__(self, threshold_min=0.04, threshold_max=0.04, p=0.5):
        self.threshold_min=threshold_min
        self.threshold_max=threshold_max
        self.p = p
        
    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
            
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
        W = 10000
        hash_vals = coords_combined[:,0] * W + coords_combined[:,1]
        unique_hashes, inverse = np.unique(hash_vals, return_inverse=True)
        summed_feats = np.zeros(len(unique_hashes), dtype=features_combined.dtype)
        np.add.at(summed_feats, inverse, features_combined)
        unique_coords = np.stack([unique_hashes // W, unique_hashes % W], axis=-1)
        
        #unique_coords, indices = np.unique(coords_combined, axis=0, return_inverse=True)
        #summed_feats = np.zeros(len(unique_coords))    
        #np.add.at(summed_feats, indices, features_combined)

        ## Get the threshold
        threshold = np.random.uniform(self.threshold_min, self.threshold_max)
        
        if np.random.rand() < self.p:
            # Create a mask for values above the threshold
            mask = summed_feats >= threshold
    
            # Apply the mask to filter features and coordinates
            unique_coords = unique_coords[mask]
            summed_feats = summed_feats[mask]        
        
        # Reshape summed_feats to (N, 1)
        summed_feats = summed_feats.reshape(-1, 1)
        
        return unique_coords, summed_feats


class ExpandedBilinearSplat:
    def __init__(self, threshold=0.04, radius=1):
        self.threshold = threshold
        self.radius = int(radius)

    def __call__(self, coords, feats):
        
        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
            
        feats = np.squeeze(feats)
        x0 = np.floor(coords[:, 1]).astype(int)
        y0 = np.floor(coords[:, 0]).astype(int)
        fx = coords[:, 1] - x0
        fy = coords[:, 0] - y0

        # All displacements in the local window
        dx = np.arange(-self.radius, self.radius + 1)
        dy = np.arange(-self.radius, self.radius + 1)

        # Broadcast to all pixels in the radius × radius window
        dx_grid, dy_grid = np.meshgrid(dx, dy, indexing='xy')
        dx_flat = dx_grid.ravel()
        dy_flat = dy_grid.ravel()

        # Compute bilinear weights for each displacement
        wx = np.clip(1.0 - np.abs(dx_flat[None, :] - fx[:, None]), 0, 1)
        wy = np.clip(1.0 - np.abs(dy_flat[None, :] - fy[:, None]), 0, 1)
        w = wx * wy  # shape: (N_points, n_disp)

        # Compute displaced coordinates
        cx = x0[:, None] + dx_flat[None, :]
        cy = y0[:, None] + dy_flat[None, :]

        # Flatten everything
        coords_all = np.stack([cy.ravel(), cx.ravel()], axis=-1)
        feats_all = (feats[:, None] * w).ravel()

        # Consolidate features at unique coordinates
        unique_coords, indices = np.unique(coords_all, axis=0, return_inverse=True)
        summed_feats = np.zeros(len(unique_coords))
        np.add.at(summed_feats, indices, feats_all)

        # Apply threshold and reshape
        mask = summed_feats >= self.threshold
        unique_coords = unique_coords[mask]
        summed_feats = summed_feats[mask].reshape(-1, 1)

        return unique_coords, summed_feats



class RandomStretch2D:
    def __init__(self, stretch_y=0.06, stretch_x=0.06, p=1):
        self.p = p
        self.stretch_y = stretch_y
        self.stretch_x = stretch_x

    def __call__(self, coords, feats):

        # Add some probability to return immediately
        if np.random.rand() > self.p: return coords, feats
        
        # Random scale factors
        fcoords = coords.astype(float)
        scale_y = np.random.normal(loc=1.0, scale=self.stretch_y)
        scale_x = np.random.normal(loc=1.0, scale=self.stretch_x)

        scale_matrix = np.array([
            [scale_y, 0.0],
            [0.0, scale_x]
        ])

        ## Randomly pick a start point
        temp_idx = np.random.randint(len(fcoords))
        center = fcoords[temp_idx]
            
        shifted = fcoords - center
        stretched = shifted @ scale_matrix
        stretched_coords = stretched + center

        return stretched_coords, feats
    
    
