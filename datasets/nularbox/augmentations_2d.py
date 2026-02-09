import random
import numpy as np
import torchvision.transforms.v2 as transforms
import core.data.augmentations_2d as aug

## Crop an x by y region from the center of the image, with a jitter on the center position
class RandomCenterCrop:
    def __init__(self, orig_size, new_size, jitter=10):
        self.orig_size = orig_size
        self.new_size = new_size
        self.jitter = jitter

    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
            
        new_feats = feats.copy()
        y_round = np.round(coords[:, 0]).astype(np.int32)
        x_round = np.round(coords[:, 1]).astype(np.int32)
        new_coords = np.stack([y_round, x_round], axis=-1)
        
        shift_y = self.new_size[0]//2 - self.orig_size[0]//2 + random.randint(-self.jitter,self.jitter)
        shift_x = self.new_size[1]//2 - self.orig_size[1]//2 + random.randint(-self.jitter,self.jitter)

        new_coords = new_coords + np.array([shift_y, shift_x])        
        mask = (new_coords[:,0] > 0) & (new_coords[:,0] < (self.new_size[0])) \
             & (new_coords[:,1] > 0) & (new_coords[:,1] < (self.new_size[1]))
                
        return new_coords[mask], new_feats[mask]

class ApplyThreshold:
    
    def __init__(self, threshold=0.2):
        self.threshold = threshold

    def __call__(self, coords, feats):

        mask = feats.squeeze() >= self.threshold
    
        # Apply the mask to filter features and coordinates
        mask_coords = coords[mask]
        mask_feats = feats[mask]        
        
        return mask_coords, mask_feats

class RandomCentralRotation2D:
    def __init__(self, angle, img_size, frac=0.2, p=1):
        self.p = p
        self.angle = angle
        self.img_size = img_size
        self.frac = frac

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

        ## Pick a point close to the center of the original image size to rotate around
        center = np.array([
            self.img_size[1]*(0.5 + np.random.uniform(-0.5, 0.5)*self.frac),
            self.img_size[0]*(0.5 + np.random.uniform(-0.5, 0.5)*self.frac)
        ])
        
        # Get the 2D rotation matrix
        R = self._M(angle)
        
        # Shift and apply the rotation
        shifted = fcoords - center
        rotated = shifted @ R
        rotated_coords = rotated + center        
        return rotated_coords, feats

class RandomCentralShear2D:
    def __init__(self, sigma_y, sigma_x, img_size, frac=0.2, p=1):
        self.p = p
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.img_size = img_size
        self.frac = frac

    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
        
        # Add some probability to return immediately
        if np.random.rand() > self.p: return coords, feats
        fcoords = coords.astype(float)

        shear_x = np.random.normal(loc=0, scale=self.sigma_x)
        shear_y = np.random.normal(loc=0, scale=self.sigma_y)

        shear_matrix = np.array([
            [1, shear_x],
            [shear_y, 1]
        ])

        # Pick a point close to the center of the original image size to shear around
        center = np.array([
            self.img_size[1]*(0.5 + np.random.uniform(-0.5, 0.5)*self.frac),
            self.img_size[0]*(0.5 + np.random.uniform(-0.5, 0.5)*self.frac)
        ])
            
        shifted = fcoords - center
        rotated = shifted @ shear_matrix
        rotated_coords = rotated + center
        return rotated_coords, feats


class RandomCentralStretch2D:
    def __init__(self, stretch_y, stretch_x, img_size, frac=0.2, p=1):
        self.p = p
        self.stretch_y = stretch_y
        self.stretch_x = stretch_x
        self.img_size = img_size
        self.frac = frac
        
    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
        
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

        # Pick a point close to the center of the original image size to stretch around
        center = np.array([
            self.img_size[1]*(0.5 + np.random.uniform(-0.5, 0.5)*self.frac),
            self.img_size[0]*(0.5 + np.random.uniform(-0.5, 0.5)*self.frac)
        ])
            
        shifted = fcoords - center
        stretched = shifted @ scale_matrix
        stretched_coords = stretched + center

        return stretched_coords, feats


def get_transform(image_size="256x256", aug_type=None, aug_prob=1):

    x_max=256
    y_max=256

    x_orig=512
    y_orig=512

    return transforms.Compose([
        aug.RandomBlockZeroImproved([5,20], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
        aug.RandomBlockZeroImproved([50,200], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
        aug.RandomVerticalFlip(y_max=y_orig, p=0.5),
        aug.GridJitter(),
        aug.JitterCoords(),
        RandomCentralRotation2D(30, img_size=[y_orig, x_orig], frac=0.2, p=aug_prob),
        RandomCentralShear2D(0.2, 0.2, img_size=[y_orig, x_orig], frac=0.4, p=aug_prob),
        RandomCentralStretch2D(0.1, 0.1, img_size=[y_orig, x_orig], frac=0.4, p=aug_prob),
    	aug.RandomGridDistortion2D(50, 4, 2, 10, p=aug_prob),
    	aug.RandomScaleCharge(0.05, p=aug_prob),
        aug.RandomJitterCharge(0.05, p=aug_prob),
    	aug.BilinearSplatMod(0.2, 0.3, p=aug_prob),            
        RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
    ])
