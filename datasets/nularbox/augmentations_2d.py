import random
import numpy as np
import torchvision.transforms.v2 as transforms
import core.data.augmentations_2d as aug

## Crop an x by y region from the center of the image, with a jitter on the center position
class RandomCenterCrop:
    def __init__(self,
                 orig_size,
                 new_size,
                 center=None,
                 jitter=10):
        self.orig_size = orig_size
        self.new_size = new_size
        self.jitter = jitter

        ## If center isn't defined, default to the literal image center
        self.center = (self.img_size // 2.0 if center is None
                       else np.asarray(center, dtype=int))
        
    def __call__(self, coords, feats):

        ## Guard against empty input
        if coords.shape[0] == 0: return coords, feats
            
        new_feats = feats.copy()
        y_round = np.round(coords[:, 0]).astype(np.int32)
        x_round = np.round(coords[:, 1]).astype(np.int32)
        new_coords = np.stack([y_round, x_round], axis=-1)
        
        shift_y = self.center[0] - self.orig_size[0]//2 + random.randint(-self.jitter,self.jitter)
        shift_x = self.center[1] - self.orig_size[1]//2 + random.randint(-self.jitter,self.jitter)

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
    def __init__(self,
                 angle,
                 img_size,
                 center=None,
                 jitter=10,
                 p=1):
        self.p = p
        self.angle = angle
        self.img_size = img_size
        self.jitter = jitter

        ## If center isn't defined, default to the literal image center
        self.center = (self.img_size / 2.0 if center is None
                       else np.asarray(center, dtype=float))

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
            self.center[1] + np.random.uniform(-self.jitter, self.jitter),
            self.center[0] + np.random.uniform(-self.jitter, self.jitter)
        ])
        
        # Get the 2D rotation matrix
        R = self._M(angle)
        
        # Shift and apply the rotation
        shifted = fcoords - center
        rotated = shifted @ R
        rotated_coords = rotated + center        
        return rotated_coords, feats

    
    
class RandomCentralShear2D:
    def __init__(self,
                 sigma_y,
                 sigma_x,
                 img_size,
                 center=None,
                 jitter=10,
                 p=1):
        self.p = p
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.img_size = img_size
        self.jitter = jitter

        ## If center isn't defined, default to the literal image center
        self.center = (self.img_size / 2.0 if center is None
                       else np.asarray(center, dtype=float))

        
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
            self.center[1] + np.random.uniform(-self.jitter, self.jitter),
            self.center[0] + np.random.uniform(-self.jitter, self.jitter)
        ])
            
        shifted = fcoords - center
        rotated = shifted @ shear_matrix
        rotated_coords = rotated + center
        return rotated_coords, feats


class RandomCentralStretch2D:
    def __init__(self,
                 stretch_y,
                 stretch_x,
                 img_size,
                 center=None,
                 jitter=10,
                 p=1):
        self.p = p
        self.stretch_y = stretch_y
        self.stretch_x = stretch_x
        self.img_size = img_size
        self.jitter = jitter

        ## If center isn't defined, default to the literal image center
        self.center = (self.img_size / 2.0 if center is None
                       else np.asarray(center, dtype=float))
        
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
            self.center[1] + np.random.uniform(-self.jitter, self.jitter),
            self.center[0] + np.random.uniform(-self.jitter, self.jitter)
        ])
            
        shifted = fcoords - center
        stretched = shifted @ scale_matrix
        stretched_coords = stretched + center

        return stretched_coords, feats

class LogAlphaCharge:
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, coords, feats):
        Z = np.log10(1.0 + self.alpha*np.maximum(feats, 0.0))/np.log10(1.0 + self.alpha)
        return coords, Z

class LogAlphaChargeRandom:
    def __init__(self, alpha_min, alpha_max):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
    def __call__(self, coords, feats):
        
        this_alpha = np.random.uniform(self.alpha_min,
                                       self.alpha_max)
        
        Z = np.log10(1.0 + this_alpha*np.maximum(feats, 0.0))/np.log10(1.0 + this_alpha)
        return coords, Z
    

def get_transform(image_size=256, aug_type=None, aug_prob=1, aug_val=None):

    x_max=y_max=image_size

    x_orig=512
    y_orig=512

    if aug_type == "minimal":
        return transforms.Compose([
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "flip":
        return transforms.Compose([
            aug.RandomVerticalFlip(y_max=y_orig, p=0.5),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])    
    
    if aug_type == "block":
        return transforms.Compose([
            aug.RandomBlockZeroImproved([5,20], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            aug.RandomBlockZeroImproved([50,200], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "dropout":
        drop_val = 0.2
        if aug_val is not None:
            drop_val = aug_val
        return transforms.Compose([
            aug.RandomDropout(drop_val, p=aug_prob),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "alpha":
        return transforms.Compose([
            LogAlphaChargeRandom(2, 8),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])    
    
    if aug_type == "splat":
        return transforms.Compose([
            aug.GridJitter(),
            aug.JitterCoords(),
            aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "splatsmalljit":
        return transforms.Compose([
            aug.GridJitter(2, 0.1),
            aug.JitterCoords(0.1),
            aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])
    
    if aug_type == "rotate":
        return transforms.Compose([
            aug.GridJitter(),
            aug.JitterCoords(),
            RandomCentralRotation2D(30, img_size=[y_orig, x_orig], frac=0.2, p=aug_prob),
            aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "rotatesmall":
        return transforms.Compose([
            aug.GridJitter(),
            aug.JitterCoords(),
            RandomCentralRotation2D(10, img_size=[y_orig, x_orig], frac=0.2, p=aug_prob),
            aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "rotatesmalljit":
        rotate_val = 10
        if aug_val is not None:
            rotate_val = aug_val
        return transforms.Compose([
            aug.GridJitter(2, 0.1),
            aug.JitterCoords(0.1),
            RandomCentralRotation2D(rotate_val, img_size=[y_orig, x_orig], frac=0.2, p=aug_prob),
            aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "threshtest":
        threshold = 0.3
        if aug_val is not None:
            threshold = aug_val
        return transforms.Compose([
            aug.GridJitter(2, 0.1),
            aug.JitterCoords(0.1),
            RandomCentralRotation2D(10, img_size=[y_orig, x_orig], frac=0.2, p=aug_prob),
            aug.BilinearSplat(threshold),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "croptest":
        crop = 10
        if aug_val is not None:
            crop = aug_val
        return transforms.Compose([
            aug.GridJitter(2, 0.1),
            aug.JitterCoords(0.1),
            RandomCentralRotation2D(10, img_size=[y_orig, x_orig], frac=0.2, p=aug_prob),
            aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], crop)
        ])     
    
    if aug_type == "shear":
        return transforms.Compose([
            aug.GridJitter(),
            aug.JitterCoords(),
            RandomCentralShear2D(0.2, 0.2, img_size=[y_orig, x_orig], frac=0.2, p=aug_prob),
            aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "stretch":
        return transforms.Compose([
            aug.GridJitter(),
            aug.JitterCoords(),
            RandomCentralStretch2D(0.1, 0.1, img_size=[y_orig, x_orig], frac=0.2, p=aug_prob),
            aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "grid":
        return transforms.Compose([
            aug.GridJitter(),
            aug.JitterCoords(),
            aug.RandomGridDistortion2D(50, 4, 2, 10, p=aug_prob),
            aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "shearjit":
        shear_val = 0.2
        if aug_val is not None:
            shear_val = aug_val
        return transforms.Compose([
            aug.GridJitter(2, 0.1),
            aug.JitterCoords(0.1),
            RandomCentralShear2D(shear_val, shear_val, img_size=[y_orig, x_orig], frac=0.2, p=aug_prob),
            aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "stretchjit":
        stretch_val = 0.1
        if aug_val is not None:
            stretch_val = aug_val
        return transforms.Compose([
            aug.GridJitter(2, 0.1),
            aug.JitterCoords(0.1),
            RandomCentralStretch2D(stretch_val, stretch_val, img_size=[y_orig, x_orig], frac=0.2, p=aug_prob),
            aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "gridjit":
        dist_val = 4
        if aug_val is not None:
            dist_val = aug_val
        return transforms.Compose([
            aug.GridJitter(2, 0.1),
            aug.JitterCoords(0.1),
            aug.RandomGridDistortion2D(50, aug_val, 2, 10, p=aug_prob),
            aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    
    if aug_type == "charge":
        charge_val = 0.05
        if aug_val is not None:
            charge_val = aug_val
        return transforms.Compose([
            aug.RandomScaleCharge(charge_val, p=aug_prob),
            aug.RandomJitterCharge(charge_val, p=aug_prob),
            LogAlphaCharge(5),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])

    if aug_type == "nominalsmalljit":
        return transforms.Compose([
            aug.RandomBlockZeroImproved([5,20], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            aug.RandomBlockZeroImproved([50,200], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
            aug.RandomVerticalFlip(y_max=y_orig, p=0.5),
            aug.GridJitter(2, 0.1),
            aug.JitterCoords(0.1),
            RandomCentralRotation2D(10, img_size=[y_orig, x_orig], frac=0.2, p=aug_prob),
            RandomCentralShear2D(0.2, 0.2, img_size=[y_orig, x_orig], frac=0.4, p=aug_prob),
            RandomCentralStretch2D(0.1, 0.1, img_size=[y_orig, x_orig], frac=0.4, p=aug_prob),
    	    aug.RandomGridDistortion2D(50, 4, 2, 10, p=aug_prob),
    	    aug.RandomScaleCharge(0.05, p=aug_prob),
            aug.RandomJitterCharge(0.05, p=aug_prob),
    	    aug.BilinearSplatMod(0.2, 0.3),
            LogAlphaCharge(5),
            aug.RandomDropout(0.1, p=aug_prob),
            RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
        ])
    
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
    	aug.BilinearSplatMod(0.2, 0.3),
        LogAlphaCharge(5),
        aug.RandomDropout(0.1, p=aug_prob),
        RandomCenterCrop([y_orig,x_orig], [y_max,x_max], 10)
    ])
