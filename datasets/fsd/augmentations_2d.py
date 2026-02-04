import torchvision.transforms.v2 as transforms
import core.data.augmentations_2d as augs

## TODO: rationalize this massive list a bit, then copy for single module --- some copy-pasta, but cleaner, I think...
def get_transform(det="single", aug_type=None, aug_prob=1):

    ThisCrop = augs.RandomCrop
    x_max = 128
    y_max = 256
    x_orig = 140
    y_orig = 280
    
    if det == "fsd":
        ThisCrop = augs.RandomCrop
        x_max=256
        y_max=768
        x_orig=256
        y_orig=800

    ## Same as bigaugbilinfix, except the aug_probs are fixed for flips
    if aug_type=="baseaug":
        return transforms.Compose([
       	    augs.RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            augs.RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
    	    augs.RandomInPlaceHorizontalFlip(p=0.5),
            augs.RandomInPlaceVerticalFlip(p=0.5),
    	    augs.RandomHorizontalFlip(x_max=x_orig, p=0.5),
            augs.RandomVerticalFlip(y_max=y_orig, p=0.5),    
            augs.JitterCoords(),
            augs.UnlogCharge(),
            augs.RandomShear2D(0.1, 0.1, p=aug_prob),
            augs.RandomRotation2D(6, p=aug_prob),
            augs.RandomStretch2D(0.1, 0.1, p=aug_prob),
    	    augs.RandomGridDistortion2D(100, 5, 2, 25, p=aug_prob),
    	    augs.RandomScaleCharge(0.05, p=aug_prob),
    	    augs.RandomJitterCharge(0.05, p=aug_prob),
    	    augs.BilinearSplat(0.5),
            augs.RelogCharge(),
            augs.RandomScaleCharge(0.02, p=aug_prob),
    	    augs.RandomJitterCharge(0.02, p=aug_prob),
            augs.SemiRandomCrop(x_max, y_max, 20),
        ])

    ## 1 - Add late stage dropout
    if aug_type=="baseaugdrop":
        return transforms.Compose([
       	    augs.RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            augs.RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
    	    augs.RandomInPlaceHorizontalFlip(p=0.5),
            augs.RandomInPlaceVerticalFlip(p=0.5),
    	    augs.RandomHorizontalFlip(x_max=x_orig, p=0.5),
            augs.RandomVerticalFlip(y_max=y_orig, p=0.5),    
            augs.JitterCoords(),
            augs.UnlogCharge(),
            augs.RandomShear2D(0.1, 0.1, p=aug_prob),
            augs.RandomRotation2D(6, p=aug_prob),
            augs.RandomStretch2D(0.1, 0.1, p=aug_prob),
    	    augs.RandomGridDistortion2D(100, 5, 2, 25, p=aug_prob),
    	    augs.RandomScaleCharge(0.05, p=aug_prob),
    	    augs.RandomJitterCharge(0.05, p=aug_prob),
    	    augs.BilinearSplat(0.5),
            augs.RelogCharge(),
            augs.RandomScaleCharge(0.02, p=aug_prob),
    	    augs.RandomJitterCharge(0.02, p=aug_prob),
            augs.SemiRandomCrop(x_max, y_max, 20),
            augs.RandomDropout(0.1, p=aug_prob)
        ])

    ## Increase the smoothing
    if aug_type=="baseaugsmooth":
        return transforms.Compose([
       	    augs.RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            augs.RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
    	    augs.RandomInPlaceHorizontalFlip(p=0.5),
            augs.RandomInPlaceVerticalFlip(p=0.5),
    	    augs.RandomHorizontalFlip(x_max=x_orig, p=0.5),
            augs.RandomVerticalFlip(y_max=y_orig, p=0.5),    
            augs.JitterCoords(),
            augs.UnlogCharge(),
            augs.RandomShear2D(0.1, 0.1, p=aug_prob),
            augs.RandomRotation2D(6, p=aug_prob),
            augs.RandomStretch2D(0.1, 0.1, p=aug_prob),
    	    augs.RandomGridDistortion2D(100, 5, 2, 25, p=aug_prob),
    	    augs.RandomScaleCharge(0.05, p=aug_prob),
    	    augs.RandomJitterCharge(0.05, p=aug_prob),
    	    augs.ExpandedBilinearSplat(0.5, 2),
            augs.RelogCharge(),
            augs.RandomScaleCharge(0.02, p=aug_prob),
    	    augs.RandomJitterCharge(0.02, p=aug_prob),
            augs.SemiRandomCrop(x_max, y_max, 20),
        ])
        
    ## More aggressive grid distortion
    if aug_type=="baseauggrid":
        return transforms.Compose([
       	    augs.RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            augs.RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
    	    augs.RandomInPlaceHorizontalFlip(p=0.5),
            augs.RandomInPlaceVerticalFlip(p=0.5),
    	    augs.RandomHorizontalFlip(x_max=x_orig, p=0.5),
            augs.RandomVerticalFlip(y_max=y_orig, p=0.5),    
            augs.JitterCoords(),
            augs.UnlogCharge(),
            augs.RandomShear2D(0.1, 0.1, p=aug_prob),
            augs.RandomRotation2D(6, p=aug_prob),
            augs.RandomStretch2D(0.1, 0.1, p=aug_prob),
    	    augs.RandomGridDistortion2D(100, 10, 2, 25, p=aug_prob),
    	    augs.RandomScaleCharge(0.05, p=aug_prob),
    	    augs.RandomJitterCharge(0.05, p=aug_prob),
    	    augs.BilinearSplat(0.5),
            augs.RelogCharge(),
            augs.RandomScaleCharge(0.02, p=aug_prob),
    	    augs.RandomJitterCharge(0.02, p=aug_prob),
            augs.SemiRandomCrop(x_max, y_max, 20),
        ])

    ## 7 - More aggressive random crops
    if aug_type=="baseaugcrop":
        return transforms.Compose([
       	    augs.RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            augs.RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
    	    augs.RandomInPlaceHorizontalFlip(p=0.5),
            augs.RandomInPlaceVerticalFlip(p=0.5),
    	    augs.RandomHorizontalFlip(x_max=x_orig, p=0.5),
            augs.RandomVerticalFlip(y_max=y_orig, p=0.5),    
            augs.JitterCoords(),
            augs.UnlogCharge(),
            augs.RandomShear2D(0.1, 0.1, p=aug_prob),
            augs.RandomRotation2D(6, p=aug_prob),
            augs.RandomStretch2D(0.1, 0.1, p=aug_prob),
    	    augs.RandomGridDistortion2D(100, 5, 2, 25, p=aug_prob),
    	    augs.RandomScaleCharge(0.05, p=aug_prob),
    	    augs.RandomJitterCharge(0.05, p=aug_prob),
    	    augs.BilinearSplat(0.5),
            augs.RelogCharge(),
            augs.RandomScaleCharge(0.02, p=aug_prob),
    	    augs.RandomJitterCharge(0.02, p=aug_prob),
            augs.SemiRandomCrop(x_max, y_max, 40, 80),
        ])

    if aug_type=="baseaugsplit":
        return transforms.Compose([
       	    augs.RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            augs.RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
    	    augs.RandomInPlaceHorizontalFlip(p=0.5),
            augs.RandomInPlaceVerticalFlip(p=0.5),
    	    augs.RandomHorizontalFlip(x_max=x_orig, p=0.5),
            augs.RandomVerticalFlip(y_max=y_orig, p=0.5),    
            augs.UnlogCharge(),
            augs.SplitJitterCoords(10),
            augs.RandomShear2D(0.1, 0.1, p=aug_prob),
            augs.RandomRotation2D(6, p=aug_prob),
            augs.RandomStretch2D(0.1, 0.1, p=aug_prob),
    	    augs.RandomGridDistortion2D(100, 10, 2, 25, p=aug_prob),
    	    augs.RandomScaleCharge(0.05, p=aug_prob),
    	    augs.RandomJitterCharge(0.05, p=aug_prob),
    	    augs.BilinearSplat(0.5),
            augs.RelogCharge(),
            augs.RandomScaleCharge(0.02, p=aug_prob),
    	    augs.RandomJitterCharge(0.02, p=aug_prob),
            augs.SemiRandomCrop(x_max, y_max, 40, 80),
        ]) 

    if aug_type=="baseaugsplitgridmod":
        return transforms.Compose([
       	    augs.RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            augs.RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
    	    augs.RandomInPlaceHorizontalFlip(p=0.5),
            augs.RandomInPlaceVerticalFlip(p=0.5),
    	    augs.RandomHorizontalFlip(x_max=x_orig, p=0.5),
            augs.RandomVerticalFlip(y_max=y_orig, p=0.5),    
            augs.UnlogCharge(),
            augs.GridJitter(),
            augs.SplitJitterCoords(10),
            augs.RandomShear2D(0.1, 0.1, p=aug_prob),
            augs.RandomRotation2D(6, p=aug_prob),
            augs.RandomStretch2D(0.1, 0.1, p=aug_prob),
    	    augs.RandomGridDistortion2D(100, 10, 2, 25, p=aug_prob),
            augs.BilinearSplatMod(0.3, 0.5, p=0.5),
    	    augs.RandomScaleCharge(0.05, p=aug_prob),
    	    augs.RandomJitterCharge(0.05, p=aug_prob),
            augs.RelogCharge(),
            augs.RandomScaleCharge(0.02, p=aug_prob),
    	    augs.RandomJitterCharge(0.02, p=aug_prob),
            augs.SemiRandomCrop(x_max, y_max, 40, 80),
        ])         
    
    if aug_type=="baseauggridmodnoise":
        return transforms.Compose([
       	    augs.RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            augs.RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
    	    augs.RandomInPlaceHorizontalFlip(p=0.5),
            augs.RandomInPlaceVerticalFlip(p=0.5),
    	    augs.RandomHorizontalFlip(x_max=x_orig, p=0.5),
            augs.RandomVerticalFlip(y_max=y_orig, p=0.5),    
            augs.UnlogCharge(),
            augs.RandomPixelNoise2D(30),
            augs.GridJitter(),
            augs.JitterCoords(),
            augs.RandomShear2D(0.1, 0.1, p=aug_prob),
            augs.RandomRotation2D(6, p=aug_prob),
            augs.RandomStretch2D(0.1, 0.1, p=aug_prob),
    	    augs.RandomGridDistortion2D(100, 10, 2, 25, p=aug_prob),
    	    augs.RandomScaleCharge(0.05, p=aug_prob),
    	    augs.RandomJitterCharge(0.05, p=aug_prob),
    	    augs.BilinearSplatMod(0.3, 0.5, p=0.5),
            augs.RelogCharge(),
            augs.RandomScaleCharge(0.02, p=aug_prob),
    	    augs.RandomJitterCharge(0.02, p=aug_prob),
            augs.SemiRandomCrop(x_max, y_max, 40, 80),
        ])
    
    if aug_type=="newbaseaug":
        return transforms.Compose([
       	    augs.RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            augs.RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
    	    augs.RandomInPlaceHorizontalFlip(p=0.5),
            augs.RandomInPlaceVerticalFlip(p=0.5),
    	    augs.RandomHorizontalFlip(x_max=x_orig, p=0.5),
            augs.RandomVerticalFlip(y_max=y_orig, p=0.5),    
            augs.UnlogCharge(),
            augs.GridJitter(),
            augs.JitterCoords(),
            augs.RandomShear2D(0.1, 0.1, p=aug_prob),
            augs.RandomRotation2D(6, p=aug_prob),
            augs.RandomStretch2D(0.1, 0.1, p=aug_prob),
    	    augs.RandomGridDistortion2D(100, 10, 2, 25, p=aug_prob),
    	    augs.RandomScaleCharge(0.05, p=aug_prob),
    	    augs.RandomJitterCharge(0.05, p=aug_prob),
    	    augs.BilinearSplatMod(0.3, 0.5, p=0.5),
            augs.RelogCharge(),
            augs.RandomScaleCharge(0.02, p=aug_prob),
    	    augs.RandomJitterCharge(0.02, p=aug_prob),
            augs.SemiRandomCrop(x_max, y_max, 10, 20),
            augs.RandomDropout(0.05, p=aug_prob)
        ])
    
    if aug_type=="newbaseaugsplit":
        return transforms.Compose([
       	    augs.RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            augs.RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
    	    augs.RandomInPlaceHorizontalFlip(p=0.5),
            augs.RandomInPlaceVerticalFlip(p=0.5),
    	    augs.RandomHorizontalFlip(x_max=x_orig, p=0.5),
            augs.RandomVerticalFlip(y_max=y_orig, p=0.5),    
            augs.UnlogCharge(),
            augs.GridJitter(),
            augs.SplitJitterCoords(5),
            augs.RandomShear2D(0.1, 0.1, p=aug_prob),
            augs.RandomRotation2D(6, p=aug_prob),
            augs.RandomStretch2D(0.1, 0.1, p=aug_prob),
    	    augs.RandomGridDistortion2D(100, 10, 2, 25, p=aug_prob),
    	    augs.RandomScaleCharge(0.05, p=aug_prob),
    	    augs.RandomJitterCharge(0.05, p=aug_prob),
    	    augs.BilinearSplatMod(0.3, 0.5, p=0.5),
            augs.RelogCharge(),
            augs.RandomScaleCharge(0.02, p=aug_prob),
    	    augs.RandomJitterCharge(0.02, p=aug_prob),
            augs.SemiRandomCrop(x_max, y_max, 10, 20),
            augs.RandomDropout(0.05, p=aug_prob)
        ])
        
    if aug_type=="newbaseaugnoise":
        return transforms.Compose([
       	    augs.RandomBlockZeroImproved([0, 50], [5,10], [0,x_orig], [0,y_orig], p=aug_prob),
            augs.RandomBlockZeroImproved([500,2000], [1,3], [0,x_orig], [0,y_orig], p=aug_prob),
    	    augs.RandomInPlaceHorizontalFlip(p=0.5),
            augs.RandomInPlaceVerticalFlip(p=0.5),
    	    augs.RandomHorizontalFlip(x_max=x_orig, p=0.5),
            augs.RandomVerticalFlip(y_max=y_orig, p=0.5),    
            augs.UnlogCharge(),
            augs.RandomPixelNoise2D(30),
            augs.GridJitter(),
            augs.JitterCoords(),
            augs.RandomShear2D(0.1, 0.1, p=aug_prob),
            augs.RandomRotation2D(6, p=aug_prob),
            augs.RandomStretch2D(0.1, 0.1, p=aug_prob),
    	    augs.RandomGridDistortion2D(100, 10, 2, 25, p=aug_prob),
    	    augs.RandomScaleCharge(0.05, p=aug_prob),
    	    augs.RandomJitterCharge(0.05, p=aug_prob),
    	    augs.BilinearSplatMod(0.3, 0.5, p=0.5),
            augs.RelogCharge(),
            augs.RandomScaleCharge(0.02, p=aug_prob),
    	    augs.RandomJitterCharge(0.02, p=aug_prob),
            augs.SemiRandomCrop(x_max, y_max, 10, 20),
            augs.RandomDropout(0.05, p=aug_prob)
        ]) 
    
    
    raise ValueError("Unknown augmentation type:", aug_type)

    
