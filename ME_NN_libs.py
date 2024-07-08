from torch import nn
import torch
import sys
import MinkowskiEngine as ME

## Calculate the average distance between pairs in the latent space
class EuclideanDistLoss(torch.nn.Module):
    def __init__(self, cutoff=0.1, pressure=10):
        super(EuclideanDistLoss, self).__init__()
        self.cutoff = cutoff
        self.pressure = pressure
        
    def forward(self, latent1, latent2):
        # Compute the Euclidean distance between each pair of corresponding tensors in the batch
        batch_size = latent1.size(0)
        distances = torch.norm(latent1 - latent2, p=2, dim=1)
        mod_penalty = torch.stack([self.calc_penalty(item) for item in distances])
        loss = mod_penalty.mean()
        return loss
        
    def calc_penalty(self, value):
        ## Apply a penalty that is the value-cutoff above the cutoff, and is penalty*(cutoff - value)**2 for values below it
        if value > self.cutoff:
            return (value - self.cutoff)**2
        else: 
            return self.pressure*(self.cutoff - value)**2

    
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
        
        all_C  = torch.cat([pred_C, targ_C], dim=0)

        _, idx, counts = torch.cat([pred_C, targ_C], dim=0).unique(dim=0, return_inverse=True, return_counts=True)
        _, idx, counts = all_C.unique(dim=0, return_inverse=True, return_counts=True)
        
        ## This is the original, but causes synchronization issues (where produces a tensor of unknown size, prompting a synchronization step)
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
        
        ## Split into blocks
        self.block1 = nn.Sequential(
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
            act_fn()
        )
        self.block1_cls = ME.MinkowskiConvolution(
            self.ch[5], 1, kernel_size=1, bias=True, dimension=2
        )
        self.block2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[5], out_channels=self.ch[4], kernel_size=2, stride=2, bias=False, dimension=2), ## 4x4 ==> 8x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn()
        )
        self.block2_cls = ME.MinkowskiConvolution(
            self.ch[4], 1, kernel_size=1, bias=True, dimension=2
        )
        
        self.block3 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[4], out_channels=self.ch[3], kernel_size=2, stride=2, bias=False, dimension=2), ## 8x8 ==> 16x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn()
        )
        self.block3_cls = ME.MinkowskiConvolution(
            self.ch[3], 1, kernel_size=1, bias=True, dimension=2
        )
            
        self.block4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[3], out_channels=self.ch[2], kernel_size=2, stride=2, bias=False, dimension=2), ## 16x16 ==> 32x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn()
        )
        self.block4_cls = ME.MinkowskiConvolution(
            self.ch[2], 1, kernel_size=1, bias=True, dimension=2
        )
        
        self.block5 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[2], out_channels=self.ch[1], kernel_size=2, stride=2, bias=False, dimension=2), ## 32x32 ==> 64x64
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            act_fn()
        )
        self.block5_cls = ME.MinkowskiConvolution(
            self.ch[1], 1, kernel_size=1, bias=True, dimension=2
        )
        
        self.block6 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[1], out_channels=self.ch[0], kernel_size=2, stride=2, bias=False, dimension=2), ## 64x64 ==> 128x128
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[0]),
            act_fn(),          
        )
        self.block6_cls = ME.MinkowskiConvolution(
            self.ch[0], 1, kernel_size=1, bias=True, dimension=2
        )
        self.block7 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=self.ch[0], out_channels=1, kernel_size=(2,1), stride=(2,1), bias=True, dimension=2), ## 128x128 ==> 256x128
            act_fn()
        )

        ## For removing points
        self.pruning = ME.MinkowskiPruning()
        
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
             
    ## This is definitely not working as expected. It only ever gives a single True value...
    @torch.no_grad()
    def get_target(self, out, target_key, kernel_size=1):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(
            target_key,
            out.tensor_stride
        )
        kernel_map = cm.kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1,
        )
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    def forward(self, x, target_key=None):
        x = self.decoder_lin(x)
        x = self.decoder_conv(x)
        return x
    
    def test(self, x):
                
        out1 = self.block1(x)
        out1_cls = self.block1_cls(out1)
        keep1 = (out1_cls.F > 0).squeeze()
        if self.training: 
            target = self.get_target(out1, target_key,1)
            keep1 += target
        out1 = self.pruning(out1, keep1)
        
        out2 = self.block2(out1)
        out2_cls = self.block2_cls(out2)
        keep2 = (out2_cls.F > 0).squeeze()
        if self.training: 
            target = self.get_target(out2, target_key,1)
            keep2 += target
        out2 = self.pruning(out2, keep2)
        
        out3 = self.block3(out2)
        out3_cls = self.block3_cls(out3)
        keep3 = (out3_cls.F > 0).squeeze()
        if self.training: 
            target = self.get_target(out3, target_key,1)
            keep3 += target
        out3 = self.pruning(out3, keep3)
        
        out4 = self.block4(out3)
        out4_cls = self.block4_cls(out4)
        keep4 = (out4_cls.F > 0).squeeze()
        if self.training: 
            target = self.get_target(out4, target_key,1)
            keep4 += target
        out4 = self.pruning(out4, keep4)
        
        out5 = self.block5(out4)
        out5_cls = self.block5_cls(out5)
        keep5 = (out5_cls.F > 0).squeeze()
        if self.training: 
            target = self.get_target(out5, target_key)
            keep5 += target
        out5 = self.pruning(out5, keep5)
        
        out6 = self.block6(out5)
        out6_cls = self.block6_cls(out6)
        keep6 = (out6_cls.F > 0).squeeze()
        if self.training: 
            target = self.get_target(out6, target_key)
            keep6 += target
        out6 = self.pruning(out6, keep6)
        
        out7 = self.block7(out6)
        
        return out7
