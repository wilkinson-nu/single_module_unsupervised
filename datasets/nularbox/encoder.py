from torch import nn
import torch
import MinkowskiEngine as ME
from core.models.utils import get_act_from_string_ME

class CCEncoderVarDepth(nn.Module):
    def __init__(self, 
                 nchan : int,
                 act_fn : object = ME.MinkowskiSiLU,
                 first_kernel : int = 3,
                 pool : str = None,
                 slow_growth : bool = False,
                 drop_fract : float = 0,
                 depth : int = 6,
                 final_linear : int = None,
                 orig_y : int = 768,
                 orig_x : int = 256
                 ):
        super().__init__()

        if slow_growth:
            self.ch = [nchan, nchan, nchan*2, nchan*2, nchan*4, nchan*4]
        else:
            self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32]
        self.conv_kernel_size = 3
        self.first_kernel_size = first_kernel
        self.pool = pool
        self.drop_fract = drop_fract
        self.depth = depth
        self.final_linear = final_linear
        self.orig_x = orig_x
        self.orig_y = orig_y
        self.act_fn = act_fn
        
        ## Optional pooling
        if self.pool == "max":
            self.global_pool = ME.MinkowskiGlobalMaxPooling()
        elif self.pool == "avg":
            self.global_pool = ME.MinkowskiGlobalAvgPooling()
        else:
            raise ValueError("A pooling layer is required")
        
        ### Convolutional section
        self.encoders = nn.ModuleList()
        self.encoders.append(self.make_cnn1())
        if self.depth >= 2: self.encoders.append(self.make_cnn2())
        if self.depth >= 3: self.encoders.append(self.make_cnn3())
        if self.depth >= 4: self.encoders.append(self.make_cnn4())
        if self.depth >= 5: self.encoders.append(self.make_cnn5())
        if self.depth >= 6: self.encoders.append(self.make_cnn6())

        ## Sort out the last linear layer
        if self.final_linear != None:
            self.fc = nn.Linear(self.ch[self.depth-1], self.final_linear)  
    
    ### Define convolutional blocks outside the init block
    def make_cnn1(self):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=self.ch[0], kernel_size=self.first_kernel_size, stride=2, bias=False, dimension=2), ## 768x256 ==> 384x128
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn2(self):
        return nn.Sequential(
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[1], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 384x128 ==> 192x64
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn3(self):
        return nn.Sequential(
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[2], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 192x64 ==> 96x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn4(self):
        return nn.Sequential(
            ME.MinkowskiBatchNorm(self.ch[2]),
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[3], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 96x32 ==> 48x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn5(self):
        return nn.Sequential(
            ME.MinkowskiBatchNorm(self.ch[3]),
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[4], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 48x16 ==> 24x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn6(self):
        return nn.Sequential(
            ME.MinkowskiBatchNorm(self.ch[4]),
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[5], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 24x8 ==> 12x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            self.act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def get_nchan_instance(self):
        nout = self.ch[self.depth-1]
        if self.final_linear != None:
            nout = self.final_linear
        return nout

    def get_nchan_cluster(self):
        nout = self.ch[self.depth-1]
        if self.final_linear != None:
            nout = self.final_linear
        return nout
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="linear")
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity="linear")
            if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)
                    m.track_running_stats = False
                    
    def forward(self, x, batch_size, return_maps=False):

        ## Loop over encoder layers
        for enc in self.encoders: x = enc(x)

        feats = self.global_pool(x).F
        if self.final_linear: feats = self.fc(feats)
        
        if return_maps: 
            dense_maps,_,_ = x.dense(shape=torch.Size([batch_size, self.ch[self.depth-1], \
                                                       self.orig_y//2**self.depth, self.orig_x//2**self.depth]))
            return dense_maps, feats, feats
        return feats, feats
            
def get_encoder(args):
    
    ## Only one architecture for now
    if args.enc_arch == "d6": ## Should be 12x4
        enc = CCEncoderVarDepth 
        depth = 6
    elif args.enc_arch == "d5": ## Should be 24x8
        enc = CCEncoderVarDepth
        depth = 5
    elif args.enc_arch == "d4": ## Should be 48x16
        enc = CCEncoderVarDepth
        depth = 4
    elif args.enc_arch == "d3": ## Should be 96x32
        enc = CCEncoderVarDepth
        depth =	3
        
    enc_act_fn=get_act_from_string_ME(args.enc_act)
    encoder = enc(nchan=args.nchan, \
                  act_fn=enc_act_fn, \
                  first_kernel=args.enc_arch_first_kernel, \
                  pool=args.enc_arch_pool, \
                  slow_growth=bool(args.enc_arch_slow_growth),
                  final_linear=args.enc_arch_final_linear,
                  drop_fract=args.dropout,
                  depth=depth)
    return encoder
