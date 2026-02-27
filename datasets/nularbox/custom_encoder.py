from torch import nn
import torch
import MinkowskiEngine as ME

class CCEncoderVarDepth(nn.Module):
    def __init__(self, 
                 nchan : int,
                 act_fn : object = ME.MinkowskiSiLU,
                 first_kernel : int = 3,
                 pool : str = None,
                 slow_growth : bool = False,
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
        self.depth = depth
        self.final_linear = final_linear
        self.orig_x = orig_x
        self.orig_y = orig_y
        self.act_fn = act_fn
        
        ## Optional pooling
        ## if self.pool == "max":
        ##     self.global_pool = ME.MinkowskiGlobalMaxPooling()
        ## elif self.pool == "avg":
        ##     self.global_pool = ME.MinkowskiGlobalAvgPooling()
        ## else:
        ##     raise ValueError("A pooling layer is required")

        ## Give max and average pooling
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        
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
            self.fc = nn.Linear(2*(self.ch[self.depth-1]), self.final_linear)  
    
    ### Define convolutional blocks outside the init block
    def make_cnn1(self):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=self.ch[0], kernel_size=self.first_kernel_size, stride=2, bias=False, dimension=2), ## 768x256 ==> 384x128
            #ME.MinkowskiBatchNorm(self.ch[0]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            #ME.MinkowskiBatchNorm(self.ch[0]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn2(self):
        return nn.Sequential(
            #ME.MinkowskiBatchNorm(self.ch[0]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[1], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 384x128 ==> 192x64
            ME.MinkowskiBatchNorm(self.ch[1]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn3(self):
        return nn.Sequential(
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[2], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 192x64 ==> 96x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn4(self):
        return nn.Sequential(
            ME.MinkowskiBatchNorm(self.ch[2]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[3], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 96x32 ==> 48x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn5(self):
        return nn.Sequential(
            ME.MinkowskiBatchNorm(self.ch[3]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[4], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 48x16 ==> 24x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn6(self):
        return nn.Sequential(
            ME.MinkowskiBatchNorm(self.ch[4]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[5], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 24x8 ==> 12x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            self.act_fn(),
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
            elif isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity="linear")
            elif isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)
                    m.track_running_stats = False
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                    
    def forward(self, x, batch_size, return_maps=False):

        ## Loop over encoder layers
        for enc in self.encoders: x = enc(x)

        # feats = self.global_pool(x).F

        ## Could add two sets of pooling
        #avg = MinkowskiGlobalAvgPooling()
        #max = MinkowskiGlobalMaxPooling()
        #feat = torch.cat([avg, max], dim=1)

        avg_pool = self.global_avg_pool(x).F
        max_pool = self.global_max_pool(x).F

        feats = torch.cat([avg_pool, max_pool], dim=1)
        
        if self.final_linear: feats = self.fc(feats)
        
        if return_maps: 
            dense_maps,_,_ = x.dense(shape=torch.Size([batch_size, self.ch[self.depth-1], \
                                                       self.orig_y//2**self.depth, self.orig_x//2**self.depth]))
            return dense_maps, feats, feats
        return feats, feats


class CCEncoderVarDepthBIG(nn.Module):
    def __init__(self, 
                 nchan : int,
                 act_fn : object = ME.MinkowskiSiLU,
                 first_kernel : int = 3,
                 pool : str = None,
                 slow_growth : bool = False,
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
        self.depth = depth
        self.final_linear = final_linear
        self.orig_x = orig_x
        self.orig_y = orig_y
        self.act_fn = act_fn
        
        ## Optional pooling
        ## if self.pool == "max":
        ##     self.global_pool = ME.MinkowskiGlobalMaxPooling()
        ## elif self.pool == "avg":
        ##     self.global_pool = ME.MinkowskiGlobalAvgPooling()
        ## else:
        ##     raise ValueError("A pooling layer is required")

        ## Give max and average pooling
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        
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
            self.fc = nn.Linear(2*(self.ch[self.depth-1]), self.final_linear)  
    
    ### Define convolutional blocks outside the init block
    def make_cnn1(self):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=self.ch[0], kernel_size=self.first_kernel_size, stride=2, bias=False, dimension=2), ## 768x256 ==> 384x128
            #ME.MinkowskiBatchNorm(self.ch[0]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            #ME.MinkowskiBatchNorm(self.ch[0]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in siz
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in siz
        )

    def make_cnn2(self):
        return nn.Sequential(
            #ME.MinkowskiBatchNorm(self.ch[0]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[1], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 384x128 ==> 192x64
            ME.MinkowskiBatchNorm(self.ch[1]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[1]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn3(self):
        return nn.Sequential(
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[2], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 192x64 ==> 96x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn4(self):
        return nn.Sequential(
            ME.MinkowskiBatchNorm(self.ch[2]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[3], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 96x32 ==> 48x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn5(self):
        return nn.Sequential(
            ME.MinkowskiBatchNorm(self.ch[3]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[4], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 48x16 ==> 24x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
        )

    def make_cnn6(self):
        return nn.Sequential(
            ME.MinkowskiBatchNorm(self.ch[4]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[5], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 24x8 ==> 12x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            self.act_fn(),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            self.act_fn(),
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
            elif isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity="linear")
            elif isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)
                    m.track_running_stats = False
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                    
    def forward(self, x, batch_size, return_maps=False):

        ## Loop over encoder layers
        for enc in self.encoders: x = enc(x)

        # feats = self.global_pool(x).F

        ## Could add two sets of pooling
        #avg = MinkowskiGlobalAvgPooling()
        #max = MinkowskiGlobalMaxPooling()
        #feat = torch.cat([avg, max], dim=1)

        avg_pool = self.global_avg_pool(x).F
        max_pool = self.global_max_pool(x).F

        feats = torch.cat([avg_pool, max_pool], dim=1)
        
        if self.final_linear: feats = self.fc(feats)
        
        if return_maps: 
            dense_maps,_,_ = x.dense(shape=torch.Size([batch_size, self.ch[self.depth-1], \
                                                       self.orig_y//2**self.depth, self.orig_x//2**self.depth]))
            return dense_maps, feats, feats
        return feats, feats
