from torch import nn
import torch
import MinkowskiEngine as ME
import math

class ClusteringLossMerged(nn.Module):
    def __init__(self, temperature=0.5, entropy_weight=1.0, match_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.match_weight = match_weight

    def forward(self, c_cat):

        batch_size = c_cat.shape[0]//2
        class_num = c_cat.shape[1]
        c_i, c_j = c_cat[:batch_size], c_cat[batch_size:]

        ## Start with the entropy term
        p_i = c_i.sum(dim=0)
        p_j = c_j.sum(dim=0)
        p_i = p_i/p_i.sum()
        p_j = p_j/p_j.sum()

        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i + 1e-10)).sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j + 1e-10)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        
        negatives_mask = (~torch.eye(class_num*2, class_num*2, dtype=bool, device=c_cat.device)).float()
        representations = torch.cat([c_i, c_j], dim=0)
        similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, class_num)
        sim_ji = torch.diag(similarity_matrix, -class_num)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2*class_num)

        return loss, ne_loss*self.entropy_weight

    
class NTXentMerged(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_cat):
        """
        emb_cat are the concatenated batches of pairs emb_cat = z_i + z_j
        """
        batch_size = emb_cat.shape[0]//2
        z_cat = nn.functional.normalize(emb_cat, dim=1)
        z_i, z_j = z_cat[:batch_size], z_cat[batch_size:]

        negatives_mask = (~torch.eye(batch_size*2, batch_size*2, dtype=bool, device=emb_cat.device)).float()
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2*batch_size)

        return loss

class CCEncoderVarDepth(nn.Module):
    def __init__(self, 
                 nchan : int,
                 act_fn : object = ME.MinkowskiSiLU,
                 first_kernel : int = 3,
                 flatten : bool = False,
                 pool : str = None,
                 slow_growth : bool = False,
                 sep_heads : bool = False,
                 drop_fract : float = 0,
                 depth : int = 6,
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
        self.flatten = flatten
        self.pool = pool
        self.drop_fract = drop_fract
        self.sep_heads = sep_heads
        self.depth = depth
        self.orig_x = orig_x
        self.orig_y = orig_y
        self.act_fn = act_fn
        
        ## Optional pooling
        if self.pool == "max":
            self.global_pool = ME.MinkowskiGlobalMaxPooling()
        elif self.pool == "avg":
            self.global_pool = ME.MinkowskiGlobalAvgPooling()
        else:
            self.global_pool = None

        ## Error checking the config
        if self.sep_heads:
            if self.global_pool == None or self.flatten == False:
                raise ValueError("To use sep_heads, you need to have both pooling and flattening enabled!")        
        
        ### Convolutional section
        self.encoders = nn.ModuleList()
        self.encoders.append(self.make_cnn1())
        if self.depth >= 2: self.encoders.append(self.make_cnn2())
        if self.depth >= 3: self.encoders.append(self.make_cnn3())
        if self.depth >= 4: self.encoders.append(self.make_cnn4())
        if self.depth >= 5: self.encoders.append(self.make_cnn5())
        if self.depth >= 6: self.encoders.append(self.make_cnn6())


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
            ME.MinkowskiBatchNorm(self.ch[2]),
        )

    def make_cnn4(self):
        return nn.Sequential(
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
            ME.MinkowskiBatchNorm(self.ch[3]),
        )

    def make_cnn5(self):
        return nn.Sequential(
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
            ME.MinkowskiBatchNorm(self.ch[4]),
        )

    def make_cnn6(self):
        return nn.Sequential(
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
            ME.MinkowskiBatchNorm(self.ch[5]),
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def get_nchan_instance(self):
        nout = 0
        if self.sep_heads:
            nout += self.ch[self.depth-1]*(self.orig_y//2**self.depth)*(self.orig_x//2**self.depth) + self.ch[self.depth-1]
        else:
            if self.flatten:
                nout += self.ch[self.depth-1]*(self.orig_y//2**self.depth)*(self.orig_x//2**self.depth)
            if self.global_pool is not None:
                nout += self.ch[self.depth-1]
        return nout

    def get_nchan_cluster(self):
        nout = 0
        if self.sep_heads:
            nout += self.ch[self.depth-1]
        else:
            if self.flatten:
                nout += self.ch[self.depth-1]*(self.orig_y//2**self.depth)*(self.orig_x//2**self.depth)
            if self.global_pool is not None:
                nout += self.ch[self.depth-1]
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

        outputs = []

        dense_maps = None
        if return_maps or self.flatten:
            dense_maps,_,_ = x.dense(shape=torch.Size([batch_size, self.ch[self.depth-1], \
                                                       self.orig_y//2**self.depth, self.orig_x//2**self.depth]))
        
        # Compute the dense tensor if we want to hand this back
        if self.flatten:
            flat = dense_maps.flatten(start_dim=1)
            outputs.append(flat)

        if self.global_pool is not None:
            glob = self.global_pool(x).F
            outputs.append(glob)

        ## Check something sensible has happened
        if len(outputs) == 0: raise ValueError("Both flat and global are disabled!")
        head_feats = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
        
        # Decide return type
        if self.sep_heads:
            if return_maps:
                return dense_maps, head_feats, outputs[1]
            return head_feats, outputs[1]
        else:
            if return_maps:
                return dense_maps, head_feats, head_feats
            return head_feats, head_feats


class CCEncoderFSD12x4Opt(nn.Module):
    def __init__(self, 
                 nchan : int,
                 act_fn : object = ME.MinkowskiSiLU,
                 first_kernel : int = 3,
                 flatten : bool = False,
                 pool : str = None,
                 slow_growth : bool = False,
                 sep_heads : bool = False,
                 drop_fract : float = 0
                 ):
        super().__init__()

        if slow_growth:
            self.ch = [nchan, nchan, nchan*2, nchan*2, nchan*4, nchan*4]
        else:
            self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32]
        self.conv_kernel_size = 3
        self.first_kernel_size = first_kernel
        self.flatten = flatten
        self.pool = pool
        self.drop_fract = drop_fract
        self.sep_heads = sep_heads

        
        ## Optional pooling
        if self.pool == "max":
            self.global_pool = ME.MinkowskiGlobalMaxPooling()
        elif self.pool == "avg":
            self.global_pool = ME.MinkowskiGlobalAvgPooling()
        else:
            self.global_pool = None

        ## Error checking the config
        if self.sep_heads:
            if self.global_pool == None or self.flatten == False:
                raise ValueError("To use sep_heads, you need to have both pooling and flattening enabled!")        
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=self.ch[0], kernel_size=self.first_kernel_size, stride=2, bias=False, dimension=2), ## 768x256 ==> 384x128
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[1], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 384x128 ==> 192x64
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[2], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 192x64 ==> 96x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[3], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 96x32 ==> 48x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[4], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 48x16 ==> 24x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[5], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 24x8 ==> 12x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            # act_fn(),
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def get_nchan_instance(self):
        nout = 0
        if self.sep_heads:
            nout += self.ch[5]*12*4 + self.ch[5]
        else:
            if self.flatten:
                nout += self.ch[5]*12*4
            if self.global_pool is not None:
                nout += self.ch[5]
        return nout

    def get_nchan_cluster(self):
        nout = 0
        if self.sep_heads:
            nout += self.ch[5]
        else:
            if self.flatten:
                nout += self.ch[5]*12*4
            if self.global_pool is not None:
                nout += self.ch[5]
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

        ## This is always the same, but can choose what to return
        x = self.encoder_cnn(x)

        outputs = []

        dense_maps = None
        if return_maps or self.flatten:
            dense_maps,_,_ = x.dense(shape=torch.Size([batch_size, self.ch[5], 12, 4]))
        
        # Compute the dense tensor if we want to hand this back
        if self.flatten:
            flat = dense_maps.flatten(start_dim=1)
            outputs.append(flat)

        if self.global_pool is not None:
            glob = self.global_pool(x).F
            outputs.append(glob)

        ## Check something sensible has happened
        if len(outputs) == 0: raise ValueError("Both flat and global are disabled!")
        head_feats = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
        
        # Decide return type
        if self.sep_heads:
            if return_maps:
                return dense_maps, head_feats, outputs[1]
            return head_feats, outputs[1]
        else:
            if return_maps:
                return dense_maps, head_feats, head_feats
            return head_feats, head_feats


class CCEncoderFSD24x8Opt(nn.Module):
    def __init__(self, 
                 nchan : int,
                 act_fn : object = ME.MinkowskiSiLU,
                 first_kernel : int = 3,
                 flatten : bool = False,
                 pool : str = None,
                 slow_growth : bool = False,
                 sep_heads : bool = False,
                 drop_fract : float = 0
                 ):
        super().__init__()

        if slow_growth:
            self.ch = [nchan, nchan, nchan*2, nchan*2, nchan*4]
        else:
            self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16]
        self.conv_kernel_size = 3
        self.first_kernel_size = first_kernel
        self.flatten = flatten
        self.pool = pool
        self.drop_fract = drop_fract
        self.sep_heads = sep_heads

        
        ## Optional pooling
        if self.pool == "max":
            self.global_pool = ME.MinkowskiGlobalMaxPooling()
        elif self.pool == "avg":
            self.global_pool = ME.MinkowskiGlobalAvgPooling()
        else:
            self.global_pool = None

        ## Error checking the config
        if self.sep_heads:
            if self.global_pool == None or self.flatten == False:
                raise ValueError("To use sep_heads, you need to have both pooling and flattening enabled!")        
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=self.ch[0], kernel_size=self.first_kernel_size, stride=2, bias=False, dimension=2), ## 768x256 ==> 384x128
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[1], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 384x128 ==> 192x64
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[2], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 192x64 ==> 96x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[3], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 96x32 ==> 48x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[4], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 48x16 ==> 24x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(self.drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def get_nchan_instance(self):
        nout = 0
        if self.sep_heads:
            nout += self.ch[4]*24*8 + self.ch[4]
        else:
            if self.flatten:
                nout += self.ch[4]*24*8
            if self.global_pool is not None:
                nout += self.ch[4]
        return nout

    def get_nchan_cluster(self):
        nout = 0
        if self.sep_heads:
            nout += self.ch[4]
        else:
            if self.flatten:
                nout += self.ch[4]*24*8
            if self.global_pool is not None:
                nout += self.ch[4]
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

        ## This is always the same, but can choose what to return
        x = self.encoder_cnn(x)

        outputs = []

        dense_maps = None
        if return_maps or self.flatten:
            dense_maps,_,_ = x.dense(shape=torch.Size([batch_size, self.ch[4], 24, 8]))
        
        # Compute the dense tensor if we want to hand this back
        if self.flatten:
            flat = dense_maps.flatten(start_dim=1)
            outputs.append(flat)

        if self.global_pool is not None:
            glob = self.global_pool(x).F
            outputs.append(glob)

        ## Check something sensible has happened
        if len(outputs) == 0: raise ValueError("Both flat and global are disabled!")
        head_feats = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
        
        # Decide return type
        if self.sep_heads:
            if return_maps:
                return dense_maps, head_feats, outputs[1]
            return head_feats, outputs[1]
        else:
            if return_maps:
                return dense_maps, head_feats, head_feats
            return head_feats, head_feats
        
    
# Instance-level
class ProjectionHead(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 nhidden: int = -1,
                 hidden_act_fn : object = nn.ReLU,
                 latent_act_fn : object = nn.Tanh):
        super().__init__()

        ## Slightly dodgy to retain previous default behaviour
        self.hidden = nhidden if nhidden != -1 else nchan//4
        
        self.proj = nn.Sequential(
            nn.Linear(nchan, self.hidden),
            hidden_act_fn(),
            nn.Linear(self.hidden, nlatent),
            latent_act_fn(),
        )
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.proj(x)


class ProjectionHeadLogits(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 nhidden: int = -1,
                 hidden_act_fn : object = nn.ReLU):
        super().__init__()

        ## Slightly dodgy to retain previous default behaviour
        self.hidden = nhidden if nhidden != -1 else nchan//4
        
        self.proj = nn.Sequential(
            nn.Linear(nchan, self.hidden),
            hidden_act_fn(),
            nn.Linear(self.hidden, nlatent),
        )
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.proj(x)


class ProjectionHeadLogitsBN(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 nhidden: int = -1,
                 hidden_act_fn : object = nn.ReLU):
        super().__init__()

        ## Slightly dodgy to retain previous default behaviour
        self.hidden = nhidden if nhidden != -1 else nchan//4

        self.proj = nn.Sequential(
            nn.Linear(nchan, self.hidden, bias=False),
            nn.BatchNorm1d(self.hidden),
            hidden_act_fn(),
            nn.Linear(self.hidden, nlatent, bias=True),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.proj(x)

    
class ProjectionHeadOneLogits(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int):
        super().__init__()

        self.proj = nn.Linear(nchan, nlatent, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.proj(x)
    
    
# Cluster assignment probabilities
class ClusteringHeadTwoLayer(nn.Module):
    def __init__(self, 
                 nchan : int, 
                 nclusters : int,
                 softmax_temp : float,
                 hidden_act_fn : object = nn.ReLU):
        super().__init__()

        self.middle_layer = max(nchan//2, nclusters*2)
        self.softmax_temp = softmax_temp
        
        self.proj = nn.Sequential(
            nn.Linear(nchan, self.middle_layer),
            hidden_act_fn(),
            nn.Linear(self.middle_layer, nclusters),
        )
        self.initialize_weights()
        self.softmax = nn.Softmax(dim=1)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.proj(x)
        x = self.softmax(x/self.softmax_temp)
        return x

    
class ClusteringHeadTwoLayerBN(nn.Module):
    def __init__(self,
                 nchan : int,
                 nclusters : int,
                 nhidden: int = -1,
                 softmax_temp : float = 1.0,
                 hidden_act_fn : object = nn.ReLU):
        super().__init__()

        ## Slightly dodgy to retain previous default behaviour
        self.hidden = nhidden if nhidden != -1 else nchan//2

        self.softmax_temp = softmax_temp

        self.proj = nn.Sequential(
            nn.Linear(nchan, self.hidden, bias=False),
            nn.BatchNorm1d(self.hidden),
            hidden_act_fn(),
            nn.Linear(self.hidden, nclusters, bias=True),
        )
        self.initialize_weights()
        self.softmax = nn.Softmax(dim=1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.proj(x)
        x = self.softmax(x/self.softmax_temp)
        return x
    

    
class ClusteringHeadOneLayer(nn.Module):
    def __init__(self,
                 nchan : int,
                 nclusters : int,
                 softmax_temp : float):
        super().__init__()

        self.softmax_temp = softmax_temp
        self.linear = nn.Linear(nchan, nclusters)
        self.softmax = nn.Softmax(dim=1)        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x/self.softmax_temp)
        return x

def get_act_from_string_ME(act_name):
    if act_name == "relu":
        return ME.MinkowskiReLU
    if act_name == "leakyrelu":
        return ME.MinkowskiLeakyReLU
    if act_name == "gelu":
        return ME.MinkowskiGELU
    if act_name in ["silu", "swish"]:
        return ME.MinkowskiSiLU
    if act_name == "selu":
        return ME.MinkowskiSELU
    if act_name == "tanh":
        return ME.MinkowskiTanh
    if act_name == "softsign":
        return ME.MinkowskiSoftsign
    return None

def get_act_from_string(act_name):
    if act_name == "relu":
        return nn.ReLU
    if act_name == "leakyrelu":
        return nn.LeakyReLU
    if act_name == "gelu":
        return nn.GELU
    if act_name in ["silu", "swish"]:
        return nn.SiLU
    if act_name == "selu":
        return nn.SELU
    if act_name == "tanh":
        return nn.Tanh
    if act_name == "softsign":
        return nn.Softsign
    return None


def get_encoder(args):
    
    ## Only one architecture for now
    depth = 0
    if args.enc_arch == "12x4":
        enc = CCEncoderFSD12x4Opt
    elif args.enc_arch == "24x8":
        enc = CCEncoderFSD24x8Opt
    elif args.enc_arch == "d6": ## Should be 12x4
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
    if depth == 0:
        encoder = enc(nchan=args.nchan, \
                      act_fn=enc_act_fn, \
                      first_kernel=args.enc_arch_first_kernel, \
                      flatten=bool(args.enc_arch_flatten), \
                      pool=args.enc_arch_pool, \
                      slow_growth=bool(args.enc_arch_slow_growth),
                      sep_heads=bool(args.enc_arch_sep_heads),
                      drop_fract=args.dropout)
    else:
        encoder = enc(nchan=args.nchan, \
                      act_fn=enc_act_fn, \
                      first_kernel=args.enc_arch_first_kernel, \
                      flatten=bool(args.enc_arch_flatten), \
                      pool=args.enc_arch_pool, \
                      slow_growth=bool(args.enc_arch_slow_growth),
                      sep_heads=bool(args.enc_arch_sep_heads),
                      drop_fract=args.dropout,
                      depth=depth)
    return encoder

def get_projhead(nchan, args):
    hidden_act_fn = get_act_from_string(args.enc_act)
    latent_act_fn=nn.Tanh
    if args.proj_arch == "logits":
        proj_head = ProjectionHeadLogits(nchan, args.latent, getattr(args, "nhidden", -1), hidden_act_fn)
    elif args.proj_arch == "logitsbn":
        proj_head = ProjectionHeadLogitsBN(nchan, args.latent, getattr(args, "nhidden", -1), hidden_act_fn)
    elif args.proj_arch == "one":
        proj_head = ProjectionHeadOneLogits(nchan, args.latent)
    else:
        proj_head = ProjectionHead(nchan, args.latent, getattr(args, "nhidden", -1), hidden_act_fn, latent_act_fn)
    return proj_head

def get_clusthead(nchan, args):

    hidden_act_fn = get_act_from_string(args.enc_act)
    if args.clust_arch == "none":
        clust_head = None
    elif args.clust_arch == "one":
        clust_head = ClusteringHeadOneLayer(nchan, args.nclusters, args.softmax_temp)
    elif args.clust_arch == "twobn":
        clust_head = ClusteringHeadTwoLayerBN(nchan, args.nclusters, getattr(args, "nhidden", -1), args.softmax_temp, hidden_act_fn)
    else:
        clust_head = ClusteringHeadTwoLayer(nchan, args.nclusters, args.softmax_temp)
    return clust_head
