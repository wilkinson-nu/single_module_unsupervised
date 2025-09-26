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


#### This is for contrastive only training
## These are the original encoders
class ContrastiveEncoderME(nn.Module):
    
    def __init__(self, 
                 nchan : int,
                 latent_dim : int,
                 hidden_act_fn : object = ME.MinkowskiReLU,
                 latent_act_fn : object = ME.MinkowskiTanh,
                 drop_fract : float = 0,
                 conv_kernel_size=3):
        """
        Inputs:
            - nchan : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        
        self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32, nchan*64, nchan*128]
        self.conv_kernel_size = conv_kernel_size
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=self.ch[0], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 256x128 ==> 128x64
            # ME.MinkowskiBatchNorm(self.ch[0]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            # ME.MinkowskiBatchNorm(self.ch[0]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[1], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 128x64 ==> 64x32
            # ME.MinkowskiBatchNorm(self.ch[1]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            # ME.MinkowskiBatchNorm(self.ch[1]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[2], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 64x32 ==> 32x16
            ME.MinkowskiBatchNorm(self.ch[2]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[3], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 32x16 ==> 16x8
            ME.MinkowskiBatchNorm(self.ch[3]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[4], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 16x8 ==> 8x4
            ME.MinkowskiBatchNorm(self.ch[4]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[5], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 8x4 ==> 4x2
            ME.MinkowskiBatchNorm(self.ch[5]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[6], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 4x2 ==> 2x1
            ME.MinkowskiBatchNorm(self.ch[6]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[7], kernel_size=(2,1), stride=(2,1), bias=False, dimension=2), ## 2x1 ==> 1x1
            ME.MinkowskiBatchNorm(self.ch[7]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
        )
        
        ### Linear section, simple for now
        self.encoder_lin = nn.Sequential(
            ME.MinkowskiLinear(self.ch[7], self.ch[4]),
            ME.MinkowskiBatchNorm(self.ch[4]),
            hidden_act_fn(),      
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiLinear(self.ch[4], self.ch[3]),
            ME.MinkowskiBatchNorm(self.ch[3]),
	    hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiLinear(self.ch[3], latent_dim),
            latent_act_fn()
        )
        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution) or \
               isinstance(m, ME.MinkowskiGenerativeConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="linear")
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity="linear")
            if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)
                    m.track_running_stats = False
                    
    def forward(self, x, batch_size):
        x = self.encoder_cnn(x)
        x = self.encoder_lin(x)
        return x

class ContrastiveEncoderShallowME(nn.Module):
    def __init__(self, 
                 nchan : int,
                 latent_dim : int,
                 hidden_act_fn : object = ME.MinkowskiReLU,
                 latent_act_fn : object = ME.MinkowskiTanh,
                 drop_fract : float = 0,
                 conv_kernel_size=3):
        super().__init__()

        self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16]
        self.conv_kernel_size = conv_kernel_size
        
        # Convolutional section — down to 8x4
        self.encoder_cnn = nn.Sequential(
            ME.MinkowskiConvolution(1, self.ch[0], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), # 256x128 -> 128x64
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[0], self.ch[0], kernel_size=3, bias=False, dimension=2),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[0], self.ch[1], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), # 128x64 -> 64x32
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[1], self.ch[1], kernel_size=3, bias=False, dimension=2),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[1], self.ch[2], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), # 64x32 -> 32x16
            ME.MinkowskiBatchNorm(self.ch[2]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[2], self.ch[2], kernel_size=3, bias=False, dimension=2),
            ME.MinkowskiBatchNorm(self.ch[2]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[2], self.ch[3], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), # 32x16 -> 16x8
            ME.MinkowskiBatchNorm(self.ch[3]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[3], self.ch[3], kernel_size=3, bias=False, dimension=2),
            ME.MinkowskiBatchNorm(self.ch[3]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[3], self.ch[4], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), # 16x8 -> 8x4
            ME.MinkowskiBatchNorm(self.ch[4]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[4], self.ch[4], kernel_size=3, bias=False, dimension=2),
            ME.MinkowskiBatchNorm(self.ch[4]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
        )

        # We'll flatten after this to shape [B, C * 16 * 8]
        self.feature_channels = self.ch[4]
        
        # Linear projection head (pure PyTorch)
        self.encoder_lin = nn.Sequential(
            nn.Linear(self.feature_channels*8*4, self.feature_channels),
            nn.BatchNorm1d(self.feature_channels),
            nn.SiLU(),
            nn.Dropout(drop_fract),
            nn.Linear(self.feature_channels, self.feature_channels//4),
            nn.BatchNorm1d(self.feature_channels//4),
            nn.SiLU(),
            nn.Dropout(drop_fract),
            nn.Linear(self.feature_channels//4, latent_dim),
            nn.Tanh()
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="linear")
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, batch_size):
        x = self.encoder_cnn(x)
        # Convert sparse tensor to dense
        dense,_,_ = x.dense(shape=torch.Size([batch_size, self.feature_channels, 8, 4]))
        #  dense = self.to_dense(x)
        flat = dense.flatten(start_dim=1)     # [B, C * 8 * 4]
        out = self.encoder_lin(flat)          # Final embedding
        return out


class CCEncoderFSD12x4Opt(nn.Module):
    def __init__(self, 
                 nchan : int,
                 act_fn : object = ME.MinkowskiSiLU,
                 first_kernel : int = 3,
                 flatten : bool = False,
                 pool : str = None,
                 slow_growth : bool = False,
                 sep_heads : bool = False
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
        self.drop_fract = 0
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
                    
    def forward(self, x, batch_size):

        ## This is always the same, but can choose what to return
        x = self.encoder_cnn(x)

        outputs = []
        
        # Compute the dense tensor if we want to hand this back
        if self.flatten:
            dense,_,_ = x.dense(shape=torch.Size([batch_size, self.ch[5], 12, 4]))
            flat = dense.flatten(start_dim=1)
            outputs.append(flat)

        if self.global_pool is not None:
            glob = self.global_pool(x).F
            outputs.append(glob)

        # Decide return type
        if self.sep_heads:
            return torch.cat(outputs, dim=1), outputs[1]
        else:
            if len(outputs) == 1:
                return outputs[0], outputs[0]
            elif len(outputs) > 1:
                return torch.cat(outputs, dim=1), torch.cat(outputs, dim=1)
            else:
                raise ValueError("Both flat and global are disabled!")

        
    
# Instance-level
class ProjectionHead(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 hidden_act_fn : object = nn.ReLU,
                 latent_act_fn : object = nn.Tanh):
        super().__init__()

        self.middle_layer = max(nchan//4, nlatent)
        
        self.proj = nn.Sequential(
            nn.Linear(nchan, self.middle_layer),
            hidden_act_fn(),
            nn.Linear(self.middle_layer, nlatent),
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
                 hidden_act_fn : object = nn.ReLU):
        super().__init__()

        self.middle_layer = max(nchan//4, nlatent)
        
        self.proj = nn.Sequential(
            nn.Linear(nchan, self.middle_layer),
            hidden_act_fn(),
            nn.Linear(self.middle_layer, nlatent),
        )
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                nn.init.zeros_(m.bias)

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
