from torch import nn
import torch
import MinkowskiEngine as ME
import math

class ClusteringLossMerged(nn.Module):
    def __init__(self, temperature=0.5, entropy_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.entropy_weight = entropy_weight

    def forward(self, c_cat):
        batch_size = c_cat.shape[0]//2
        class_num = c_cat.shape[1]
        c_i, c_j = c_cat[:batch_size], c_cat[batch_size:]

        negatives_mask = (~torch.eye(batch_size*2, batch_size*2, dtype=bool, device=c_cat.device)).float()
        representations = torch.cat([c_i, c_j], dim=0)
        similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2*batch_size)

        ## Now add the entropy term
        p_i = c_i.mean(dim=0)
        p_j = c_j.mean(dim=0)

        # Compute entropy and normalize by log(K)
        p_i = p_i/p_i.sum()
        p_j = p_j/p_j.sum()

        entropy_i = -torch.sum(p_i * torch.log(p_i + 1e-10))/math.log(class_num)
        entropy_j = -torch.sum(p_j * torch.log(p_j + 1e-10))/math.log(class_num)

        ne_loss = -0.5 * (entropy_i + entropy_j)
        
        return loss + ne_loss

    
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

class NTXentMergedTopTenNeg(nn.Module):
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

        sorted_indices = torch.argsort(similarity_matrix, dim=1)  # Sort similarities in ascending order
        top_10_percent = int(batch_size * 2 * 0.1)
        filtered_mask = torch.zeros_like(negatives_mask)

        for i in range(batch_size * 2):
            # Keep only the top 10% least similar negatives
            top_negatives = sorted_indices[i, :top_10_percent]
            filtered_mask[i, top_negatives] = 1.0

        # Adjust mask to include only top 10% least similar negatives
        final_negatives_mask = negatives_mask * filtered_mask
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = final_negatives_mask * torch.exp(similarity_matrix / self.temperature)

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

## The FSD encoders
class ContrastiveEncoderFSD(nn.Module):
    
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
        
        self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32, nchan*64, nchan*128, nchan*256]
        self.conv_kernel_size = conv_kernel_size
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=self.ch[0], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 512x256 ==> 256x128
            # ME.MinkowskiBatchNorm(self.ch[0]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            # ME.MinkowskiBatchNorm(self.ch[0]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[1], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 256x128 ==> 128x64
            # ME.MinkowskiBatchNorm(self.ch[1]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            # ME.MinkowskiBatchNorm(self.ch[1]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[2], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 128x64 ==> 64x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[3], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 64x32 ==> 32x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[4], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 32x16 ==> 16x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[5], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 16x8 ==> 8x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[6], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 8x4 ==> 4x2
            ME.MinkowskiBatchNorm(self.ch[6]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[6], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[6]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[6], out_channels=self.ch[7], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 4x2 ==> 2x1
            ME.MinkowskiBatchNorm(self.ch[7]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[7], out_channels=self.ch[7], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[7]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[7], out_channels=self.ch[8], kernel_size=(2,1), stride=(2,1), bias=False, dimension=2), ## 2x1 ==> 1x1
            ME.MinkowskiBatchNorm(self.ch[8]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            
        )
        
        ### Linear section, simple for now
        self.encoder_lin = nn.Sequential(
            ME.MinkowskiLinear(self.ch[8], self.ch[4]),
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

class ContrastiveEncoderShallowFSD(nn.Module):
    def __init__(self, 
                 nchan : int,
                 latent_dim : int,
                 hidden_act_fn : object = ME.MinkowskiReLU,
                 latent_act_fn : object = ME.MinkowskiTanh,
                 drop_fract : float = 0,
                 conv_kernel_size=3):
        super().__init__()

        self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32]
        self.conv_kernel_size = conv_kernel_size
        
        # Convolutional section — down to 8x4
        self.encoder_cnn = nn.Sequential(
            ME.MinkowskiConvolution(1, self.ch[0], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), # 512x256 -> 256x128
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[0], self.ch[0], kernel_size=3, bias=False, dimension=2),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[0], self.ch[1], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), # 256x128 -> 128x64
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[1], self.ch[1], kernel_size=3, bias=False, dimension=2),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[1], self.ch[2], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), # 128x64 -> 64x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[2], self.ch[2], kernel_size=3, bias=False, dimension=2),
            ME.MinkowskiBatchNorm(self.ch[2]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[2], self.ch[3], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), # 64x32 -> 32x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[3], self.ch[3], kernel_size=3, bias=False, dimension=2),
            ME.MinkowskiBatchNorm(self.ch[3]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[3], self.ch[4], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), # 32x16 -> 16x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[4], self.ch[4], kernel_size=3, bias=False, dimension=2),
            ME.MinkowskiBatchNorm(self.ch[4]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[4], self.ch[5], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), # 16x8 -> 8x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(self.ch[5], self.ch[5], kernel_size=3, bias=False, dimension=2),
            ME.MinkowskiBatchNorm(self.ch[5]),
            hidden_act_fn(),
            ME.MinkowskiDropout(drop_fract),
        )

        # We'll flatten after this to shape [B, C * 16 * 8]
        self.feature_channels = self.ch[5]
        
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

class CCEncoderFSDGlobal(nn.Module):
    def __init__(self, 
                 nchan : int,
                 act_fn : object = ME.MinkowskiSiLU,
                 drop_fract : float = 0):
        super().__init__()
        
        self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32]
        self.conv_kernel_size = 3
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=self.ch[0], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 768x256 ==> 384x128
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[1], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 384x128 ==> 192x64
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[2], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 192x64 ==> 96x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[3], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 96x32 ==> 48x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[4], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 48x16 ==> 24x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[5], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 24x8 ==> 12x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiGlobalMaxPooling()           
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def get_nchan(self):
        return self.ch[5]
        
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
        x = self.encoder_cnn(x)
        return x.F

class CCEncoderFSD12x4(nn.Module):
    def __init__(self, 
                 nchan : int,
                 act_fn : object = ME.MinkowskiSiLU,
                 drop_fract : float = 0):
        super().__init__()
        
        self.ch = [nchan, nchan*2, nchan*4, nchan*8, nchan*16, nchan*32]
        self.conv_kernel_size = 3
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=self.ch[0], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 768x256 ==> 384x128
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[0], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[0], out_channels=self.ch[1], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 384x128 ==> 192x64
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[1], kernel_size=3, bias=False, dimension=2), ## No change in size
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[1], out_channels=self.ch[2], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 192x64 ==> 96x32
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[2], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[2]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[2], out_channels=self.ch[3], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 96x32 ==> 48x16
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[3], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[3]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[3], out_channels=self.ch[4], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 48x16 ==> 24x8
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[4]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[4], out_channels=self.ch[5], kernel_size=self.conv_kernel_size, stride=2, bias=False, dimension=2), ## 24x8 ==> 12x4
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
            ME.MinkowskiDropout(drop_fract),
            ME.MinkowskiConvolution(in_channels=self.ch[5], out_channels=self.ch[5], kernel_size=3, bias=False, dimension=2), ## No change in size
            ME.MinkowskiBatchNorm(self.ch[5]),
            act_fn(),
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def get_nchan(self):
        return self.ch[5]*4*12
        
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
        x = self.encoder_cnn(x)

        # Convert sparse tensor to dense
        dense,_,_ = x.dense(shape=torch.Size([batch_size, self.ch[5], 12, 4]))

        ## Option to return the feature maps for debugging
        if return_maps: return dense
        
        flat = dense.flatten(start_dim=1)     # [B, C * 12 * 4]
        return flat    


    
# Instance-level
class ProjectionHead(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 hidden_act_fn : object = nn.ReLU,
                 latent_act_fn : object = nn.Tanh):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(nchan, nchan//4),
            hidden_act_fn(),
            nn.Linear(nchan//4, nlatent),
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

# Cluster assignment probabilities
class ClusteringHeadTwoLayer(nn.Module):
    def __init__(self, 
                 nchan : int, 
                 nclusters : int,
                 hidden_act_fn : object = nn.ReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(nchan, nchan//4),
            hidden_act_fn(),
            nn.Linear(nchan//4, nclusters),
            nn.Softmax(dim=1),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.proj(x)
        return x

class ClusteringHeadOneLayer(nn.Module):
    def __init__(self,
                 nchan : int,
                 nclusters : int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(nchan, nclusters),
            nn.Softmax(dim=1),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.proj(x)
        return x
