from torch import nn
import torch
import MinkowskiEngine as ME

## Calculate the average distance between pairs in the latent space
class EuclideanDistLoss(torch.nn.Module):
    def __init__(self):
        super(EuclideanDistLoss, self).__init__()
        
    def forward(self, latent1, latent2):
        # Compute the Euclidean distance between each pair of corresponding tensors in the batch
        norm_lat1 = nn.functional.normalize(latent1, p=2, dim=1)
        norm_lat2 = nn.functional.normalize(latent2, p=2, dim=1)
        distances = torch.norm(norm_lat1 - norm_lat2, p=2, dim=1)

        mod_penalty = torch.stack([item**2 for item in distances])
        loss = mod_penalty.mean()
        return loss
    
class CosDistLoss(torch.nn.Module):
    def __init__(self):
        super(CosDistLoss, self).__init__()
        
    def forward(self, latent1, latent2):
        norm_lat1 = nn.functional.normalize(latent1, p=2, dim=1)
        norm_lat2 = nn.functional.normalize(latent2, p=2, dim=1)
                
        sim = nn.functional.cosine_similarity(norm_lat1, norm_lat2, dim=1)
        loss = 1 - sim.mean()
        return loss

## class NTXent(torch.nn.Module):
##     def __init__(self, temperature=0.5):
##         super(NTXent, self).__init__()
##         self.temperature = temperature
##         
##     def forward(self, latent1, latent2):
##         batch_size = latent1.shape[0]
##         z_i = nn.functional.normalize(latent1, p=2, dim=1)
##         z_j = nn.functional.normalize(latent2, p=2, dim=1)
##         
##         similarity_matrix = self.calc_similarity_batch(z_i, z_j)
##         mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(similarity_matrix.device)
## 
##         sim_ij = torch.diag(similarity_matrix, batch_size)
##         sim_ji = torch.diag(similarity_matrix, -batch_size)
## 
##         positives = torch.cat([sim_ij, sim_ji], dim=0)
## 
##         nominator = torch.exp(positives / self.temperature)
##         denominator = mask*torch.exp(similarity_matrix / self.temperature)
## 
##         all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
##         loss = torch.sum(all_losses) / (2 * self.batch_size)
##         return loss
    
class NTXent(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        batch_size = emb_i.shape[0]
        z_i = nn.functional.normalize(emb_i, dim=1)
        z_j = nn.functional.normalize(emb_j, dim=1)
        
        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool, device=emb_i.device)).float()
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        
        return loss

    
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

    
## This is a loss function to deweight the penalty for getting blank pixels wrong
class AsymmetricL2LossME(torch.nn.Module):
    def __init__(self, nonzero_cost=2.0, zero_cost=1.0, batch_size=512):
        super(AsymmetricL2LossME, self).__init__()
        self.nonzero_cost = nonzero_cost
        self.zero_cost = zero_cost
        self.batch_size = batch_size
    
    def forward(self, pred, targ):
        #diff = pred - targ
        #loss = torch.sum(diff.F**2)
        #return loss/512/128/256
        
        # Extract coordinates and features from both sparse tensors
        pred_C = pred.C
        pred_F = pred.F
    
        targ_C = targ.C
        targ_F = targ.F
        
        _, idx, counts = torch.cat([pred_C, targ_C], dim=0).unique(dim=0, return_inverse=True, return_counts=True)
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
        
        return (common+only_p+only_t)/(self.batch_size*128*256)

## This is a relic
class ProjectionHead(nn.Module):
    def __init__(self, dim, act_fn=ME.MinkowskiReLU):
        super(ProjectionHead, self).__init__()

        self.linear_proj = nn.Sequential(
            ME.MinkowskiLinear(dim[0], dim[1], bias=False),
            ME.MinkowskiBatchNorm(dim[1]),
            act_fn(), 
            ME.MinkowskiLinear(dim[1], dim[2], bias=False),
            ME.MinkowskiBatchNorm(dim[2]),
            act_fn(), 
            ME.MinkowskiLinear(dim[2], dim[3], bias=False),
            act_fn(), 
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiLinear):
                ## use xavier because it's a bounded activation function
                # ME.utils.xavier_normal_(m.linear.weight)
                ME.utils.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='relu')   
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

        
    def forward(self, x):
        x = self.linear_proj(x)
        return x


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
                    
    def forward(self, x):
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
            nn.Linear(self.feature_channels*8*4, self.feature_channels*2),
            nn.SiLU(),
            nn.Dropout(drop_fract),
            nn.Linear(self.feature_channels*2, self.feature_channels),
            nn.SiLU(),
            nn.Dropout(drop_fract),
            nn.Linear(self.feature_channels, latent_dim),
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
        dense,_,_ = x.dense(torch.Size([batch_size, self.feature_channels, 8, 4]))
        #  dense = self.to_dense(x)
        flat = dense.flatten(start_dim=1)     # [B, C * 8 * 4]
        out = self.encoder_lin(flat)          # Final embedding
        return out