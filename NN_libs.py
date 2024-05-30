from torch import nn
import torch
import sys

### Alternative L2 loss
class AsymmetricL2Loss(nn.Module):
    def __init__(self, nonzero_cost=2.0, zero_cost=1.0, l1_weight=0):
        super(AsymmetricL2Loss, self).__init__()
        self.nonzero_cost = nonzero_cost
        self.zero_cost = zero_cost
        self.l1_weight = l1_weight
    
    def forward(self, predictions, targets, encoder, decoder):
        ## Calculate the absolute difference between predictions and targets
        sq_err = (predictions - targets)**2
        
        ## Calculate the loss for nonzero values
        nonzero = self.nonzero_cost * torch.where(targets != 0, sq_err, torch.zeros_like(sq_err))
        
        ## Calculate the loss for predicting a nonzero value for zero
        zero = self.zero_cost * torch.where(targets == 0, torch.where(predictions != 0, sq_err, torch.zeros_like(sq_err)), torch.zeros_like(sq_err))

        ## Total loss is the sum of nonzero_loss and zero_loss
        total_loss = torch.mean(zero + nonzero)

        ## Add the optional L1 norm term
        if l1_weight != 0:
            l1_norm = torch.tensor(0., device=predictions.device)
            num_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
            for param in encoder.parameters(): l1_norm += torch.norm(param, p=1)
            for param in decoder.parameters(): l1_norm += torch.norm(param, p=1)
            total_loss += self.l1_weight*reco_loss.item()*l1_norm/num_params
            
        return total_loss


## Alternative L1 loss
class AsymmetricL1Loss(nn.Module):
    def __init__(self, nonzero_cost=2.0, zero_cost=1.0, l1_weight=0):
        super(AsymmetricL1Loss, self).__init__()
        self.nonzero_cost = nonzero_cost
        self.zero_cost = zero_cost
        self.l1_weight = l1_weight
    
    def forward(self, predictions, targets, encoder, decoder):
        ## Calculate the absolute difference between predictions and targets
        diff = torch.abs(predictions - targets)
        
        ## Calculate the loss for nonzero values
        nonzero = self.nonzero_cost * torch.where(targets != 0, diff, torch.zeros_like(diff))
        
        ## Calculate the loss for predicting a nonzero value for zero
        zero = self.zero_cost * torch.where(targets == 0, torch.where(predictions != 0, diff, torch.zeros_like(diff)), torch.zeros_like(diff))

        ## Total loss is the sum of nonzero_loss and zero_loss
        total_loss = torch.mean(zero + nonzero)

        ## Add the optional L1 norm term
        if l1_weight != 0:
            l1_norm = torch.tensor(0., device=predictions.device)
            num_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
            for param in encoder.parameters(): l1_norm += torch.norm(param, p=1)
            for param in decoder.parameters(): l1_norm += torch.norm(param, p=1)
            total_loss += self.l1_weight*reco_loss.item()*l1_norm/num_params
            
        return total_loss

    
## Calculate the average distance between pairs in the latent space
class EuclideanDistLoss(torch.nn.Module):
    def __init__(self):
        super(EuclideanDistLoss, self).__init__()
    
    def forward(self, latent1, latent2):
        # Compute the Euclidean distance between each pair of corresponding tensors in the batch
        batch_size = latent1.size(0)
        distances = torch.norm(latent1 - latent2, p=2, dim=1)
        
        # Average the distances over the batch
        loss = distances.mean()
        
        return loss

    
## Simple with dropout and batch norm
class EncoderSimple(nn.Module):
    
    def __init__(self, 
                 n_chan : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU,
                 drop_fract : float = 0.2):
        """
        Inputs:
            - n_chan : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_chan, kernel_size=3, stride=2, padding=1), ## 280x140 ==> 140x70
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=n_chan, out_channels=2*n_chan, kernel_size=3, stride=2, padding=1), ## 140x70 ==> 70x35
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=2*n_chan, out_channels=4*n_chan, kernel_size=3, stride=2, padding=1), ## 35x18
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Dropout(drop_fract)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### Linear section, simple for now
        self.encoder_lin = nn.Sequential(
            nn.Linear(4*n_chan*35*18, 1024),
            act_fn(),      
            nn.Dropout(drop_fract),
            nn.Linear(1024, latent_dim),
        )
        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or \
               isinstance(m, nn.ConvTranspose2d) or \
               isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class DecoderSimple(nn.Module):
    
    def __init__(self, 
                 n_chan : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU):
        """
        Inputs:
            - n_chan : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            act_fn(),
            nn.Linear(1024, 4*n_chan*35*18),
            act_fn()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(4*n_chan, 35, 18))

        self.decoder_conv = nn.Sequential(  
            nn.ConvTranspose2d(in_channels=4*n_chan, out_channels=2*n_chan, kernel_size=3, stride=2, padding=1, output_padding=(1,0)), ## 35x18 ==> 70x35
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(), 
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(), 
            nn.ConvTranspose2d(in_channels=2*n_chan, out_channels=n_chan, kernel_size=3, stride=2, padding=1, output_padding=1), ## 70x35 ==> 140x70
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.ConvTranspose2d(in_channels=n_chan, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1), ## 140x70 ==> 280x140
            act_fn()
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or \
               isinstance(m, nn.ConvTranspose2d) or \
               isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)      

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

## Deep1   
class EncoderDeep1(nn.Module):
    
    def __init__(self, 
                 n_chan : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU,
                 drop_fract : float = 0.2):
        """
        Inputs:
            - n_chan : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ## Note the assumption that the input image has a single channel
            nn.Conv2d(in_channels=1, out_channels=n_chan, kernel_size=3, stride=2, padding=1), ## 280x140 ==> 140x70
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=n_chan, out_channels=2*n_chan, kernel_size=3, stride=2, padding=1), ## 140x70 ==> 70x35
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            nn.Dropout(drop_fract),
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            nn.Dropout(drop_fract),
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, out_channels=4*n_chan, kernel_size=3, stride=2, padding=1), ## 70x35 ==> 35x18
            nn.BatchNorm2d(4*n_chan),
            nn.Dropout(drop_fract),
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, out_channels=4*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(4*n_chan),
            nn.Dropout(drop_fract),
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, out_channels=4*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(4*n_chan),
            nn.Dropout(drop_fract),
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, out_channels=8*n_chan, kernel_size=3, stride=2, padding=1), ## 35x18 ==> 18x9
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),            
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### Linear section, simple for now
        self.encoder_lin = nn.Sequential(
            ## Number of nodes in last layer multiplied by number of pixels in deepest layer
            nn.Linear(8*n_chan*18*9, 1024),
            act_fn(),      
            nn.Dropout(drop_fract),
            nn.Linear(1024, latent_dim),
            act_fn()      
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or \
               isinstance(m, nn.ConvTranspose2d) or \
               isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class DecoderDeep1(nn.Module):
    
    def __init__(self, 
                 n_chan : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU):
        """
        Inputs:
            - n_chan : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            act_fn(),
            nn.Linear(1024, 8*n_chan*18*9),
            act_fn()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(8*n_chan, 18, 9))

        self.decoder_conv = nn.Sequential(  
            nn.ConvTranspose2d(in_channels=8*n_chan, out_channels=4*n_chan, kernel_size=3, stride=2, padding=1, output_padding=(0,1)), ## 18x9 ==> 35x18
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, out_channels=4*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, out_channels=4*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.ConvTranspose2d(in_channels=4*n_chan, out_channels=2*n_chan, kernel_size=3, stride=2, padding=1, output_padding=(1,0)), ## 35x18 ==> 70x35
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(), 
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(), 
            nn.ConvTranspose2d(in_channels=2*n_chan, out_channels=n_chan, kernel_size=3, stride=2, padding=1, output_padding=1), ## 70x35 ==> 140x70
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.ConvTranspose2d(in_channels=n_chan, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1), ## 140x70 ==> 280x140
            act_fn()
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or \
               isinstance(m, nn.ConvTranspose2d) or \
               isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x


## Deep2
class EncoderDeep2(nn.Module):
    
    def __init__(self, 
                 n_chan : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU,
                 drop_fract : float = 0.2):
        """
        Inputs:
            - n_chan : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ## Note the assumption that the input image has a single channel
            nn.Conv2d(in_channels=1, out_channels=n_chan, kernel_size=3, stride=2, padding=1), ## 280x140 ==> 140x70
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=n_chan, out_channels=2*n_chan, kernel_size=3, stride=2, padding=1), ## 140x70 ==> 70x35
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=2*n_chan, out_channels=4*n_chan, kernel_size=3, stride=2, padding=1), ## 70x35 ==> 35x18
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=4*n_chan, out_channels=4*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=4*n_chan, out_channels=4*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, out_channels=8*n_chan, kernel_size=3, stride=2, padding=1), ## 35x18 ==> 18x9
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan, out_channels=8*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=8*n_chan, out_channels=8*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=8*n_chan, out_channels=16*n_chan, kernel_size=3, stride=2, padding=1), ## 18x9 ==> 9x5
            nn.BatchNorm2d(16*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),            
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### Linear section, simple for now
        self.encoder_lin = nn.Sequential(
            ## Number of nodes in last layer multiplied by number of pixels in deepest layer
            nn.Linear(16*n_chan*9*5, 1024),
            act_fn(),      
            nn.Dropout(drop_fract),
            nn.Linear(1024, latent_dim),
            act_fn()      
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or \
               isinstance(m, nn.ConvTranspose2d) or \
               isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class DecoderDeep2(nn.Module):
    
    def __init__(self, 
                 n_chan : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU):
        """
        Inputs:
            - n_chan : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            act_fn(),
            nn.Linear(1024, 16*n_chan*9*5),
            act_fn()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(16*n_chan, 9, 5))

        self.decoder_conv = nn.Sequential(  
            nn.ConvTranspose2d(in_channels=16*n_chan, out_channels=8*n_chan, kernel_size=3, stride=2, padding=1, output_padding=(1,0)), ## 9x5 ==> 18x9
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan, out_channels=8*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan, out_channels=8*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.ConvTranspose2d(in_channels=8*n_chan, out_channels=4*n_chan, kernel_size=3, stride=2, padding=1, output_padding=(0,1)), ## 18x9 ==> 35x18
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, out_channels=4*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, out_channels=4*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.ConvTranspose2d(in_channels=4*n_chan, out_channels=2*n_chan, kernel_size=3, stride=2, padding=1, output_padding=(1,0)), ## 35x18 ==> 70x35
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(), 
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(), 
            nn.ConvTranspose2d(in_channels=2*n_chan, out_channels=n_chan, kernel_size=3, stride=2, padding=1, output_padding=1), ## 70x35 ==> 140x70
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.ConvTranspose2d(in_channels=n_chan, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1), ## 140x70 ==> 280x140
            act_fn()
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or \
               isinstance(m, nn.ConvTranspose2d) or \
               isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

## Deep3
class EncoderDeep3(nn.Module):
    
    def __init__(self, 
                 n_chan : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU,
                 drop_fract : float = 0.2):
        """
        Inputs:
            - n_chan : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ## Note the assumption that the input image has a single channel
            nn.Conv2d(in_channels=1, out_channels=n_chan, kernel_size=3, stride=2, padding=1), ## 280x140 ==> 140x70
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=n_chan, out_channels=2*n_chan, kernel_size=3, stride=2, padding=1), ## 140x70 ==> 70x35
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=2*n_chan, out_channels=4*n_chan, kernel_size=3, stride=2, padding=1), ## 70x35 ==> 35x18
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=4*n_chan, out_channels=4*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, out_channels=4*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=4*n_chan, out_channels=8*n_chan, kernel_size=3, stride=2, padding=1), ## 35x18 ==> 18x9
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=8*n_chan, out_channels=8*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=8*n_chan, out_channels=8*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=8*n_chan, out_channels=16*n_chan, kernel_size=3, stride=2, padding=1), ## 18x9 ==> 9x5
            nn.BatchNorm2d(16*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=16*n_chan, out_channels=16*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(16*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=16*n_chan, out_channels=16*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(16*n_chan),
            act_fn(),
            nn.Dropout(drop_fract),
            nn.Conv2d(in_channels=16*n_chan, out_channels=32*n_chan, kernel_size=3, stride=2, padding=1), ## 9x5 ==> 5x3
            nn.BatchNorm2d(32*n_chan),
            act_fn(), 
            nn.Dropout(drop_fract),
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### Linear section, simple for now
        self.encoder_lin = nn.Sequential(
            ## Number of nodes in last layer multiplied by number of pixels in deepest layer
            nn.Linear(32*n_chan*5*3, 1024),
            act_fn(),      
            nn.Dropout(drop_fract),
            nn.Linear(1024, latent_dim),
            act_fn()      
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or \
               isinstance(m, nn.ConvTranspose2d) or \
               isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class DecoderDeep3(nn.Module):
    
    def __init__(self, 
                 n_chan : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU):
        """
        Inputs:
            - n_chan : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            act_fn(),
            nn.Linear(1024, 32*n_chan*5*3),
            act_fn()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32*n_chan, 5, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32*n_chan, out_channels=16*n_chan, kernel_size=3, stride=2, padding=1, output_padding=(0,0)), ## 5x3 ==> 9x5
            nn.BatchNorm2d(16*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=16*n_chan, out_channels=16*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(16*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=16*n_chan, out_channels=16*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(16*n_chan),
            act_fn(),
            nn.ConvTranspose2d(in_channels=16*n_chan, out_channels=8*n_chan, kernel_size=3, stride=2, padding=1, output_padding=(1,0)), ## 9x5 ==> 18x9
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan, out_channels=8*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan, out_channels=8*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(8*n_chan),
            act_fn(),
            nn.ConvTranspose2d(in_channels=8*n_chan, out_channels=4*n_chan, kernel_size=3, stride=2, padding=1, output_padding=(0,1)), ## 18x9 ==> 35x18
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, out_channels=4*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, out_channels=4*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(4*n_chan),
            act_fn(),
            nn.ConvTranspose2d(in_channels=4*n_chan, out_channels=2*n_chan, kernel_size=3, stride=2, padding=1, output_padding=(1,0)), ## 35x18 ==> 70x35
            nn.BatchNorm2d(2*n_chan),
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(), 
            nn.Conv2d(in_channels=2*n_chan, out_channels=2*n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(2*n_chan),
            act_fn(), 
            nn.ConvTranspose2d(in_channels=2*n_chan, out_channels=n_chan, kernel_size=3, stride=2, padding=1, output_padding=1), ## 70x35 ==> 140x70
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Conv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=3, padding=1), ## No change in size
            nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.ConvTranspose2d(in_channels=n_chan, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1), ## 140x70 ==> 280x140
            act_fn()
        )

        # Initialize weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or \
               isinstance(m, nn.ConvTranspose2d) or \
               isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

    
def get_model(name=None):

    if name == "simple":
        return EncoderSimple, DecoderSimple
    if name == "deep1":
        return EncoderDeep1, DecoderDeep1      
    if name == "deep2":
        return EncoderDeep2, DecoderDeep2
    if name == "deep3":
        return EncoderDeep3, DecoderDeep3    

    print("Unknown model name:", name)
    sys.exit()
