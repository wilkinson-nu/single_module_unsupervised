from torch import nn


class EncoderSimple(nn.Module):
    
    def __init__(self, 
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU):
        """
        Inputs:
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        n_chan = base_channel_size
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ## Note the assumption that the input image has a single channel
            nn.Conv2d(in_channels=1, 
                      out_channels=n_chan, 
                      kernel_size=3, stride=2, padding=1), ## 280x140 ==> 140x70
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, stride=2, padding=1), ## 140x70 ==> 70x35
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=4*n_chan, 
                      kernel_size=3, stride=2, padding=(1,0)), ## 35x17
            act_fn()
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### Linear section, simple for now
        self.encoder_lin = nn.Sequential(
            ## Number of nodes in last layer multiplied by number of pixels in deepest layer
            nn.Linear(4*n_chan*35*17, 1000),
            act_fn(),      
            nn.Linear(1000, latent_dim),
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class DecoderSimple(nn.Module):
    
    def __init__(self, 
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU):
        """
        Inputs:
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        n_chan = base_channel_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            act_fn(),
            nn.Linear(1000, 4*n_chan*35*17),
            act_fn()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(4*n_chan, 35, 17))

        self.decoder_conv = nn.Sequential(  
            nn.ConvTranspose2d(in_channels=4*n_chan, 
                               out_channels=2*n_chan, 
                               kernel_size=3, stride=2, padding=(1,0), output_padding=(1,0)), ## 8x8 ==> 16x16
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan,
                      out_channels=2*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(), 
            nn.Conv2d(in_channels=2*n_chan,
                      out_channels=2*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(), 
            nn.ConvTranspose2d(in_channels=2*n_chan, 
                               out_channels=n_chan, 
                               kernel_size=3, stride=2, padding=1, output_padding=1), ## 16x16 ==> 32x32
            act_fn(),
            nn.Conv2d(in_channels=n_chan,
                      out_channels=n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan,
                      out_channels=n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.ConvTranspose2d(in_channels=n_chan, 
                               out_channels=1, 
                               kernel_size=3, stride=2, padding=1, output_padding=1), ## 32x32 ==> 64x64
            act_fn()
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

    
class EncoderDeep(nn.Module):
    
    def __init__(self, 
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU):
        """
        Inputs:
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        n_chan = base_channel_size
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ## Note the assumption that the input image has a single channel
            nn.Conv2d(in_channels=1, 
                      out_channels=n_chan, 
                      kernel_size=3, stride=2, padding=1), ## 280x140 ==> 140x70
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, stride=2, padding=1), ## 140x70 ==> 70x35
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=4*n_chan, 
                      kernel_size=3, stride=2, padding=(1,0)), ## 35x17
            act_fn()
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### Linear section, simple for now
        self.encoder_lin = nn.Sequential(
            ## Number of nodes in last layer multiplied by number of pixels in deepest layer
            nn.Linear(4*n_chan*35*17, 1000),
            act_fn(),      
            nn.Linear(1000, latent_dim),
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class DecoderDeep(nn.Module):
    
    def __init__(self, 
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU):
        """
        Inputs:
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        n_chan = base_channel_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            act_fn(),
            nn.Linear(1000, 4*n_chan*35*17),
            act_fn()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(4*n_chan, 35, 17))

        self.decoder_conv = nn.Sequential(  
            nn.ConvTranspose2d(in_channels=4*n_chan, 
                               out_channels=2*n_chan, 
                               kernel_size=3, stride=2, padding=(1,0), output_padding=(1,0)), ## 8x8 ==> 16x16
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan,
                      out_channels=2*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(), 
            nn.Conv2d(in_channels=2*n_chan,
                      out_channels=2*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.ConvTranspose2d(in_channels=2*n_chan, 
                               out_channels=n_chan, 
                               kernel_size=3, stride=2, padding=1, output_padding=1), ## 16x16 ==> 32x32
            act_fn(),
            nn.Conv2d(in_channels=n_chan,
                      out_channels=n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan,
                      out_channels=n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan,
                      out_channels=n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan,
                      out_channels=n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.ConvTranspose2d(in_channels=n_chan, 
                               out_channels=1, 
                               kernel_size=3, stride=2, padding=1, output_padding=1), ## 32x32 ==> 64x64
            act_fn()
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x



class EncoderDeeper(nn.Module):
    
    def __init__(self, 
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU):
        """
        Inputs:
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        n_chan = base_channel_size
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ## Note the assumption that the input image has a single channel
            nn.Conv2d(in_channels=1, 
                      out_channels=n_chan, 
                      kernel_size=3, stride=2, padding=1), ## 280x140 ==> 140x70
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, stride=2, padding=1), ## 140x70 ==> 70x35
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=4*n_chan, 
                      kernel_size=3, stride=2, padding=(1,0)), ## 70x35 ==> 35x17
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, 
                      out_channels=4*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, 
                      out_channels=4*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, 
                      out_channels=4*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, 
                      out_channels=4*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan, 
                      out_channels=8*n_chan, 
                      kernel_size=3, stride=2, padding=(0,0)), ## 35x17 ==> 17x8
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan, 
                      out_channels=8*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan, 
                      out_channels=8*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan, 
                      out_channels=8*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan, 
                      out_channels=8*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan, 
                      out_channels=16*n_chan, 
                      kernel_size=3, stride=2, padding=(0,1)), ## 17x8 ==> 8x4
            act_fn(),
            
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### Linear section, simple for now
        self.encoder_lin = nn.Sequential(
            ## Number of nodes in last layer multiplied by number of pixels in deepest layer
            nn.Linear(16*n_chan*8*4, 1000),
            act_fn(),   
            nn.Linear(1000, latent_dim),
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class DecoderDeeper(nn.Module):
    
    def __init__(self, 
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.LeakyReLU):
        """
        Inputs:
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        n_chan = base_channel_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            act_fn(),
            nn.Linear(1000, 16*n_chan*8*4),
            act_fn()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(16*n_chan, 8, 4))

        self.decoder_conv = nn.Sequential(  
            nn.ConvTranspose2d(in_channels=16*n_chan, 
                               out_channels=8*n_chan, 
                               kernel_size=3, stride=2, padding=(0,1), output_padding=(0,1)), ## 8x4 ==> 17x8
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan,
                      out_channels=8*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan,
                      out_channels=8*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan,
                      out_channels=8*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=8*n_chan,
                      out_channels=8*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.ConvTranspose2d(in_channels=8*n_chan, 
                               out_channels=4*n_chan, 
                               kernel_size=3, stride=2, padding=(0,0), output_padding=(0,0)), ## 17x8 ==> 35x17
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan,
                      out_channels=4*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan,
                      out_channels=4*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan,
                      out_channels=4*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=4*n_chan,
                      out_channels=4*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.ConvTranspose2d(in_channels=4*n_chan, 
                               out_channels=2*n_chan, 
                               kernel_size=3, stride=2, padding=(1,0), output_padding=(1,0)), ## 35x17 ==> 70x35
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan,
                      out_channels=2*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(), 
            nn.Conv2d(in_channels=2*n_chan,
                      out_channels=2*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(), 
            nn.Conv2d(in_channels=2*n_chan,
                      out_channels=2*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(), 
            nn.Conv2d(in_channels=2*n_chan,
                      out_channels=2*n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(), 
            nn.ConvTranspose2d(in_channels=2*n_chan, 
                               out_channels=n_chan, 
                               kernel_size=3, stride=2, padding=1, output_padding=1), ## 70x35 ==> 140x70
            act_fn(),
            nn.Conv2d(in_channels=n_chan,
                      out_channels=n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan,
                      out_channels=n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan,
                      out_channels=n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.Conv2d(in_channels=n_chan,
                      out_channels=n_chan,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.ConvTranspose2d(in_channels=n_chan, 
                               out_channels=1, 
                               kernel_size=3, stride=2, padding=1, output_padding=1), ## 140x70 ==> 280x140
            act_fn()
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
