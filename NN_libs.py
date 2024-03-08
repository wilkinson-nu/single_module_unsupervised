from torch import nn


class Encoder(nn.Module):
    
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
            #nn.BatchNorm2d(n_chan),
            act_fn(),
            nn.Conv2d(in_channels=n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, stride=2, padding=1), ## 140x70 ==> 70x35
            act_fn(),
            nn.Conv2d(in_channels=2*n_chan, 
                      out_channels=2*n_chan, 
                      kernel_size=3, padding=1), ## No change in size
            act_fn()
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### Linear section, simple for now
        ## This is 8960...
        self.encoder_lin = nn.Sequential(
            ## Number of nodes in last layer (16*n_chan) multiplied by number of pixels in deepest layer (4x4)
            nn.Linear(2*n_chan*70*35, latent_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class Decoder(nn.Module):
    
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
            nn.Linear(latent_dim, 2*n_chan*70*35),
            act_fn()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(2*n_chan, 70, 35))

        self.decoder_conv = nn.Sequential(  
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
            #nn.BatchNorm2d(n_chan),
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
        #x = torch.sigmoid(x)
        return x


## Define the encoder and decoders that do the business
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
        ## This is 8960...
        self.encoder_lin = nn.Sequential(
            ## Number of nodes in last layer (16*n_chan) multiplied by number of pixels in deepest layer (4x4)
            nn.Linear(4*n_chan*35*17, n_chan*35*17),
            act_fn(),      
            nn.Linear(n_chan*35*17, latent_dim),
            #act_fn()
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
            nn.Linear(latent_dim, n_chan*35*17),
            act_fn(),
            nn.Linear(n_chan*35*17, 4*n_chan*35*17),
            act_fn()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(4*n_chan, 35, 17))

        self.decoder_conv = nn.Sequential(  
            nn.ConvTranspose2d(in_channels=4*n_chan, 
                               out_channels=2*n_chan, 
                               kernel_size=3, stride=2, padding=1, output_padding=1), ## 8x8 ==> 16x16
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
                               kernel_size=3, stride=2, padding=(1,0), output_padding=1), ## 16x16 ==> 32x32
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
        # x = torch.sigmoid(x)
        return x

    ## Define the encoder and decoders that do the business
from torch import nn
class EncoderComplex(nn.Module):
    
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
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            ## Note the assumption that the input image has a single channel
            nn.Conv2d(in_channels=1, 
                      out_channels=base_channel_size, 
                      kernel_size=5, stride=1, padding=2), ## No change (280x140)
            act_fn(),
            # nn.MaxPool2d(kernel_size=2, stride=2), ## 32x140x70
            nn.Conv2d(in_channels=base_channel_size, 
                      out_channels=base_channel_size*2, 
                      kernel_size=3, stride=2, padding=1), ## 140*70
            act_fn(),
            nn.Conv2d(in_channels=base_channel_size*2, 
                      out_channels=base_channel_size*2, 
                      kernel_size=3, stride=1, padding=1), ## No change
            act_fn(),            
            #nn.MaxPool2d(kernel_size=2, stride=2), ## 16x70x35
            nn.Conv2d(in_channels=base_channel_size*2, 
                      out_channels=base_channel_size*4, 
                      kernel_size=3, stride=2, padding=1), ## 70x35
            act_fn(),
            nn.Conv2d(in_channels=base_channel_size*4, 
                      out_channels=base_channel_size*4, 
                      kernel_size=3, stride=1, padding=1), ## 70x35
            act_fn(),
            # nn.MaxPool2d(kernel_size=2, stride=2), # 8x35x17
            nn.Conv2d(in_channels=base_channel_size*4, 
                      out_channels=base_channel_size*8, 
                      kernel_size=3, stride=2, padding=(1,0)), ## 35*17
            act_fn(),
            nn.Conv2d(in_channels=base_channel_size*8, 
                      out_channels=base_channel_size*8, 
                      kernel_size=3, padding=1, stride=1), ## No change
            act_fn()            
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        self.encoder_lin = nn.Sequential(
            nn.Linear(base_channel_size*8*35*17, latent_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class DecoderComplex(nn.Module):
    
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

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, base_channel_size*8*35*17),
           act_fn()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(base_channel_size*8, 35, 17))      

        self.decoder_conv = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=base_channel_size*8, 
                               out_channels=base_channel_size*4, 
                               kernel_size=3, stride=2, padding=1, output_padding=1), ## 8x70x35
            act_fn(),
            nn.Conv2d(in_channels=base_channel_size*4,
                      out_channels=base_channel_size*4,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),
            nn.ConvTranspose2d(in_channels=base_channel_size*4,
                               out_channels=base_channel_size*2,
                               kernel_size=3, stride=2, padding=(1,0), output_padding=1), ## 16x140x70
            act_fn(),
            nn.Conv2d(in_channels=base_channel_size*2,
                      out_channels=base_channel_size*2,
                      kernel_size=3, padding=1), ## No change in size
            act_fn(),            
            nn.ConvTranspose2d(in_channels=base_channel_size*2, 
                               out_channels=base_channel_size, 
                               kernel_size=3, stride=2, padding=1, output_padding=1), ## 32x280x140
            act_fn(),
            nn.Conv2d(in_channels=base_channel_size, out_channels=1,
                      kernel_size=5, stride=1, padding=2),  # 1x280x140)
            act_fn()
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
