from torch import nn
import torch
import MinkowskiEngine as ME
from datasets.nularbox.resnetv2_blocks import PreActBasicBlock, PreActBottleneck
from collections import OrderedDict

## This is taken from the MinkowskiEngine implementation, but modified for v2 blocks
## Some inspiration + error checking against https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    ## In channels = 1 and outchannels are set by the network
    def __init__(self, stem_pool=False, stem_norm=False, D=2):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.stem_pool = stem_pool
        self.stem_norm = stem_norm
        self.network_initialization(D)
        self.weight_initialization()

        
    def make_stem(self):
        
        stem = OrderedDict()
        ch = (self.INIT_DIM//2, self.INIT_DIM)
        if self.stem_pool: ch = (self.INIT_DIM, self.INIT_DIM)
        
        ## As is common for ResNet implementations, use 3 3x3 convoutions instead of an initial 7x7 one
        stem['conv1'] = ME.MinkowskiConvolution(in_channels=1, out_channels=ch[0], kernel_size=3, stride=2, dimension=self.D)
        if self.stem_norm: stem['norm1'] = ME.MinkowskiInstanceNorm(ch[0]),
        stem['relu1'] = ME.MinkowskiReLU(inplace=True)
        stem['conv2'] = ME.MinkowskiConvolution(in_channels=ch[0], out_channels=ch[0], kernel_size=3, stride=1, dimension=self.D)
        if self.stem_norm: stem['norm2'] = ME.MinkowskiInstanceNorm(ch[0]),
        stem['relu2'] = ME.MinkowskiReLU(inplace=True)
        stem['conv3'] = ME.MinkowskiConvolution(in_channels=ch[0], out_channels=ch[0], kernel_size=3, stride=1, dimension=self.D)
        if self.stem_norm: stem['norm3'] = ME.MinkowskiInstanceNorm(ch[0]),
        stem['relu3'] = ME.MinkowskiReLU(inplace=True)
        
        ## Option of having a pooling layer or an extra downsampling convolution
        if self.stem_pool:
            stem['pool'] = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=self.D)
        else:
            stem['conv4'] = ME.MinkowskiConvolution(in_channels=ch[0], out_channels=ch[1], kernel_size=3, stride=2, dimension=self.D)

        return nn.Sequential(stem)

    
    def network_initialization(self, D):

        self.inplanes = self.INIT_DIM
        self.stem = self.make_stem()

        ## In the original ME implementation, layer 1 has stride 2, which is nonstandard
        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=1
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2
        )

        ## Note that I removed the conv5 from the original ME implementation, there was no need for such an aggressive downsampling for this purpose
        ## So removed for simplicity        
        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)


    def _make_layer(self, block, planes, num_blocks, stride, dilation=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, dimension=self.D))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, batch_size=None):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.glob_pool(x)
        return x.F

    ## These are relics based on how things used to work, but... okay...
    def get_nchan_instance(self):
        return self.BLOCK.expansion * self.PLANES[-1]
    def get_nchan_cluster(self):
        return self.BLOCK.expansion * self.PLANES[-1]    

    
class ResNet18(ResNetBase):
    BLOCK = PreActBasicBlock
    LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
    BLOCK = PreActBasicBlock
    LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
    BLOCK = PreActBottleneck
    LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
    BLOCK = PreActBottleneck
    LAYERS = (3, 4, 23, 3)


class ResNet152(ResNetBase):
    BLOCK = PreActBottleneck
    LAYERS = (3, 8, 36, 3)
    

def get_encoder(args):

    stem_pool = False
    stem_norm = False
    
    ## Only one architecture for now
    if args.enc_arch == "ResNet18":
        enc = ResNet18
    elif args.enc_arch == "ResNet34":
        enc = ResNet34
    elif args.enc_arch == "ResNet50":
        enc = ResNet50
    elif args.enc_arch == "ResNet101":
        enc = ResNet101
    elif args.enc_arch == "ResNet152":
        enc = ResNet152

    encoder = enc(stem_pool=stem_pool,
                  stem_norm=stem_norm)
    return encoder
