## These blocks are based on the resnet block definitions in MinkowskiEngine:
## https://github.com/NVIDIA/MinkowskiEngine/blob/master/MinkowskiEngine/modules/resnet_block.py
## But are modified to allow ResNetv2 style preactivations
## I also took the ResNet D option from https://arxiv.org/pdf/1812.01187 for the bottleneck block
import torch.nn as nn
import MinkowskiEngine as ME

class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 bn_momentum=0.1,
                 dimension=2):
        super(PreActBasicBlock, self).__init__()
        assert dimension > 0

        self.norm1 = ME.MinkowskiBatchNorm(inplanes, momentum=bn_momentum)        
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)

        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)

        self.relu = ME.MinkowskiReLU(inplace=True)

        ## Change downsample to a 2x2 patch avg pooling layer and 1x1 convolution
        if stride != 1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=dimension),
                ME.MinkowskiConvolution(inplanes, self.expansion*planes, kernel_size=1, stride=1, dimension=dimension)
            )

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.relu(out)

        ## Apply the downsampling shortcut after the normalization and relu in this version
        residual = self.shortcut(out) if hasattr(self, 'shortcut') else x
        
        out = self.conv1(x)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 bn_momentum=0.1,
                 dimension=2):
        super(PreActBottleneck, self).__init__()
        assert dimension > 0

        self.norm1 = ME.MinkowskiBatchNorm(inplanes, momentum=bn_momentum)
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, dimension=dimension)

        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)

        self.norm3 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv3 = ME.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, dimension=dimension)

        self.relu = ME.MinkowskiReLU(inplace=True)

        ## Change downsample to a 2x2 patch avg pooling layer and 1x1 convolution
        if stride != 1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=dimension),
                ME.MinkowskiConvolution(inplanes, self.expansion*planes, kernel_size=1, stride=1, dimension=dimension)
            )
            
    def forward(self, x):
        
        out = self.norm1(x)
        out = self.relu(out)

        ## Apply the downsampling shortcut after the normalization and relu in this version
        residual = self.shortcut(out) if hasattr(self, 'shortcut') else x
        
        out = self.conv1(x)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out
