## These blocks are based on the resnet block definitions in MinkowskiEngine:
## https://github.com/NVIDIA/MinkowskiEngine/blob/master/MinkowskiEngine/modules/resnet_block.py
## But are modified to allow ResNetv2 style preactivations
## I also took the ResNet D option from https://arxiv.org/pdf/1812.01187 for the bottleneck block
import torch.nn as nn
import MinkowskiEngine as ME
from collections import OrderedDict

class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 bn_momentum=0.1,
                 res_pool=False,
                 apply_norm=True,
                 dimension=2):
        super(PreActBasicBlock, self).__init__()
        assert dimension > 0

        ## Make BN optional
        self.apply_norm	= apply_norm

        if self.apply_norm:
            self.norm1 = ME.MinkowskiBatchNorm(inplanes, momentum=bn_momentum)
        else: self.norm1 = None
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)

        if self.apply_norm:
            self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        else: self.norm2 = None
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)

        self.relu = ME.MinkowskiReLU(inplace=True)

        ## Support a few options for the residual connection
        if stride != 1 or inplanes != self.expansion*planes:
            res = OrderedDict()
            if res_pool and stride != 1:
                res['res_pool'] = ME.MinkowskiAvgPooling(kernel_size=stride, stride=stride, dimension=dimension)
                res['res_conv'] = ME.MinkowskiConvolution(inplanes, self.expansion*planes, kernel_size=1, stride=1, dimension=dimension)
            else:
                res['res_conv'] = ME.MinkowskiConvolution(inplanes, self.expansion*planes, kernel_size=1, stride=stride, dimension=dimension)
            self.shortcut = nn.Sequential(res)

    def forward(self, x):
        out = x
        if self.norm1: out = self.norm1(x)
        out = self.relu(out)

        ## Apply the downsampling shortcut after the normalization and relu in this version
        residual = self.shortcut(out) if hasattr(self, 'shortcut') else x
        
        out = self.conv1(out)
        if self.norm2: out = self.norm2(out)
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
                 res_pool=False,
                 apply_norm=True,
                 dimension=2):
        super(PreActBottleneck, self).__init__()
        assert dimension > 0

        ## Make BN optional
        self.apply_norm	= apply_norm

        if self.apply_norm:
            self.norm1 = ME.MinkowskiBatchNorm(inplanes, momentum=bn_momentum)
        else:
            self.norm1 = None
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, dimension=dimension)

        if self.apply_norm:
            self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        else: self.norm2 = None
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)

        if self.apply_norm:
            self.norm3 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        else: self.norm3 = None
        self.conv3 = ME.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, dimension=dimension)

        self.relu = ME.MinkowskiReLU(inplace=True)

        ## Support a few options for the residual connection
        if stride != 1 or inplanes != self.expansion*planes:
            res = OrderedDict()
            if res_pool and stride != 1:
                res['res_pool'] = ME.MinkowskiAvgPooling(kernel_size=stride, stride=stride, dimension=dimension)
                res['res_conv'] = ME.MinkowskiConvolution(inplanes, self.expansion*planes, kernel_size=1, stride=1, dimension=dimension)
            else:
                res['res_conv'] = ME.MinkowskiConvolution(inplanes, self.expansion*planes, kernel_size=1, stride=stride, dimension=dimension)
            self.shortcut = nn.Sequential(res)
            
    def forward(self, x):

        out = x
        if self.norm1: out = self.norm1(x)
        out = self.relu(out)

        ## Apply the downsampling shortcut after the normalization and relu in this version
        residual = self.shortcut(out) if hasattr(self, 'shortcut') else x
        
        out = self.conv1(out)
        if self.norm2: out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm3: out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out
