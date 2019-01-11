import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) * rgb_range

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        modules = []
        if (scale & (scale - 1)) == 0:    
            for _ in range(int(math.log(scale, 2))):
                modules.append(_PyramidPoolingModule(n_feat,n_feat,(1,2,3,6)))
                modules.append(conv(5*n_feat,4*n_feat,3,bias))
                modules.append(nn.PixelShuffle(2))
                if bn: modules.append(nn.BatchNorm2d(n_feat))
                if act: modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feat, 9 * n_feat, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if bn: modules.append(nn.BatchNorm2d(n_feat))
            if act: modules.append(act())
           
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*modules)
        
class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(reduction_dim,reduction_dim,kernel_size=3,padding=(3//2),bias = True),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        default_conv
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out
