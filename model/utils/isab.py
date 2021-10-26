"""
Internal_feature_Split_Attention_Block
Created by: Yuanzhi Wang
Email: w906522992@gmail.com
Core Link: https://github.com/mdswyz/SISN-Face-Hallucination
Paper Link: https://dl.acm.org/doi/10.1145/3474085.3475682
"""

import math
import torch
import torch.nn as nn


from .ISA import Internal_feature_Split_Attention


class Internal_feature_Split_Attention_Block(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=4, cardinality=1, bottleneck_width=64, dilation=1, norm_layer=nn.BatchNorm2d, change_channel=False):
        super(Internal_feature_Split_Attention_Block, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.radix = radix

        self.conv2 = Internal_feature_Split_Attention(
            group_width, group_width, kernel_size=3,
            stride=stride, padding=dilation,
            dilation=dilation, groups=cardinality, bias=False,
            radix=radix, norm_layer=norm_layer)
        self.conv3 = nn.Conv2d(
            group_width, planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes)

        self.change_channel = change_channel
        if inplanes != planes:
            self.change_channel = True

        self.skip = nn.Conv2d( inplanes, planes, (1, 1), stride=(stride, stride), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.change_channel:
           residual = self.skip(residual)
           

        out += residual
        out = self.relu(out)

        return out