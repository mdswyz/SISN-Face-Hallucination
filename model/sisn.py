"""
SISN
Created by: Yuanzhi Wang
Email: w906522992@gmail.com
Core Link: https://github.com/mdswyz/SISN-Face-Hallucination
Paper Link: https://dl.acm.org/doi/10.1145/3474085.3475682
"""

from .utils import common
from .utils.isab import Internal_feature_Split_Attention_Block
import torch.nn as nn
from model import ops


## External-internal Split Attention Group (ESAG)
class External_internal_Split_Attention_Group(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_ISABs, conv2 = common.default_conv):
        super(External_internal_Split_Attention_Group, self).__init__()
        modules_body = []
        modules_body = [
            conv(
                n_feat, n_feat) \
            for _ in range(n_ISABs)]
        modules_body.append(conv2(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
        self.IA = Internal_feature_Split_Attention_Block(n_feat,n_feat)

    def forward(self, x):
        res = self.body(x)
        x = self.IA(x)
       
        res += x
        return res

## Split Attention in Split_Attention Network (SISN)
class Net(nn.Module):
    def __init__(self, opt, conv = Internal_feature_Split_Attention_Block, conv2 = common.default_conv):
        super(Net, self).__init__()
        
        n_ESAGs = opt.num_groups
        n_ISABs = opt.num_blocks
        n_feats = opt.num_channels
        kernel_size = 3
        reduction = opt.reduction
        scale = opt.scale
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(255, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [
            ops.DownBlock(opt.scale),
            nn.Conv2d(3 * opt.scale ** 2, opt.num_channels, 3, 1, 1),
        ]

        # define body module
        modules_body = [
            External_internal_Split_Attention_Group(
                conv, n_feats, kernel_size, n_ISABs=n_ISABs) \
            for _ in range(n_ESAGs)]

        modules_body.append(conv2(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv2, scale, n_feats, act=False),
            conv2(n_feats, 3, kernel_size)]

        self.add_mean = common.MeanShift(255, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 
