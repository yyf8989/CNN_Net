#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
__author__ = 'YYF'
__mtime__ = '2018/11/20'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓   ┏┓
            ┏┛┻━━━┛┻┓
           ┃   ☃   ┃
           ┃ ┳┛ ┗┳ ┃
           ┃   ┻    ┃
            ┗━┓   ┏━┛
              ┃    ┗━━━┓
               ┃ 神兽保佑 ┣┓
               ┃ 永无BUG! ┏┛
                ┗┓┓┏ ━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.Conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.Conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(Inception, self).__init__()
        self.branchlxl = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3db1_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3db1_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3db1_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(
            in_channels, pool_features, kernel_size=1 )

    def forward(self, x):
        branch1x1 = self.branchlxl(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3db1 = self.branch3x3db1_1(x)
        branch3x3db1 = self.branch3x3db1_2(branch3x3db1)
        branch3x3db1 = self.branch3x3db1_3(branch3x3db1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3db1, branch_pool]
        return torch.cat(outputs, 1)
