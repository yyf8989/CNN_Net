#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
__author__ = 'YYF'
__mtime__ = '2018/11/18'
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
import torch
import torchvision
import torch.nn as nn
import numpy as np


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('Conv1', nn.Conv2d(1, 6, 3, padding=1))
        layer1.add_module('Pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('Conv1', nn.Conv2d(6, 16, 5))
        layer2.add_module('Pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('MLP1', nn.Linear(400, 120))
        layer3.add_module('MLP2', nn.Linear(120, 84))
        layer3.add_module('MLP3', nn.Linear(84, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x
