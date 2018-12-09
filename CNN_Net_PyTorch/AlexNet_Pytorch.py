#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
__author__ = 'YYF'
__mtime__ = '2018/11/19'
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


class AleNet(nn.Module):
    def __init__(self, num_classes):
        super(AleNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),)

        self.classfilter = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),)

    def foreard(self, x):
        x = self.features(x),
        x = x.view(x.size(0), 256*6*6)
        x = self.classfilter(x)
        return x