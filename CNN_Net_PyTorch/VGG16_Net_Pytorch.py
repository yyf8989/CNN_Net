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


class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 第二个卷积层
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第三个卷积层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 第四个卷积层
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 第五个卷积层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 第六个卷积层
            nn.Conv2d(256, 256, 3, 1),
            nn.ReLU(True),
            # 第七个卷积层
            nn.Conv2d(256, 256, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 第八个卷积层
            nn.Conv2d(256, 512, 3, 1),
            nn.ReLU(True),
            # 第九个卷积层
            nn.Conv2d(512, 512, 3, 1),
            nn.ReLU(True),
            # 第十个卷积层
            nn.Conv2d(512, 512, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 第十一个卷积层
            nn.Conv2d(512, 512, 3, 1),
            nn.ReLU(True),
            # 第十二个卷积层
            nn.Conv2d(512, 512, 3, 1),
            nn.ReLU(True),
            # 第十三个卷积层
            nn.Conv2d(512, 512, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), )

        self.classfilter = nn.Sequential(
            # 第一个全连接层
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 第二个全连接层
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 第三个全连接层
            nn.Linear(4096, num_classes),)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classfilter(x)

        return x
