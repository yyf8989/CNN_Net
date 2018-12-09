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





class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # 输入 3, 32, 32
        layer1 = nn.Sequential()
        layer1.add_module('Conv1', nn.Conv2d(3, 32, 3, 1, padding=1))
        # 输出32, 32, 32
        layer1.add_module('Relu1', nn.ReLU(True))
        # 形状不变
        layer1.add_module('Pool1', nn.MaxPool2d(2, 2))
        # 输出 32, 16, 16
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('Conv2', nn.Conv2d(32, 64, 3, 1, padding=1))
        # 输出64, 16, 16
        layer2.add_module('Relu2', nn.ReLU(True))
        # 形状不变
        layer2.add_module('Pool2', nn.MaxPool2d(2, 2))
        # 输出 64, 8, 8
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('Conv3', nn.Conv2d(64, 128, 3, 1, padding=1))
        # 输出为128, 8, 8
        layer3.add_module('Relu3', nn.ReLU(True))
        # 形状不变
        layer3.add_module('Pool3', nn.MaxPool2d(2, 2))
        # 输出128， 4， 4
        self.layer3 = layer3

        layer4 = nn.Sequential()
        # 输入为128*4*4
        layer4.add_module('MLP1', nn.Linear(2048, 512))
        layer4.add_module('MLP1_Relu', nn.ReLU(True))
        layer4.add_module('MLP2', nn.Linear(512, 64))
        layer4.add_module('MLP2_Relu', nn.ReLU(True))
        layer4.add_module('MLP3', nn.Linear(64, 10))
        self.layer4 = layer4

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        mlp_input = conv3.view(conv3.size(0), -1)
        mlp_output = self.layer4(mlp_input)
        return mlp_output


if __name__ == '__main__':
    model = CNNNet()
    print(model)

    # 只要模型中的部分模型，如前两个卷积层
    new_model = nn.Sequential(*list(model.children())[:2])
    print(new_model)

    # # 提取模型中所有卷积层
    # convmodel = nn.Sequential()
    # for layer in model.named_modules():
    #     if isinstance(layer[1], nn.Conv2d):
    #         convmodel.add_module(layer[0], layer[1])
    #         print(convmodel)

    # 打印出全部参数
    # for param in model.named_parameters():
    #     print(param[0])