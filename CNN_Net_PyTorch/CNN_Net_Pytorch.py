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
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np

save_cnn = r"E:\Pycharmprojects\MLP_Net\MLP_PyTorch\MLP_Net_ckpt\1-ck"
# 控制输出的随机数一致
torch.manual_seed(1)

# 定义训练批次
Epoch = 1
# 定义数据集大小
Batch_size = 50
# 定义学习率
LR = 0.001
# 已经下载好了数据集可以写成False
Download_Mnist = False

train_data = torchvision.datasets.MNIST(
    # 定义文件位置
    root=r'./mnist_data',
    # 训练数据
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=Download_Mnist,
)

test_data = torchvision.datasets.MNIST(root=r'./mnist_data/', train=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True)

# 整理测试集为2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255
test_y = test_data.test_labels[:2000]


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # 因为是mnist数据集，那么数据类型为（28,28,1）
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),  # output为(28,28,16)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # 输出为(14, 14, 16)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # 输出为（14， 14， 32）
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 输出为（7,7,32）
        )

        self.out = nn.Linear(7 * 7 * 32, 10)

    # 进行前向传播
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


if __name__ == '__main__':
    # 实例化变量
    cnn = CNNNet()
    print(cnn)
    # 设置优化器
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    # 设置损失函数
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(Epoch):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = cnn(test_x[:10])
                # pre_y = torch.argmax(test_output)
                pre_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                print('pre_y:', pre_y)
                true_label = test_y[:10].numpy()
                print('true_label:', true_label)
                test_acc = np.mean(np.float32((pre_y == true_label)))
                print('Epoch:{}, train_loss:{:.4f}, test_accuracy:{:.2f}%'.format(step, loss, test_acc * 100))

                # 保存model
                torch.save(cnn, save_cnn)