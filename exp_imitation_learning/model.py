import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import time
from torch.autograd import Variable
import pandas as pd
import numpy as np


class RNN(nn.Module):
    '''RNN model for game2048'''
    def __init__(self, size):
        super().__init__()

        self.my_rnn = nn.LSTM(
            input_size=4,
            hidden_size=size,
            num_layers=4,
            batch_first=True
        )
        self.out = nn.Linear(size, 4)

    def forward(self, x):

        r_out, (h_n, h_c) = self.my_rnn(x,None)
        out = self.out(r_out[:, -1 ,:])
        return out

class RNN_onehot(nn.Module):
    '''RNN model using onehot input'''
    def __init__(self):
        super().__init__()

        self.my_rnn = nn.LSTM(
            input_size=4,
            hidden_size=128,
            num_layers=4,
            batch_first=True
        )
        
        self.conv1 = nn.Conv2d(
                in_channels=12,    # 输入通道数
                out_channels=6,  # 输出通道数
                kernel_size=3,    # 卷积核的尺寸是(5,5)
                stride=1,         # 步长为1
                padding=1        # 零填充保持图片宽高不变,padding = (kernel_size - stride) / 2
        )
        
        self.conv2 = nn.Conv2d(
                in_channels=6,    # 输入通道数
                out_channels=3,  # 输出通道数
                kernel_size=3,    # 卷积核的尺寸是(5,5)
                stride=1,         # 步长为1
                padding=1        # 零填充保持图片宽高不变,padding = (kernel_size - stride) / 2
        )
        self.conv3 = nn.Conv2d(
                in_channels=3,    # 输入通道数
                out_channels=1,  # 输出通道数
                kernel_size=1,    # 卷积核的尺寸是(5,5)
                stride=1,         # 步长为1
                padding=0         # 零填充保持图片宽高不变,padding = (kernel_size - stride) / 2
        )

        

        self.out = nn.Linear(128, 4)
        self.out2 = nn.Linear(16, 1)#（16，1）
        
    def forward(self, x):
        
        w1 = self.conv1(x)

        w2 = self.conv2(w1)

        w3 = self.conv3(w2)

        w=w3[:,0,:,:]
        
        r_out, (h_n, h_c) = self.my_rnn(w,None)
        out = self.out(r_out[:, -1 ,:])
    
        return out