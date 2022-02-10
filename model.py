import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, kernel, pooling, BN=True, relu_type='leaky'):
        super().__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=False) if relu_type=='leaky' else nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(drop_rate, inplace=False) 
       
    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class _CNN(nn.Module):
    def __init__(self, fil_num, drop_rate):
        super(_CNN, self).__init__()
        self.block1 = ConvLayer(1, fil_num, 0.1, (7, 2, 0), (3, 2, 0))
        self.block2 = ConvLayer(fil_num, 2*fil_num, 0.1, (4, 1, 0), (2, 2, 0))
        self.block3 = ConvLayer(2*fil_num, 4*fil_num, 0.1, (3, 1, 0), (2, 2, 0))
        self.block4 = ConvLayer(4*fil_num, 8*fil_num, 0.1, (3, 1, 0), (2, 1, 0))
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(8*fil_num*6*8*6, 30),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(30, 2),
        )

    def forward(self, x, stage='normal'):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        feats = self.dense1(x)
        out = self.dense2(feats)
        if stage == 'get_features':
            return (feats, out)
        else:
            return out
