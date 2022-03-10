from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CNN(nn.Module):
    def __init__(self, drop_out_p = 0.5):
        super(CNN, self).__init__()
        #Initial input size = (22, 1000,1) -> (C, H, W)
        self.conv1 = conv_block(in_channels=22, 
                                out_channels=25,
                                drop_out=drop_out_p, 
                                kernel_size = (10,1),
                                padding = "same")

        #input size = (25, 334,1)
        self.conv2 = conv_block(in_channels=25, 
                                out_channels=50,
                                drop_out=drop_out_p,
                                kernel_size = (10,1),
                                padding = "same")
        
        #input size = (50, 112,1)
        self.conv3 = conv_block(in_channels=50, 
                                out_channels=100,
                                drop_out=drop_out_p, 
                                kernel_size = (10,1),
                                padding = "same")
        
        #input size = (100, 38,1)
        self.conv4 = conv_block(in_channels=100, 
                                out_channels=200, 
                                drop_out=drop_out_p,
                                kernel_size = (10,1),
                                padding = "same")
        
        #input size = (200, 12,1)
        #need flatten during forward()
        self.fc = nn.Sequential(
            nn.Linear(200*12, 144), 
            nn.BatchNorm1d(144),
            nn.ELU(),
            nn.Dropout(drop_out_p),
            nn.Linear(144,44),
            nn.BatchNorm1d(44),
            nn.ELU(),
            nn.Linear(44,4)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        batch_num = x.size(0)
        x = x.view(batch_num, -1) #change x to (N, input_dim)
        out = self.fc(x)
        return out


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out, **kwargs):
        super(conv_block, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels,out_channels, **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=(3,1))
        self.BN = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.elu(self.conv(x))
        x = self.maxpool(x)
        x = self.BN(x)
        x = self.dropout(x)
        return x

# test = torch.rand(1000,22,1000,1)
# model = CNN()

# out = model(test)
# print(out.shape)