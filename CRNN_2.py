import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN import fc_block

'''
LSTM
'''
class Pure_LSTM(nn.Module):
    def __init__(self, input_dim, stack_num = 4, drop_out_p = 0.5, bidirection=True):
        super(Pure_LSTM, self).__init__()
        h_dim = 200
        self.lstm = nn.LSTM(22,h_dim,stack_num,dropout = drop_out_p, batch_first = True, bidirectional = bidirection)

        x_trivial = torch.zeros(1, *input_dim)
        N,C,H,W = x_trivial.size()
        x_trivial = x_trivial.permute(0,2,1,3)
        x_trivial = x_trivial.view(N,H,-1)
        out_tensor,(_,_) = self.lstm(x_trivial)
        out_shape = torch.flatten(out_tensor).shape[0]
        self.fc1 = fc_block(in_shape=out_shape, out_channels=4, drop_out=drop_out_p)
    
    def forward(self, x):
        batch_num,C,H,W = x.size()
        x = x.permute(0,2,1,3)

        x = x.view(batch_num, H,-1) #change x to (N, input_dim)
        x,(_,_) = self.lstm(x)

        x = x.reshape(batch_num, -1)
        out = self.fc1(x)
        return out

class CNN2_LSTM(nn.Module):
    def __init__(self, input_dim, stack_num = 2, drop_out_p = 0.5, bidirection=True):
        super(CNN2_LSTM, self).__init__()
        self.conv1 = conv_block2(in_channels=22, 
                                out_channels=25,
                                drop_out=drop_out_p, 
                                kernel_size = (10,1),
                                padding = "same")

        self.conv2 = conv_block2(in_channels=25, 
                                out_channels=50,
                                drop_out=drop_out_p,
                                kernel_size = (10,1),
                                padding = "same")

        
        x_trivial = self.conv2(self.conv1(torch.zeros(1,*input_dim)))
        N,C,H,W = x_trivial.size()
        x_trivial = x_trivial.permute(0,2,1,3)
        x_trivial = x_trivial.view(N,H,-1)

        #We want to make it N, H, Hin = C*W
        h_dim = 200
        self.lstm = nn.LSTM(50,h_dim,stack_num,dropout = drop_out_p, batch_first = True, bidirectional = bidirection)

        out_tensor,(_,_) = self.lstm(x_trivial)
        out_shape = torch.flatten(out_tensor).shape[0]
        self.fc1 = fc_block(in_shape=out_shape, out_channels=4, drop_out=drop_out_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        batch_num,C,H,W = x.size()
        x = x.permute(0,2,1,3)

        x = x.view(batch_num, H,-1) #change x to (N, input_dim)
        x,(_,_) = self.lstm(x)

        x = x.reshape(batch_num, -1)
        out = self.fc1(x)
        return out

'''
GRU
'''
class Pure_GRU(nn.Module):
    def __init__(self, input_dim, stack_num = 4, drop_out_p = 0.5, bidirection=True):
        super(Pure_GRU, self).__init__()
        h_dim = 200
        self.gru = nn.GRU(22,h_dim,stack_num,dropout = drop_out_p, batch_first = True, bidirectional = bidirection)

        x_trivial = torch.zeros(1, *input_dim)
        N,C,H,W = x_trivial.size()
        x_trivial = x_trivial.permute(0,2,1,3)
        x_trivial = x_trivial.view(N,H,-1)
        out_tensor,_ = self.gru(x_trivial)
        out_shape = torch.flatten(out_tensor).shape[0]
        self.fc1 = fc_block(in_shape=out_shape, out_channels=4, drop_out=drop_out_p)
    
    def forward(self, x):
        batch_num,C,H,W = x.size()
        x = x.permute(0,2,1,3)

        x = x.view(batch_num, H,-1) #change x to (N, input_dim)
        x,_ = self.gru(x)

        x = x.reshape(batch_num, -1)
        out = self.fc1(x)
        return out

class CNN2_GRU(nn.Module):
    def __init__(self, input_dim, stack_num = 4, drop_out_p = 0.5, bidirection=True):
        super(CNN2_GRU, self).__init__()
        self.conv1 = conv_block2(in_channels=22, 
                                out_channels=25,
                                drop_out=drop_out_p, 
                                kernel_size = (10,1),
                                padding = "same")

        self.conv2 = conv_block2(in_channels=25, 
                                out_channels=50,
                                drop_out=drop_out_p,
                                kernel_size = (10,1),
                                padding = "same")  

        x_trivial = self.conv2(self.conv1(torch.zeros(1,*input_dim)))
        N,C,H,W = x_trivial.size()
        x_trivial = x_trivial.permute(0,2,1,3)
        x_trivial = x_trivial.view(N,H,-1)

        #We want to make it N, H, Hin = C*W
        h_dim = 200
        self.gru = nn.GRU(50,h_dim,stack_num,dropout = drop_out_p, batch_first = True, bidirectional = bidirection)

        out_tensor,_ = self.gru(x_trivial)
        out_shape = torch.flatten(out_tensor).shape[0]
        self.fc1 = fc_block(in_shape=out_shape, out_channels=4, drop_out=drop_out_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        batch_num,C,H,W = x.size()
        x = x.permute(0,2,1,3)

        x = x.view(batch_num, H,-1) #change x to (N, input_dim)
        x,_ = self.gru(x)

        x = x.reshape(batch_num, -1)
        out = self.fc1(x)
        return out

class conv_block2(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out, **kwargs):
        super(conv_block2, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels,out_channels, **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.BN = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.elu(self.conv(x))
        x = self.maxpool(x)
        x = self.BN(x)
        x = self.dropout(x)
        return x
