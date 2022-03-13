from unicodedata import bidirectional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN import conv_block,fc_block

'''
LSTM
'''
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, drop_out_p = 0.5, bidirection=True):
        super(CNN_LSTM, self).__init__()
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
        
        #input size = (N, C-100, L-38,W-1)
        self.conv4 = conv_block(in_channels=100, 
                                out_channels=200, 
                                drop_out=drop_out_p,
                                kernel_size = (10,1),
                                padding = "same")

        
        x_trivial = self.conv4(self.conv3(self.conv2(self.conv1(torch.zeros(1,*input_dim)))))
        N,C,H,W = x_trivial.size()
        x_trivial = x_trivial.permute(0,2,1,3)
        x_trivial = x_trivial.view(N,H,-1)

        #We want to make it N, H, Hin = C*W
        h_dim = 200
        self.lstm = nn.LSTM(200,h_dim,2,dropout = drop_out_p, batch_first = True, bidirectional = bidirection)

        out_tensor,(_,_) = self.lstm(x_trivial)
        out_shape = torch.flatten(out_tensor).shape[0]
        self.fc1 = fc_block(in_shape=out_shape, out_channels=4, drop_out=drop_out_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
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

class CNN_GRU(nn.Module):
    def __init__(self, input_dim, drop_out_p = 0.5, bidirection=True):
        super(CNN_LSTM, self).__init__()
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
        
        #input size = (N, C-100, L-38,W-1)
        self.conv4 = conv_block(in_channels=100, 
                                out_channels=200, 
                                drop_out=drop_out_p,
                                kernel_size = (10,1),
                                padding = "same")        

        x_trivial = self.conv4(self.conv3(self.conv2(self.conv1(torch.zeros(1,*input_dim)))))
        N,C,H,W = x_trivial.size()
        x_trivial = x_trivial.permute(0,2,1,3)
        x_trivial = x_trivial.view(N,H,-1)

        #We want to make it N, H, Hin = C*W
        h_dim = 200
        self.gru = nn.GRU(200,h_dim,2,dropout = drop_out_p, batch_first = True, bidirectional = bidirection)

        out_tensor,(_,_) = self.lstm(x_trivial)
        out_shape = torch.flatten(out_tensor).shape[0]
        self.fc1 = fc_block(in_shape=out_shape, out_channels=4, drop_out=drop_out_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        batch_num,C,H,W = x.size()
        x = x.permute(0,2,1,3)

        x = x.view(batch_num, H,-1) #change x to (N, input_dim)
        x,(_,_) = self.gru(x)

        x = x.reshape(batch_num, -1)
        out = self.fc1(x)
        return out
