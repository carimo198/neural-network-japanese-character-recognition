# kuzu.py
# ZZEN9444, CSE, UNSW
# Mohammad Reza Hosseinzadeh | z5388543

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

### NetLin model - computes a linear function of the pixels in the image
# followed by log softmax
class NetLin(nn.Module):
    # define structure of the network - 1 x linear layer
    # input size = image size = 28x28x1
    # output = 10 Hiragana characters
    def __init__(self):
        super(NetLin, self).__init__()
        self.linear_layer1 = nn.Linear(784, 10) 
    
    # apply network and return output
    def forward(self, x):
        # flatten input by reshaping to a one-dimensional tensor
        x = x.view(x.shape[0], -1)
        # apply linear layer to input
        out_sum = self.linear_layer1(x)
        # apply log softmax
        output = F.log_softmax(out_sum, dim=1)
        return output 

### NetFull model - fully connected 2-layer network 
# 1 x FC hidden layer using tanh 
# 1 x FC output layer using log softmax
class NetFull(nn.Module):
    # define structure of model 
    def __init__(self):
        super(NetFull, self).__init__()
        # FC hidden layer
        self.fc_hidden = nn.Linear(784, 260)
        # FC output layer
        self.fc_output = nn.Linear(260, 10) 

    # apply network and return output
    def forward(self, x):
        # flatten input to 1-D tensor
        x = x.view(x.shape[0], -1)
        # apply tanh to input in hidden layer
        hid_fc = self.fc_hidden(x)
        hidden = torch.tanh(hid_fc)
        # apply log softmax to output layer
        out_fc = self.fc_output(hidden)
        output = F.log_softmax(out_fc, dim=1)
        return output 

### NetConv model -
# 2 x convolutional hidden layers using ReLU activation function
# 1 x fully connected hidden layer using ReLU activation function
# 1 x fully connected output layer using log softmax
# max pooling layer (2x2) after the Conv layers - improves efficiency of net
class NetConv1(nn.Module):
    # define structure of network
    def __init__(self):
        super(NetConv1, self).__init__()
        # hidden convolution layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=5, padding=2)
        # hidden convolution layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=24,
                               kernel_size=5, padding=2)
        
        # max pool
        self.max_pool = nn.MaxPool2d(2, 2)

        # first fully connected hidden layer        
        self.fc1 = nn.Linear(1176, 169)
        # second fully connected output layer
        self.fc2 = nn.Linear(169, 10)

    def forward(self, x):
       # apply the network and return output
       x = self.conv1(x)
       x = F.relu(x)
       x = self.max_pool(x)

       x = self.conv2(x)
       x = F.relu(x)
       x = self.max_pool(x)
             
       # flatten inputs to 1-dimension tensor
       x = x.view(x.shape[0], -1)
       
       x = self.fc1(x)
       x = F.relu(x)
       
       x = self.fc2(x)
       x = F.log_softmax(x, dim=1)

       return x

# Experimented with different architecture
# The below model is more efficient and has better all round performance 
# Refer to PDF report for more detailed discussion  
class NetConv(nn.Module):
    # define structure of network
    def __init__(self):
        super(NetConv, self).__init__()
        # hidden convolution layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, padding=2, stride=2)
        # hidden convolution layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=2, stride=1)
        
        # max pool
        self.max_pool = nn.MaxPool2d(2, 2)

        # dropout
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25) 

        # first fully connected hidden layer        
        self.fc1 = nn.Linear(1024, 128)
        # second fully connected output layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
       # apply the network and return output
       x = self.conv1(x)
       x = F.relu(x)
       x = self.max_pool(x)

       x = self.conv2(x)
       x = F.relu(x)
       x = self.max_pool(x)

       x = self.dropout1(x)
             
       # flatten inputs to 1-dimension tensor
       x = x.view(x.shape[0], -1)
       
       x = self.fc1(x)
       x = F.relu(x)

       x = self.dropout2(x)
       
       x = self.fc2(x)
       x = F.log_softmax(x, dim=1)

       return x


