'''
bulid base_conv_cell
we can define base_conv_cell on there,for add attentionlayers,decide whether use BN or activate
    by:W.H
'''
import torch.nn as nn
from selects import *

class conv_cell(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            is_BN=True,is_avtivate=True):
        super(conv_cell, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        if is_BN == True:
            self.BN = nn.BatchNorm2d(out_channels)    #bulidnormalization
        else:
            self.BN = None
        if is_avtivate != False:
            if is_avtivate == True:
                self.activation = selectActivation('relu')(inplace=True)           #buildactivation
            else:
                self.activation = selectActivation(is_avtivate)()
        else:
            self.activation = None

    def forward(self,x):
        y = self.conv(x)
        if self.BN:
            y = self.BN(y)
        if self.activation:
            y = self.activation(y)

        return y


