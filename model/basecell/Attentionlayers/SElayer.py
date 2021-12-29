'''
build channel wise attentionlayers

    by: W.H
'''
import torch
import torch.nn as nn


class SElayer(nn.Module):
    def __init__(self,channel,reduction=16,mode='conv'):
        super(SElayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        assert mode in ['conv','linear'], 'unsupport activation type %s...' % mode
        if mode == 'conv':
            self.func = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel,
                    out_channels=int(channel/reduction)+1,
                    kernel_size=1,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=int(channel/reduction)+1,
                    out_channels=channel,
                    kernel_size=1,
                ),
                nn.Sigmoid()
            )
        if mode == 'linear':
            self.func = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_features=channel,
                    out_features=int(channel/reduction)+1,
                    bias=False
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    in_features=int(channel / reduction) + 1,
                    out_features=channel,
                    bias=False
                ),
                nn.Sigmoid()
            )
    def forward(self,x):
        atten_x = self.pool(x)
        atten_x = self.func(atten_x).view(x.shape[0],-1,1,1)
        y = x * atten_x
        return y
