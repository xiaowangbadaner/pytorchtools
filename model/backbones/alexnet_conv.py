'''
bulid alexnet with no linear
we only use the first n layers as the backbones
    by:W.H
'''
import torch.nn as nn
from basecell.ConvCell import conv_cell

class alexnet_conv(nn.Module):
    def __init__(self):
        super(alexnet_conv, self).__init__()
        self.conv1 = nn.Sequential(
            conv_cell(in_channels=3,out_channels=64,kernel_size=11,stride=4,padding=2,is_BN=False),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.conv2 = nn.Sequential(
            conv_cell(in_channels=64, out_channels=192, kernel_size=5,padding=2, is_BN=False),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = conv_cell(in_channels=192, out_channels=384, kernel_size=3, padding=1, is_BN=False)
        self.conv4 = conv_cell(in_channels=384, out_channels=256, kernel_size=3, padding=1, is_BN=False)
        self.conv5 = nn.Sequential(
            conv_cell(in_channels=256,out_channels=256,kernel_size=3,padding=1,is_BN=False),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)

        return [out1,out2,out3,out4,out5]

# net = torchvision.models.alexnet()
# net = alexnet_conv()
# net = nn.Sequential(*list(net.children())[:-2])
# x = torch.ones((1,3,128,128))
# y = net(x)
# print(net)
# from torchsummary import summary
# summary(net, (3, 128, 128),device='cpu')