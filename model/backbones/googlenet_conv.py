import torch
import torch.nn as nn
from basecell.ConvCell import conv_cell

class googlenet_conv(nn.Module):
    def __init__(self):
        super(googlenet_conv, self).__init__()
        self.block1 = nn.Sequential(
            conv_cell(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)
        )
        self.block2 = nn.Sequential(
            conv_cell(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                bias=False
            ),
            conv_cell(64,192,3,1,1,is_BN=False),
            nn.MaxPool2d(3,2,ceil_mode=True)
        )
        self.block3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.block4 = nn.Sequential(

            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )

        self.block5 = nn.Sequential(

            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.4)
        )
    def forward(self,x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)

        return [out1,out2,out3,out4,out5]
class Inception(nn.Module):
    def __init__(self,in_channels,ch1,ch3reduce,ch3,ch5reduce,ch5,pool_proj):
        super(Inception, self).__init__()
        self.branch1 = conv_cell(
            in_channels=in_channels,
            out_channels=ch1,
            kernel_size=1,
            bias=False
        )

        self.branch2 = nn.Sequential(
            conv_cell(
                in_channels=in_channels,
                out_channels=ch3reduce,
                kernel_size=1,
                bias=False
            ),
            conv_cell(
                in_channels=ch3reduce,
                out_channels=ch3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        )

        self.branch3 = nn.Sequential(
            conv_cell(
                in_channels=in_channels,
                out_channels=ch5reduce,
                kernel_size=1,
                bias=False
            ),
            conv_cell(
                in_channels=ch5reduce,
                out_channels=ch5,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False
            )
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            conv_cell(
                in_channels=in_channels,
                out_channels=pool_proj,
                kernel_size=1,
                bias=False
            )
        )

    def forward(self,x):
        block1 = self.branch1(x)
        block2 = self.branch2(x)
        block3 = self.branch3(x)
        block4 = self.branch4(x)

        block = [block1, block2, block3, block4]

        return torch.cat(block, dim=1)

class InceptionAux(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(InceptionAux, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            conv_cell(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self,x):
        y = self.fc(x)
        return y





import torchvision
# x = torchvision.models.GoogLeNet()
x = googlenet_conv()
from torchsummary import summary
summary(x, (3, 128, 128),device='cpu')
























