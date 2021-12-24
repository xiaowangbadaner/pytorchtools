'''
bulid resnet with no linear
we only use the first n layers as the backbones
for resnet18 and resnet 34,we use BasicBlock
else, we use Bottleneck
    by:W.H
'''
import torch
import torch.nn as nn
from model.base_cell.ConvCell import conv_cell


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        kernel_size=3,
        padding = 1,
        bias = False,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_cell(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.conv2 = conv_cell(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
            is_avtivate=False,
        )
        if stride==2:
            self.downsample = conv_cell(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                bias=bias,
                is_avtivate=False,
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.downsample:
            x = self.downsample(x)
        return self.relu(y+x)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            kernel_size=3,
            padding=1,
            bias=False,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_cell(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=bias
        )
        self.conv2 = conv_cell(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.conv3 = conv_cell(
            in_channels=out_channels,
            out_channels=out_channels*self.expansion,
            kernel_size=1,
            stride=1,
            bias=bias,
            is_avtivate=False,
        )
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.downsample = conv_cell(
                in_channels=in_channels,
                out_channels=out_channels*self.expansion,
                kernel_size=1,
                stride=stride,
                bias=bias,
                is_avtivate=False,
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.downsample:
            x = self.downsample(x)
        return self.relu(y+x)

class resnet(nn.Module):
    def __init__(
            self,
            block,
            num_blocks,
    ):
        super(resnet, self).__init__()
        self.conv1 = conv_cell(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1)
        self.in_channels = 64
        self.conv64_64 = self.__make_layers(
            block, num_blocks[0], stride=1,out_channels=64)
        self.conv64_128 = self.__make_layers(
            block, num_blocks[1], stride=2,out_channels=128)
        self.conv126_256 = self.__make_layers(
            block, num_blocks[2], stride=2,out_channels=256)
        self.conv256_512 = self.__make_layers(
            block, num_blocks[3], stride=2,out_channels=512)

    def forward(self,x):
        out1 = self.conv1(x)
        out1_ = self.maxpool(out1)
        out2 = self.conv64_64(out1_)
        out3 = self.conv64_128(out2)
        out4 = self.conv126_256(out3)
        out5 = self.conv256_512(out4)
        # x = self.conv1(x)
        # x = self.maxpool(x)
        # x = self.conv64_64(x)
        # x = self.conv64_128(x)
        # x = self.conv126_256(x)
        # out5 = self.conv256_512(x)
        return [out1,out1_,out2,out3,out4,out5]
        # return out5

    def __make_layers(self,block,num_block,stride,out_channels):
        strides = [stride] + [1] * (num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels,out_channels,stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

def ResNet18():
    return resnet(BasicBlock, [2,2,2,2])

def ResNet34():
    return resnet(BasicBlock, [3,4,6,3])

def ResNet50():
    return resnet(Bottleneck, [3,4,6,3])

def ResNet101():
    return resnet(Bottleneck, [3,4,23,3])

def ResNet152():
    return resnet(Bottleneck, [3,8,36,3])

def ResNet(block,layers_list):
    '''
    notes: we can diy resnet on there,and that you should use a list with len == 4\n
    :param block: select which block to use
    :param layers_list: define the depth of every bolck
    :return: resnet which can return 6 outputs
    '''
    return resnet(block,layers_list)




