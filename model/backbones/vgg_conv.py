'''
bulid vgg16 with no linear
we only use the first n layers as the backbones
    by:W.H
'''
import torch
import torch.nn as nn
from model.base_cell.ConvCell import conv_cell
import torchvision


class vgg_conv(nn.Module):
    def __init__(self,arch):
        super(vgg_conv, self).__init__()
        self.conv3_64 = self.__make_layers(
            in_channels=3,out_channels=64,num=arch[0])
        self.conv64_128 = self.__make_layers(
            in_channels=64,out_channels=128, num=arch[1])
        self.conv128_256 = self.__make_layers(
            in_channels=128,out_channels=256, num=arch[2])
        self.conv256_512 = self.__make_layers(
            in_channels=256,out_channels=512, num=arch[3])
        self.conv512_512 = self.__make_layers(
            in_channels=512,out_channels=512, num=arch[4])

    def forward(self,x):
        out1 = self.conv3_64(x)
        out2 = self.conv64_128(out1)
        out3 = self.conv128_256(out2)
        out4 = self.conv256_512(out3)
        out5 = self.conv512_512(out4)

        #we will return five out for segmentors
        #if you use the net for classfication,you can only use the out5
        return [out1,out2,out3,out4,out5]

    def __make_layers(self,in_channels,out_channels,num):
        layers = []
        for i in range(num):
            layers.append(conv_cell(
                in_channels=in_channels,
                out_channels=out_channels,kernel_size=3,
                padding=1,
                stride=1,
                is_BN=False))   #文章中貌似未使用BN
            in_channels = out_channels
        layers.append(nn.MaxPool2d(2,2))
        return nn.Sequential(*layers)


def VGG_11():
    return vgg_conv([1, 1, 2, 2, 2])

def VGG_13():
    return vgg_conv([1, 1, 2, 2, 2])

def VGG_16():
    return vgg_conv([2, 2, 3, 3, 3])

def VGG_19():
    return vgg_conv([2, 2, 4, 4, 4])

def VGG(layers_list):
    '''
    notes:  we can diy vggnet on there, and that you should use a list with len == 5\n
    :param layers_list: define the depth of every bolck
    :return: vggnet which can return 5 outputs
    '''
    return vgg_conv(layers_list)

# net = VGG([1,1,2,2,2])
# net = torchvision.models.vgg11()
# print(net)