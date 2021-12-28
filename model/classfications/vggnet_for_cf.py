# from model.backbones.vgg_conv import VGG,VGG_16
from __selectModels import *
import torch
import torch.nn as nn

class vggnet_for_cf(nn.Module):
    def __init__(self,backbone,num_class):
        super(vggnet_for_cf, self).__init__()
        self.backbone = backbone
        self.Adaptpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,num_class)
        )
    def forward(self,x):
        conv_res = self.backbone(x)[-1]
        conv_res = self.Adaptpool(conv_res)
        y = self.classifier(conv_res)
        return y

# import torchvision
# backbone = VGG_19()
# net = vggnet_for_cf(backbone,1000)
# # net = torchvision.models.VGG()
# from torchsummary import summary
# summary(net, (3, 128, 128),device='cpu')
# print(net)