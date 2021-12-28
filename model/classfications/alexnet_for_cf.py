'''
bulid alexnet's linear for classfication
we only use the last n layers as the backbones
    by:W.H
'''
from model.backbones.alexnet_conv import alexnet_conv
import torch
import torch.nn as nn

class alexnet_for_cf(nn.Module):
    def __init__(self,backbone,num_class):
        super(alexnet_for_cf, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,num_class)
        )
    def forward(self,x):
        conv_res = self.backbone(x)[-1]
        y = self.classifier(conv_res)

        return y

# import torchvision
# backbone = alexnet_conv
# net = alexnet(backbone,1000)
# net = torchvision.models.alexnet()
# from torchsummary import summary
# summary(net, (3, 128, 128),device='cpu')
# print(net)