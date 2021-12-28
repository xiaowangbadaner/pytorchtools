from model.backbones.resnet_conv import ResNet,BasicBlock,Bottleneck,ResNet18
import torch
import torch.nn as nn

class resnet_for_cf(nn.Module):
    def __init__(self,backbone,num_class):
        super(resnet_for_cf, self).__init__()
        # self.backbone = ResNet(block,num_blocks)
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512*self.backbone.expansion,num_class)
        )
    def forward(self,x):
        conv_res = self.backbone(x)[-1]
        y = self.classifier(conv_res)
        return y


# import torchvision
# backbone = ResNet18()
# net = resnet(backbone,1000)
# net = torchvision.models.resnet50()
# from torchsummary import summary
# summary(net, (3, 128, 128),device='cpu')
# print(net)