from model.backbones.resnet_conv import ResNet,BasicBlock,Bottleneck
import torch
import torch.nn as nn

class resnet(nn.Module):
    def __init__(self,block,num_blocks,num_class):
        super(resnet, self).__init__()
        self.backbone = ResNet(block,num_blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512*block.expansion,num_class)
        )
    def forward(self,x):
        conv_res = self.backbone(x)[-1]
        print(conv_res.shape)
        y = self.classifier(conv_res)
        return y
