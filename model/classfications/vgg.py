from model.backbones.vgg_conv import VGG
import torch
import torch.nn as nn

class vggnet(nn.Module):
    def __init__(self,arch,num_class):
        super(vggnet, self).__init__()
        self.backbone = VGG(arch)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7,7)),
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
        y = self.classifier(conv_res)

        return y
