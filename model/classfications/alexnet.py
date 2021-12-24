from model.backbones.alexnet_conv import alexnet_conv
import torch
import torch.nn as nn

class alexnet(nn.Module):
    def __init__(self,num_class):
        super(alexnet, self).__init__()
        self.backbone = alexnet_conv()
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

