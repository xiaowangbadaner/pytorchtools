from model.backbones import resnet_conv,vgg_conv
import torch
import torchvision
# net = torchvision.models.resnet50()
net = resnet_conv.ResNet50()
from torchsummary import summary
summary(net, (3, 128, 128),device='cpu')


'''
I have a question that I define a network resnet50 which has the same parameters with the torchvision.models.resnet50,
it's total params and parames size is less than torchvision.models.resnet50
BUT!!! it's Forward/backward pass size and Estimated Total Size is bigger than torchvision.models.resnet50!
you can run the code in this pyfile to watch the question.
'''