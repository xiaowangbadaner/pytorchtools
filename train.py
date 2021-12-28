# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader,Dataset,TensorDataset
# import torchvision.transforms as transforms
# import torchvision
#
# from model.classfications import resnet_for_cf,Bottleneck,BasicBlock
# from model.classfications import alexnet_for_cf
# from model.classfications import vggnet_for_cf
# # from backbones import alexnet_conv
# from __selectModels import *
# # torch.backends.cudnn.deterministic=True
# # torch.backends.cudnn.benchmark=False
# # torch.manual_seed(1111)
# transform = transforms.Compose([
#     # transforms.RandomCrop(32,padding=4),
#     transforms.Resize(64),
#      transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# trainset = torchvision.datasets.ImageFolder(root=r'E:\data\train',transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,shuffle=True)
#
# validset = torchvision.datasets.ImageFolder(root=r'E:\data\valid',transform=transform)
# validloader = torch.utils.data.DataLoader(validset, batch_size=16,shuffle=True)
#
# # net = resnet(Bottleneck, [3,4,6,3],10).cuda()
# backbone = alexnet_conv()
# net = alexnet_for_cf(backbone,10).cuda()
# # net = vggnet([2, 2, 3, 3, 3],10).cuda()
# # net = resnet(Bottleneck, [3,4,23,3],10).cuda()
# # net = torchvision.models.vgg16()
# # net.classifier._modules['6'] = nn.Linear(4096, 10)
# # net = net.cuda()
#
# from torchsummary import summary
# # summary(net, (3, 128, 128),device='cuda')
# # summary(net1, (3, 128, 128),device='cuda')
# print(net)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(),lr=0.1)
# net.train()
# for epoch in range(10):
#     correct = 0
#     total = 0
#     i=0
#     for features,labels in trainloader:
#         i += 1
#         features, labels = features.cuda(),labels.cuda()
#         optimizer.zero_grad()
#         output = net(features)
#         loss = loss_fn(output,labels)
#         loss.backward()
#         optimizer.step()
#
#         predicted = output.argmax(dim=1)
#         total += labels.size(0)
#         correct += (predicted==labels).sum().item()
#         # del features,labels
#         # torch.cuda.synchronize()
#         if (i+1)%20==0:
#             print('[epoch:%d] loss:%.3f acc:%.3f%%' %(epoch+1,loss.item(),100*correct/total)
#             )
#
# net.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     i=0
#     for features,labels in validloader:
#         i += 1
#         features, labels = features.cuda(),labels.cuda()
#         output = net(features)
#         loss = loss_fn(output,labels)
#
#         predicted = output.argmax(dim=1)
#         total += labels.size(0)
#         correct += (predicted==labels).sum().item()
#         # del features,labels
#         # torch.cuda.synchronize()
#         if (i+1)%20==0:
#             print('loss:%.3f acc:%.3f%%' %(loss.item(),100*correct/total)
#             )