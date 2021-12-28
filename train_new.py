import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import time
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,TensorDataset
import torchvision
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.manual_seed(1111)

# import __selectModels
from model.classfications import resnet_for_cf,Bottleneck
from model.backbones.__selectModels import *

## 获取数据
transform = transforms.Compose([
    transforms.Resize(64),
     transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.ImageFolder(root=r'E:\data\train',transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,shuffle=True)

validset = torchvision.datasets.ImageFolder(root=r'E:\data\valid',transform=transform)
validloader = torch.utils.data.DataLoader(validset, batch_size=16,shuffle=True)

backbone = ResNet(BasicBlock,[1,1,1,1])
net = resnet_for_cf(backbone,10).cuda()
net.to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(),lr=0.001)

epochs = 10
lr = 0.01
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
class trainer():
    def __init__(self,**kwargs):
        '''
        init the class
        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = kwargs

        self.net = self.cfg['net']
        self.loss_fn = self.cfg['loss_fn'] if 'loss_fn' in self.cfg.keys() else warnings.filterwarnings('you must define the loss function..')
        self.optimizer = self.cfg['optimizer'] if 'optimizer' in self.cfg.keys() else torch.optim.Adam(self.net.parameters(),lr=0.001) and print('auto set optimizer as Adam with lr = 0.001')
        self.epochs = self.cfg['epochs'] if 'epochs' in self.cfg.keys() else 10 #and print('auto set epochs = 10')

        self.trainloader = self.cfg['trainloader'] if 'trainloader' in self.cfg.keys() else warnings.filterwarnings('you must give the trainloader..')
        self.validloader = self.cfg['validloader'] if 'validloader' in self.cfg.keys() else warnings.filterwarnings('you must give the validloader..')
        # self.testloader = self.cfg['testloader'] if 'testloader' in self.cfg.keys() else warnings.filterwarnings('you must give the testloader..')

        self.train_losses = []
        self.train_acces = []
        self.valid_losses = []
        self.valid_acces = []
        self.test_losses = []
        self.test_acces = []
        # self.print_parameters()

    def fit(self):
        '''
        train the network
        :return:
        '''
        for epoch in range(self.epochs):
            self.net.train()
            loss = 0
            total, correct = 0, 0
            len_ = self.trainloader.__len__()
            with tqdm(total=len_) as p_bar:
                for i, data in enumerate(self.trainloader):
                    features, labels = data
                    features, labels = features.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    output = self.net(features)
                    l = self.loss_fn(output, labels)
                    l.backward()
                    self.optimizer.step()

                    predicted = output.argmax(dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss += l.item()
                    p_bar.update()
                    p_bar.set_description("epoch:{}/{}, iteration:{}/{}, loss={:.3f}, "
                                          .format(epoch + 1, epochs, i + 1, len_, l.item()))
            ##val
            self.net.eval()
            with torch.no_grad():
                val_correct = 0
                val_total = 0
                val_loss = 0
                for j, data in enumerate(self.validloader):
                    features, labels = data
                    features, labels = features.to(self.device), labels.to(self.device)
                    output = self.net(features)
                    l = self.loss_fn(output, labels)

                    predicted = output.argmax(dim=1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_loss += l.item()

            time.sleep(0.5)
            time_ = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # print('%s epoch:%d/%d train_acc:%.3f%%' % (time_,epoch + 1,epochs,100 * correct / total))
            print('{} epoch:{}/{} train_acc:{:.3f}% train_loss:{:.3f} val_acc:{:.3f}% val_loss:{:.3f}'
                  .format(time_, epoch + 1, epochs, correct / total * 100, loss / (i + 1),
                          val_correct / val_total * 100, val_loss / (j + 1)))
            time.sleep(0.5)
            self.train_losses.append(loss / (i + 1))
            self.train_acces.append(correct / total * 100)
            self.valid_losses.append(val_loss / (j + 1))
            self.valid_acces.append(val_correct / val_total * 100)


    def print_history(self):
        '''
        save logs and draw the acc or loss
        :return:
        '''
        plt.plot(range(len(self.train_losses)), self.train_losses,label='train_losses',color="r")
        plt.plot(range(len(self.valid_losses)), self.valid_losses, color="g",label='valid_losses')
        plt.title('the loss change')
        plt.legend()
        plt.show()
        plt.plot(range(len(self.train_acces)), self.train_acces, color="y",label='train_acces')
        plt.plot(range(len(self.valid_acces)), self.valid_acces, color="b",label='valid_acces')
        plt.title('the acc change')
        plt.legend()
        plt.show()
    def save_model(self):
        '''
        save or load models(undetermined)
        :return:
        '''
        pass
    def load_model(self):
        '''
        undetermined
        :return:
        '''
        pass
    def print_parameters(self):
        print('use {} to run the model'.format(self.device))
        print(self.loss_fn)
        print(self.optimizer)


t = trainer(n='10',epochs=10,net=net,loss_fn=loss_fn,optimizer=optimizer,trainloader=trainloader,validloader=validloader)
t.fit()
t.print_history()
# t.print_()

# def train(dataloader):
#     losses,acc = [],[]  #以后会用来画图
#     for epoch in range(epochs):
#         net.train()
#         loss = 0
#         total,correct = 0,0
#         len_ = dataloader.__len__()
#         with tqdm(total=len_) as p_bar:
#             for i,data in enumerate(dataloader):
#                 features, labels = data
#                 features, labels = features.to(device), labels.to(device)
#
#                 optimizer.zero_grad()
#                 output = net(features)
#                 l = loss_fn(output, labels)
#                 l.backward()
#                 optimizer.step()
#
#                 predicted = output.argmax(dim=1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 loss += l.item()
#                 p_bar.update()
#                 p_bar.set_description("epoch:{}/{}, iteration:{}/{}, loss={:.3f}, "
#                                       .format(epoch+1,epochs, i+1,len_, l.item()))
#
#         ##val
#         net.eval()
#         with torch.no_grad():
#             val_correct = 0
#             val_total = 0
#             val_loss = 0
#             for j,data in enumerate(validloader):
#                 features, labels = data
#                 features, labels = features.cuda(), labels.cuda()
#                 output = net(features)
#                 l = loss_fn(output, labels)
#
#                 predicted = output.argmax(dim=1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()
#                 val_loss += l.item()
#
#         time.sleep(0.5)
#         time_ = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#         # print('%s epoch:%d/%d train_acc:%.3f%%' % (time_,epoch + 1,epochs,100 * correct / total))
#         print('{} epoch:{}/{} train_acc:{:.3f}% train_loss:{:.3f} val_acc:{:.3f}% val_loss:{:.3f}'
#               .format(time_,epoch+1,epochs,correct/total*100,loss/(i+1),val_correct/val_total*100,val_loss/(j+1)))
#         time.sleep(0.5)
# # train(trainloader)





