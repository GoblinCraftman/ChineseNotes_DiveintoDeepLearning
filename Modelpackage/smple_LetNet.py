
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d,MaxPool2d, Flatten,Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

dataset_train=torchvision.datasets.CIFAR10(r'./dataset',train=False,transform=torchvision.transforms.ToTensor()
                                           ,download=True)
dataset_test=torchvision.datasets.CIFAR10(r'./dataset',train=False,transform=torchvision.transforms.ToTensor()
                                          ,download=True)
train_size=len(dataset_train)
test_size=len(dataset_test)

print('训练集长度为{}'.format(train_size))
print('测试集长度为{}'.format(test_size))

dataloader_train=DataLoader(dataset_train,batch_size=64,shuffle=True)
dataloader_test=DataLoader(dataset_train,batch_size=64,shuffle=True)

class zmz(nn.Module):
    def __init__(self):
            super(zmz, self).__init__()
            self.modle=nn.Sequential(
            Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            MaxPool2d(kernel_size=2,ceil_mode=False),
            Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            MaxPool2d(kernel_size=2, ceil_mode=False),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, ceil_mode=False),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
            )
    def forward(self,input):
        output=self.modle(input)
        return output
Z=zmz()
if torch.cuda.is_available():
    Z=Z.cuda()

loss_fn=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn=loss_fn.cuda()

learnrate=0.01
optimizer=torch.optim.SGD(Z.parameters(),lr=learnrate)

writer=SummaryWriter(r'./logs')
train_time=0
tset_time=0
epoch=100
step=0
step2=0

starttime=time.time()
for i in range(epoch):
    print('train time {} start'.format(i + 1), end=' ')
    running_loss = 0.0
    Z.train()
    for data in dataloader_train:
        imgs,targets=data
        if torch.cuda.is_available():
            imgs=imgs.cuda()
            targets=targets.cuda()
        output=Z(imgs)
        loss=loss_fn(output,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss = loss.item() + running_loss
        step+=1
        writer.add_scalar(r"train loss trend",loss.item(),step)
    writer.add_scalar(r"part train loss trend", running_loss,i+ 1)
    print('loss is {}'.format(running_loss), end=' ')

    running_loss = 0.0
    arcurate_all=0.0
    Z.eval()
    with torch.no_grad():
        for data in dataloader_test:
            img,targets=data
            if torch.cuda.is_available():
                img = img.cuda()
                targets = targets.cuda()
            output=Z(img)
            loss=loss_fn(output,targets)
            running_loss = loss.item() + running_loss
            arcurate=(output.argmax(1)==targets).sum()
            arcurate_all+=arcurate
            step2 += 1
            writer.add_scalar(r"test loss trend", loss.item(), step2)
            writer.add_scalar(r"test accuracy rate", arcurate, step2)
        writer.add_scalar(r"part test loss trend", running_loss,i + 1)
    print(',and loss in test part is {},accuracy rate is {}'.format(running_loss,arcurate_all/train_size))

writer.close()
endtime=time.time()
time=starttime-endtime
torch.save(Z,r'C:\Pythondata\PyTorch\demolist\train_model\CIFA10model_100.pth')
print('using time {}'.format(time))