
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Dropout2d, Dropout ,ReLU
from torch.nn.functional import  log_softmax
from torch.utils.data import DataLoader
from torchvision import datasets
#图片显示模块
from plot_img import *
import time
if __name__ == '__main__':
    time_start=time.time()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.1307),(0.3081))])
    dataset1 = datasets.MNIST('../datapackage/',train=True,transform=transform,download=True)
    dataset2 = datasets.MNIST('../datapackage/',train=False,transform=transform,download=True)
    dataset_train=DataLoader(dataset1,batch_size=32,shuffle=True,num_workers=0)
    dataset_test=DataLoader(dataset2,batch_size=32,shuffle=True,num_workers=0)

    #(1,28,28)
    class Net(nn.Module):
        def __init__(self):
                super(Net, self).__init__()
                self.modle=nn.Sequential(
                Conv2d(in_channels=1,out_channels=10,kernel_size=5,stride=1),
                MaxPool2d(kernel_size=2,ceil_mode=False),
                ReLU(),
                Conv2d(in_channels=10,out_channels=20,kernel_size=5,stride=1),
                MaxPool2d(kernel_size=2, ceil_mode=False),
                Dropout2d(0.2),
                ReLU(),
                Flatten(),
                Linear(320,50),
                ReLU(),
                Dropout(0.2),
                Linear(50,10)
                )
        def forward(self,input):
            output=self.modle(input)
            return output

    train_size=len(dataset1)
    test_size=len(dataset2)

    def fit(epoch, model, data_train, data_test):
            net = model()
            learnrate = 0.01
            optimizer = torch.optim.SGD(net.parameters(), lr=learnrate)
            if torch.cuda.is_available():
                net = net.cuda()
            loss_fn = nn.CrossEntropyLoss()
            if torch.cuda.is_available():
                loss_fn = loss_fn.cuda()
            for i in range(epoch):
                print('train time {} start'.format(i + 1), end=':')
                net.train()
                running_loss = 0
                running_currecr = 0
                arcurate_all = 0

                for img in data_train:
                    data,target = img
                    data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)
                    output = net(data)
                    loss = loss_fn(output, target)
                    running_loss+=loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print('loss is {}'.format(running_loss), end=',')

                net.eval()
                with torch.no_grad():
                    for data in data_test:
                        img, target = data
                        if torch.cuda.is_available():
                            img = img.cuda()
                            target = target.cuda()
                        output = net(img)
                        loss = loss_fn(output, target)
                        running_loss = loss.item() + running_loss
                        arcurate = (output.argmax(1) == target).sum()
                        arcurate_all += arcurate
                print('and loss in test part is {},accuracy rate is {}'.format(running_loss, arcurate_all / test_size))
                if i == epoch-1 :
                    torch.save(net.state_dict(), 'MINST_50_static.pth')

    fit(50,Net,dataset_train,dataset_test)
    time_end=time.time()
    print('using time:{}'.format(time_end-time_start))


