
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Dropout2d, Dropout ,ReLU
from torch.nn.functional import  log_softmax
from torch.utils.data import DataLoader
from torchvision import datasets
#图片显示模块
from torchvision.models import VGG16_Weights

from plot_img import *
import time

if __name__ == '__main__':
    time_start=time.time()
    vgg16_True = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
    torch.load(r'../train_model/vgg16/vgg16.features.state_dict().pth')
    transform = torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Resize((224, 224)),
                                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                 [0.229, 0.224, 0.225])
                                              ])
    dataset1 = datasets.CIFAR10('../datapackage/CIFAR10/',train=True,transform=transform,download=True)
    dataset2 = datasets.CIFAR10('../datapackage/CIFAR10/',train=False,transform=transform,download=True)
    dataset_train = DataLoader(dataset1,batch_size=32,shuffle=True,num_workers=6)
    dataset_test = DataLoader(dataset2,batch_size=32,shuffle=True,num_workers=6)
    print(dataset1[0][0].shape)

    Z = vgg16_True
    if torch.cuda.is_available():
        Z = Z.cuda()

    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    learnrate = 0.01
    optimizer = torch.optim.SGD(Z.classifier.parameters(), lr=learnrate)

    train_time = 0
    tset_time = 0
    epoch = 100
    step = 0
    step2 = 0
    train_size=len(dataset2)

    starttime = time.time()
    for i in range(epoch):
        print('train time {} start'.format(i + 1), end=' ')
        running_loss = 0.0
        Z.train()
        for data in dataset_train:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = Z(imgs)
            loss = loss_fn(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = loss.item() + running_loss
            step += 1
        print('loss is {}'.format(running_loss), end=' ')

        running_loss = 0.0
        arcurate_all = 0.0
        Z.eval()
        with torch.no_grad():
            for data in dataset_test:
                img, targets = data
                if torch.cuda.is_available():
                    img = img.cuda()
                    targets = targets.cuda()
                output = Z(img)
                loss = loss_fn(output, targets)
                running_loss = loss.item() + running_loss
                arcurate = (output.argmax(1) == targets).sum()
                arcurate_all += arcurate
                step2 += 1
        print(',and loss in test part is {},accuracy rate is {}'.format(running_loss, arcurate_all / train_size))

    endtime = time.time()
    time = starttime - endtime
    print('using time {}'.format(time))
