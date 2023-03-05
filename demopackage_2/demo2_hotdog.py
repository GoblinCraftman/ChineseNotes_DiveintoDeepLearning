
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

test_imgs = torchvision.datasets.ImageFolder(r'../datapackage/hotdog/test/',transform=train_augs)
test_size = len(test_imgs)
train_imgs = torchvision.datasets.ImageFolder(r'../datapackage/hotdog/train/',transform=test_augs)
train_size = len(train_imgs)

print(type(train_imgs))
print(train_imgs)
print(train_imgs.class_to_idx)
print('-'*20)
print(type(test_imgs))
print(test_imgs)
print(test_imgs.class_to_idx)
print('-'*20)

finetune_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)
finetune_net.cuda()

train_socre = []
test_socre = []
loss_socre =[]
loss_test = []
step = []

def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(train_imgs,batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_imgs,batch_size=batch_size, shuffle=True)

    loss = nn.CrossEntropyLoss()
    loss.cuda()

    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                  lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer,T_max=10)

    for i in range(num_epochs):
        loss_all = 0
        arcurate_all = 0
        print('train time {} start'.format(i + 1))
        net.train()
        for data in train_iter:
            imgs,targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            output = net(imgs)
            loss_mid = loss(output,targets)
            trainer.zero_grad()
            loss_mid.backward()
            trainer.step()

            arcurate = (output.argmax(1) == targets).sum()
            arcurate_all += arcurate.cpu()
            loss_all += loss_mid.cpu()
        with torch.no_grad():
            step.append(i + 1)
            train_socre.append(float(arcurate_all / train_size))
            loss_socre.append(float(loss_all))
        scheduler.step()

        loss_all = 0
        arcurate_all = 0
        net.eval()
        with torch.no_grad():
            for data in test_iter:
                imgs, targets = data
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    targets = targets.cuda()

                output = net(imgs)
                loss_mid = loss(output, targets)

                arcurate = (output.argmax(1) == targets).sum()
                arcurate_all += arcurate.cpu()
                loss_all += loss_mid.cpu()
            test_socre.append(float(arcurate_all / test_size))
            loss_test.append(float(loss_all))

train_fine_tuning(finetune_net, 5e-5,param_group=True,num_epochs=40)

plt.subplot(1, 2, 1)
plt.title('train_acc and test acc')
plt.plot(step, train_socre, label = 'train_acc')
plt.plot(step, test_socre, label = 'test_acc')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.title('train_loss and test loss')
plt.plot(step, loss_socre, label = 'train_loss')
plt.plot(step, loss_test, label = 'test_loss')
plt.grid()
plt.legend()

plt.show()




#scratch_net = torchvision.models.resnet18()
#scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
#train_fine_tuning(scratch_net, 5e-4, param_group=False)