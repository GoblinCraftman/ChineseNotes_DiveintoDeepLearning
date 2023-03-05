
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

train_imgs = torchvision.datasets.ImageFolder(r'../datapackage/hotdog/test/')
test_imgs = torchvision.datasets.ImageFolder(r'../datapackage/hotdog/train/')

print(type(train_imgs))
print(train_imgs)
print(train_imgs.class_to_idx)
print('-'*20)
print(type(test_imgs))
print(test_imgs)
print(test_imgs.class_to_idx)
print('-'*20)
print(type(train_imgs[0][0]))
print(train_imgs[0][1])

step = 0
img = 0
adress = 0
while True:
    if train_imgs[step][1] == 0:
        adress += 1
        plt.subplot(2,4,adress)
        plt.title(step)
        plt.imshow(train_imgs[step][0])
        img += 1
    step += 1
    if img >= 4:
        break

step = 0
img = 0
while True:
    if train_imgs[step][1] != 0:
        adress += 1
        plt.subplot(2, 4, adress)
        plt.title(step)
        plt.imshow(train_imgs[step][0])
        img += 1
    step += 1
    if img >= 4:
        break

plt.show()