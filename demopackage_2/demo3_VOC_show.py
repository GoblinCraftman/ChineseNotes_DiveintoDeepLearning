
import os
import numpy as np
import torch
import torchvision
from d2l import torch as d2l
from matplotlib import pyplot as plt

def imgs_show(imgs):
    w = len(imgs)
    judge = True
    if isinstance(imgs[0],list):
        h = len(imgs[0])
    else:
        judge = False
        h = 1
    if judge:
        for i in range(w):
            for r in range(h):
                plt.subplot(h,w,i+r*w+1)
                plt.imshow(np.array(imgs[i][r]))

    else:
        for i in range(w):
            for r in range(h):
                plt.subplot(h,w,i+r*w+1)
                plt.imshow(np.array(imgs[i]))
    plt.show()

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')#读取训练集或者测试集图片名称
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

voc_dir = r'../datapackage/VOCdevkit/VOC2012'
train_features, train_labels = read_voc_images(voc_dir, True)

n = 5
imgs = train_features[0:n] + train_labels[0:n]#list相加就是拼接起来
imgs = [img.permute(1,2,0) for img in imgs]#对于数据集中的文件来说，其格式为[batch_size,channel，h,w],因此进行维度变换为[batch_size,h,w,channel]
img_1 = imgs[0:n]
img_2 = imgs[n:]
img_3 = []
for num,i in enumerate(img_1):
    img_3.append([i,img_2[num]])
imgs_show(img_3)
imgs_show(img_1)
imgs_show(img_2)