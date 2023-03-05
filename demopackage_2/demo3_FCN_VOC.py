
import os
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torchvision.models import ResNet18_Weights


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

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
#[11,3]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

#转换入哈希表
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        #熟知以下种类哈希表的构建，可能很常用
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i#因为最大值每个RGB是256，对任意的[3,256]的数据，可以保证任何数据对于该哈希表而言绝对不会冲突
        #其中i代表种类
    return colormap2label

#换出哈希表
#               已经转换了色彩的图片   映射算法
def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    #[channel,h,w]->[h,w,channel]
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')#先对tensor进行换维，然后转换为int的numpy（tensor不能直接show，图片一般都是numpy）
    #提起单个元素会导致通道的维度消失
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])#上表中颜色已经在colorlabel中对应了label，因此计算后坐标位置与label对应
    #colormap的提取为一个[h,w]的矩阵，每个位置是该像素对应的label
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))#该函数用来获取四维坐标，rect是其返回值
    #对提取出来的裁剪框进行裁剪
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ( ' train ' if is_train else ' test ' ) + 'examples')

    #对已经进行简单归一化的像素再次进行归一化
    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    #对图片大小进行筛选，保证其中仅有大小大于裁剪尺寸的图像
    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        #      原图片（tensor）       转化后的像素点为class的图片格式的tensor
        #      [channal,h,w]        [h,w]
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    voc_dir = r'../datapackage/VOCdevkit/VOC2012'
    num_workers = 0
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter


pretrained_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
print(type(pretrained_net))

#取出其中最后一个层以及剩下两个（池化层以及连接层）
net = nn.Sequential(*list(pretrained_net.children())[:-2])
print(net)

#net将会将图片的大小计算为原先的1/32，因此使用转置卷积来恢复
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,kernel_size=64, padding=16, stride=32))

#双线性差值初始化（可不看）
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
#                             [in_channel,out_channel,kernel_size]
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))


W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)


batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = load_data_voc(batch_size, crop_size)


loss1 = nn.CrossEntropyLoss(reduction='mean')
if torch.cuda.is_available():
    loss1.cuda()

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

epoch_list = []
train_loss = []
test_loss = []
train_acc = []
test_acc = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

for epoch in range(num_epochs):
    print('train time {} start'.format(epoch+1))
    epoch_list.append(epoch+1)
    net.train()
    for data in train_iter:
        img_,label_ = data
        img = img_.to(device)
        label = label_.to(device)
        output = net(img)
        loss = loss1(output,label)
        trainer.zero_grad()
        loss.backward()
        trainer.step()
        train_loss.append(float(loss.cpu()))
        train_acc.append(float((output.argmax(dim=1) == label).sum() / (label_.shape[0] * label_.shape[1])))

    net.eval()
    with torch.no_grad():
        for data in test_iter:
            img_,label_ = data
            img = img_.to(device)
            label = label_.to(device)
            output = net(img)
            loss = loss1(output, label).mean(1).mean(1)
            test_loss.append(float(loss.cpu()))
            test_acc.append(float((output.argmax(dim=1) == label).sum() / (label_.shape[0] * label_.shape[1])))

#torch.save()

plt.subplot(1,2,1)
plt.title('train_loss and test_loss')
plt.plot(epoch_list,train_loss,label='train_loss')
plt.plot(epoch_list,test_loss,label='train_loss')
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.title('train_acc and test_acc')
plt.plot(epoch_list,train_acc,label='train_acc')
plt.plot(epoch_list,test_acc,label='train_acc')
plt.grid()
plt.legend()

plt.show()


