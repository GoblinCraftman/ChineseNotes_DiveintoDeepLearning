
import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l
from matplotlib import pyplot as plt
from torch import nn
from  torch.nn import functional as F

def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = r'../datapackage/banana-detection'
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

#*******************************************************************************************

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

def forward(x, block):
    return block(x)

def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    #当为p in ps 时，数据从低到高分批输出,但是数据数量为一维大小
    #使用p in ps 会导致降维，但是数据此时是有五个拼接的五维数据，使得数据大小仍然符合四维
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X,   blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    #会自动读取Y的宽和高来生成锚框
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        #此时数据会从[batch_size,num_ach,4]变成[batch_size,num_ach*5（五次的数量不一样）,4],dim=1指的是在第一维拼接
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        #-1处为生成锚框数量,此时生成数据结构为[batch_size,num_ach_total,classes_num+1]
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        #数据结构为[batch_size,offsets_pred(4)*num_ach]
        bbox_preds = concat_preds(bbox_preds)

        return anchors, cls_preds, bbox_preds

batch_size = 32
train_iter, _ = load_data_bananas(batch_size)

device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    #[batch_size,num_ach_total,classes_num+1],_,[batch_size,offsets_pred(4)*num_ach],_,[batch_size,ach_num,4]
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    #open batch_size,and put all put all element into Two-dimensional vector
    cls = cls_loss(cls_preds.reshape(-1, num_classes),#[batch_size*anchors_num,num_classes+1]
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)#[batch_size,num_anchors_num]
    #out size is [batch_szie,1]
    #mention the background box and give 0 loss to it
    #in every batch anchors one by one
    bbox = bbox_loss(bbox_preds * bbox_masks,#[batch_size,anchors_num,4]
                     bbox_labels * bbox_masks).mean(dim=1)#[batch_size,anchors_num,4]
    #out is [batch_size,1]
    #return the mean of each row
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    # 返回anchors对应的classes,最重要的是此时label的真实框已经+1，这与pred里+1位作为未匹配刚好对应
    # [batch,ach_num,classes+1]->[batch,ach_num],and turn into int,   [batch，ach_num]
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    #cls_preds：[batch_size,num_ach_total（h*w）,classes_num+1]
    #这步在第三维（即dim=2）计算出0,1的数值占比，这时处于01的值即为p(0)和p(1)，再reshape为[batch_size,classes_num+1,num_ach_total](h*w)即获得概率在所有锚框上的分布
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    #[batch_size,classes_num+1,num_ach_total],[batch_size,anchors(h*w)*4(offset)],[batch_size,num_ach,4]
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)#[1，anchors,1+1+4](种类，准确率，四位坐标)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]#这样操作是因为在算法可靠程度较高的情况下，每一类都只会有一个输出，其他的被抑制
    #获得其中
    return output[0, idx]#返回自带种类的种类总数大小的[classes,6]的tensor

output = predict(X)

def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)