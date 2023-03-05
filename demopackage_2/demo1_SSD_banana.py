
import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l
from matplotlib import pyplot as plt
from torch import nn

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

num_epochs, timer = 50, d2l.Timer()
net = net.to(device)
epoch_list = []
loss_list = []
class_list = []
offset_list = []
step = 0
for epoch in range(num_epochs):
    epoch_list.append(epoch)
    print('time {} start'.format(epoch))
    loss_sum = 0
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        #锚框，每个锚框的类别预测，每个锚框的offset预测
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        # [batch_size,anchors*4(offset)],[batch_size,anchors*4(sign)],[batch_size,anchors(对应种类+1)]
        # cla_labels其中数据值+1（因为有0作为没有）
        #真实的offset，有分配的锚框标记，真实的锚框类别
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        #[batch_size,num_ach_total,classes_num],_,[batch_size,num_achors_total*offsets_pred(4)],_,[batch_size,ach_num_total,4]
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        loss_sum += float(l.mean().cpu())
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    loss_list.append(loss_sum)
    cls_acc, bbox_mae = metric[0] / metric[1], metric[2] / metric[3]
    class_list.append(cls_acc)
    offset_list.append(bbox_mae)
    #此时预测的锚框有分配，但是一个真实框会有很多的匹配锚框，因此输出时我们使用nms选出其中最匹配的锚框即可

print(f'class err {cls_acc:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')

plt.subplot(3,1,1)
plt.title('train_loss')
plt.plot(epoch_list, loss_list, label = 'train_loss')
plt.grid()
plt.legend()

plt.subplot(3,1,2)
plt.title('class_acc')
plt.plot(epoch_list, class_list, label = 'class_acc')
plt.grid()
plt.legend()

plt.subplot(3,1,3)
plt.title('bbox_offset_mae')
plt.plot(epoch_list, offset_list, label = 'bbox_offset_mae')
plt.grid()
plt.legend()
plt.show()