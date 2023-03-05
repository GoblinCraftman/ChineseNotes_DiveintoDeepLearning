
#对卷积进行拼接（增加层）
#预处理时图像大小不变（指的是输出大小不变）
#该算法可以进行上下和不变的采样，但是中间层较多
import torch
from torch import nn

class BasicConv2d(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes,out_planes,kernel_size,stride,padding)
        self.bn = nn.BatchNorm2d(out_planes,eps=0.001,momentum=0.1,affine=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Inception_A(nn.Module):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(64,96,1,1)
        self.branch1 = nn.Sequential(
            BasicConv2d(64,64,1,1),
            BasicConv2d(64,96,3,1,1))
        self.branch2 = nn.Sequential(
            BasicConv2d(64,64,1,1),
            BasicConv2d(64,96,3,1,1),
            BasicConv2d(96,96,3,1,1))
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3,1,1,count_include_pad=False),
            BasicConv2d(64,96,1,1))
    def forward(self,x):
        x0 = self.branch0(x)
        print(x0.shape)
        x1 = self.branch1(x)
        print(x1.shape)
        x2 = self.branch2(x)
        print(x2.shape)
        x3 = self.branch3(x)
        print(x3.shape)
        out = torch.cat((x0,x1,x2,x3),1)
        print(out.shape)
        return out

net = Inception_A().cuda()
x = torch.rand(size=(1, 64, 224, 224))
out = net(x.cuda())
print(x)

