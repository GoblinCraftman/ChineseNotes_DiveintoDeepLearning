
import torch
from torch import nn
from torch.nn import functional as F

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layers = self._make_layers()

    def forward(self,x):
        y = self.layers(x)
        return y

    def _make_layers(self):
        cfg = [64,64,'M',128,128,'M',256,256,'M',512,512,512]
        layers = []
        in_channels =3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(2,2,ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels,x,3,1),nn.ReLU(inplace=True)]
                in_channels = x
        print(layers)
        return nn.Sequential(*layers)

vgg = VGG16()
print(vgg)

class L2Norm(nn.Module):
    def __init__(self,in_features,scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)
    def reset_parameters(self,scale):
        nn.init.constant(self.weight,scale)
    def forward(self,x):
        x = F.normalize(x,dim=1)
        scale = self.weight[None,:,None,None]
        return x*scale

class VGG16Extractor300(nn.Module):
    def __init__(self):
        super(VGG16Extractor300, self).__init__()

        self.features = VGG16()
        self.norm4 = L2Norm(512,20)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv6 = nn.Conv2d(512, 1024 ,3, 6, 6)
        self.conv7 = nn.Conv2d(512, 1024, 1)

        self.conv8_1 = nn.Conv2d(1024,256,1)
        self.conv8_2 = nn.Conv2d(256,512,3,2,1)

        self.conv9_1 = nn.Conv2d(512,128,1)
        self.conv9_2 = nn.Conv2d(123,256,3,2,1)

        self.conv10_1 = nn.Conv2d(256,128,1)
        self.conv10_2 = nn.Conv2d(128,256,3)

        self.conv11_1 = nn.Conv2d(256,128,1)
        self.conv11_2 = nn.Conv2d(128,256,3)

    def forward(self,x):
        hs = []
        h = self.features(x)
        hs.append(self.norm4(h))#38*18

        h = F.max_pool2d(h,2,2,ceil_mode=True)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h,3,1,1,ceil_mode=True)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)#19*19

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)#10*10

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)#5*5

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)#3*3

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)#1*1

        return hs

