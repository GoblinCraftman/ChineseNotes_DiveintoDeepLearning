
from torch import nn

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