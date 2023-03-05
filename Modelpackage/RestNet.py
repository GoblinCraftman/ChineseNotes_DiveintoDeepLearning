
#跳过部分中间层，减少梯度传播
from torch import nn

def conv3x3(in_planes,out_planes,stride=1,groups=1,dilation=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes,out_planes,kernel_size=(3,3),stride=stride,padding=dilation)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']
    def __init__(self,inplanes,planes,stride=1,downsample=None,groups=1,
                 basic_width=64,dilation=1,norm_layer=None):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes,planes,stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            out += identity
            out = self.relu(out)
        return out

class DepthWiseSep(nn.Module):
    def __init__(self,nin,kernels_per_layer,nout):
        super(DepthWiseSep, self).__init__()
        self.depthwise = nn.Conv2d(nin,nin*kernels_per_layer,3,1,groups=nin)
        self.pointwise = nn.Conv2d(nin^kernels_per_layer,nout,1)

    def forward(self,x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out