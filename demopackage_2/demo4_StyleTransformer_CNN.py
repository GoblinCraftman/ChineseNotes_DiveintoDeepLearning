
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from PIL import Image
from torchvision.models import VGG19_Weights

content_img = Image.open('../pic/style_in/c/1.jpg')
style_img = Image.open('../pic/style_in/s/1.jpg')

rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        #  [c,h,w]                        中值           平均值          out = (in -mean)/std
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    #                   [batch_size,c,h,w]
    return transforms(img).unsqueeze(0)

def postprocess(img):
    #[batch_size,c,h,w] -> [c,h,w]
    img = img[0].to(rgb_std.device)#cuda() to cpu()
    #                     [c,h,w] -> [h,w,c] 这一步的作用是使乘积作用于每个像素的三原色位置，否则会因为无法广播或者广播造成error或者错误的输出
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)#反归一化,指示合理的范围
    #                                          [h,w,c] -> [c.h.w] -> PIL.Image
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

pretrained_net = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT)

# 1, 6, 11, 20, 29      0, 5, 10, 19, 28
#                      maxpool in 4,9,18,27  and before 25 is ReLU
#                      [[64][128][256][512][512]]  [512]
style_layers, content_layers = [1, 6, 11, 20, 29], [25]

net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])

print(net)

#传入的X是已经处理过的图片
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):#len(net) = 28 + 1
        X = net[i](X)#沿着网络逐层对X进行处理，位置到达时提取特征
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

def get_contents(image_shape, device):
    #预处理
    content_X = preprocess(content_img, image_shape).to(device)
    #提取特征
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
#   返回 预处理图像（tensor）已提取的特征（[tensor]）
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y

#内容损失模块
#   MSEloss  均方误差   (pred(y)-y)**2/num
def content_loss(Y_hat, Y):
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。 Y.detach()分离了梯度，不会进行训练
    return torch.square(Y_hat - Y.detach()).mean()

#这个风格损失比较复杂，叫做格拉姆斯矩阵
#格拉姆斯矩阵用于表示特征之间的相关性度量
#操作为将矩阵扁平化（根据通道），然后计算该矩阵和其转置的积来获得相关性
def gram(X):
    #[batch_szie,channel,h,w],h*w     x里的所有元素/通道数
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    #重新处理图像，将图像转换为格拉姆斯矩阵，做扁平化处理
    X = X.reshape((num_channels, n))
    #使用matual时两个tensor均为二维时，这个操作返回矩阵乘法的结果
    #然后对其进行归一化使其更加具备反向传播的训练价值
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。 Y.detach()分离了梯度，不会进行训练
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

def tv_loss(Y_hat):
    #这一步比较关键，意思是用一个长宽均小于原矩阵1的矩阵在左上和右下位置进行选取然后相减，这和定义的减法效果相同并且使用了矩阵运算
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

#训练权重
content_weight, style_weight, tv_weight = 1, 1e4, 10

#可以通过调节超参数来改变图片的侧重方向
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、风格损失和全变分损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

# 与之前训练的网络不同，之前的网络训练的参数是每层网络里面的权重。但是在卷积神经网络的风格迁移中，唯一需要更新的变量是最后需要合成的图像。
# 我们可以定义一个简单的模型SynthesizedImage，并将合成的图像视为模型参数。模型的前向传播只需返回模型参数。
# 可以这样写是因为我们本身训练的就是nn.Module类（网络）的权重，这里初始化网络的权重大小来训练，网络本身的内容已经无意义了（因为网络本身除开权重没有进行任何输出）
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight

def get_inits(X, device, lr, styles_Y):
    # 这里初始化了X的权重为大小[channel,h,w]的矩阵，gen_img是X的权重
    gen_img = SynthesizedImage(X.shape).to(device)
    # 这一步对权重进行了初始化。初始化数据为图片的内容，因此最开始的输出会和我们初始化的图片基本一样
    gen_img.weight.data.copy_(X.data)
    # 这里的优化器对于网络进行优化，优化的内容是权重（也就是图像内容）
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    # 对所有风格层的每层数据做格拉姆斯矩阵运算
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    # 这里的返回值也非常关键，我们发现return是gen_img()，意思是返回的是gen_img的权重，然后对应的就是合成图片的内容
    return gen_img(), styles_Y_gram, trainer

# X是内容图片的预处理后的数据
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    # 这一步非常关键，具体为创建了一个具备图像的大小的权重矩阵gen_img
    # 注意，此时X已经进行了数据更新，之前X是预处理图像，现在X是gen_img类的做了初始化的权重
    # 这里获得了网络权重（内容为合成图片），原风格图片的风格的格拉姆斯矩阵计算结果（避免多次计算，因为数值不变），指向网络的优化器
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    # 设置了一个以五十为步长的学习率减小优化器
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    for epoch in range(num_epochs):
        if (epoch + 1) % 10 == 0:
            print('train time {} start'.format(epoch+1))
        trainer.zero_grad()
        # 重新提取的特征（此时为网络权重的特征）
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        # 计算网络权重的内容和风格相较于原先提取的内容图片内容和风格图片风格（这里已经提前转换为了格拉姆斯矩阵结果）的损失
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        #开始优化
        l.backward()
        trainer.step()
        scheduler.step()
        if epoch == (num_epochs-1):
            postprocess(X).save('../pic/style_out/13.jpg')
    return X

#1080, 1920
#392, 720
# 给定初始参数
device, image_shape = d2l.try_gpu(), (1080, 1920)
# cuda化
net = net.to(device)
# 这一步很复杂，按照顺序是 transform预处理图像 传入网络获得提取的内容和风格（存储在list里）
#   预处理后的图片
# [batch_size,c,h,w]
# 这里只提取了内容图片的预处理图片和内容层数据
content_X, contents_Y = get_contents(image_shape, device)#输入参数暂时是不可以调整的，如果想要改变输入的图片需要去最开始的读取路径区域修改
# 这一步提取了风格图片的风格层数据（这个函数也会提取风格图片的预处理数据，但是我们不需要所以最开始是_）
_, styles_Y = get_styles(image_shape, device)
# 传入需要的开始训练的数据
#  [batch_size,c,h,w] [内容层数据，大小暂时不看]   [风格层数据，大小也暂时不看]
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)




