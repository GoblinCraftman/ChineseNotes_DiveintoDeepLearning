
import torch
import torch.cuda
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from torchvision.models import vgg19, VGG19_Weights

imsize = 512
is_cuda = torch.cuda.is_available()

prep = transforms.Compose([
    transforms.Resize((imsize,imsize)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
    #减去均值
    transforms.Normalize(mean=[0.40760392,0.45795686,0.48501961],std=[1,1,1]),
    transforms.Lambda(lambda x: x.mul_(255)),
    #更换三原色顺序至BGR
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])
    ])

postpa = transforms.Compose([
    transforms.Lambda(lambda x: x.mul_(1./255)),
    #加上均值
    transforms.Normalize(mean=[-0.40760392,-0.45795686,-0.48501961],std=[1,1,1]),
    #RGB
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])
    ])

postpb = transforms.Compose([transforms.ToPILImage()])

def postp(tensor):
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img

def img_loader(img_path):
    img = Image.open(img_path)
    img = Variable(prep(img))
    #from torch.Size([3, 618, 512]) to torch.Size([1,3, 618, 512])
    img = img.unsqueeze(0)
    return img

img_s = img_loader('../pic/style_in/s/1.jpg')
img_c = img_loader('../pic/style_in/c/1.jpg')
#噪声图片
opt_img = Variable(img_c.data.clone(),requires_grad=True)

vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
for param in vgg.parameters():
    param.requires_grad = False

criterion = nn.MSELoss()

class GramMatrix(nn.Module):
    def forward(self,input):
        b,c,h,w = input.size()
        features = input.view(b,c,h*w)
        gram_matrix = torch.bmm(features,features.transpose(1,2))
        gram_matrix.div_(h*w)
        return gram_matrix

class Styleloss(nn.Module):
    def forward(self,inputs,targets):
        out = nn.MSELoss()(GramMatrix()(inputs),targets)
        return (out)

class LayerActivations():
    features = []
    def __init__(self,model,layer_nums):
        self.hooks = []
        for layer_num in layer_nums:
            self.hooks.append(model[layer_num].register_forward_hook(self.hook_fn))
    def hook_fn(self,module,input,output):
        self.features.append(output)
    def remove(self):
        for hook in self.hooks:
            hook.remove()

def extract_layers(layers,img,mode1=None):
    la = LayerActivations(mode1,layers)
    #清除缓存(上一个数据）
    la.features = []
    out = mode1(img)
    #清楚当前函数内部缓存
    la.remove()
    return la.features

style_layers = [1,6,11,20,25]
content_layers = [21]

target_c = extract_layers(content_layers,img_c,mode1=vgg)
target_s = extract_layers(style_layers,img_s,mode1=vgg)

content_targets = [t.detach() for t in target_c]
style_targets = [GramMatrix()(t).detach for t in target_s]

targets = content_targets + style_targets
loss_layers = style_layers + content_layers

style_weights = [1e3/n**2 for n in [62,128,256,512,512]]
content_weights = [1e0]
weights = style_weights + content_weights

print(vgg)

loss_fns = [Styleloss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)

optimizer = optim.LBFGS([opt_img])

max_iter = 500
show_iter = 50
n_iter = [0]

while n_iter[0] <= max_iter:
    def closure():
        optimizer.zero_grad()
        out = extract_layers(loss_layers,opt_img,mode1=vgg)
        layer_losses = [weights[a] * loss_fns[a](A,targets[a]) for a,A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter[0]+=1
        #打印损失值
        if n_iter[0] % show_iter == (show_iter-1):
            print('Iteration: %d, loss: %f' % (n_iter[0]+1,loss.data[0]))

        return loss
    optimizer.step(closure)