
import torch
from torch import nn
from torch.nn import Conv2d,MaxPool2d, Flatten,Linear
import torchvision.transforms
from PIL import Image

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_path=r"../hymenoptera_data/test/test1.jpg"
image=Image.open(img_path)
print(image)

image=image.convert('RGB')

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                          torchvision.transforms.ToTensor()])

image=transform(image)
print(image.shape)

class zmz(nn.Module):
    def __init__(self):
            super(zmz, self).__init__()
            self.modle=nn.Sequential(
            Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            MaxPool2d(kernel_size=2,ceil_mode=False),
            Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            MaxPool2d(kernel_size=2, ceil_mode=False),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, ceil_mode=False),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
            )
    def forward(self,input):
        output=self.modle(input)
        return output

model=torch.load(r'..\train_model\CIFA10model_100.pth')
print(model)
model=model.to(device)

image=torch.reshape(image,(1,3,32,32))
image=image.to(device)
with torch.no_grad():
    output=model(image)
print(output)
print(output.argmax(1))