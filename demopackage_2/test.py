
import torchvision
from torch import nn
from torchvision.models import ResNet18_Weights

finetune_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
params = [param for name, param in finetune_net.named_parameters()]
params_1x = [param for name, param in finetune_net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
print(len(params))
print(len(params_1x))

for name, param in finetune_net.named_parameters():
    print(name)