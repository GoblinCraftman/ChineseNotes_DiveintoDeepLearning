
from torchvision.models import alexnet, AlexNet_Weights

#处理224*224
model = alexnet(weights=AlexNet_Weights.DEFAULT)
print(model)
