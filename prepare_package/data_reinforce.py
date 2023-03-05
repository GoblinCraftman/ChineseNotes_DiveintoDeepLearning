
import random
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

img = Image.open('../pic/1.jpg')
print(type(img))
for i in range(6):
    plt.subplot(2,3,i+1)
    if i != 0:
        img1 = transforms.RandomRotation(90)(img)
        img1 = transforms.RandomHorizontalFlip()(img1)
        img1 = transforms.RandomVerticalFlip()(img1)
        img1 = transforms.ColorJitter(brightness=random.random(),contrast=random.random(),saturation=random.random(),hue=random.random()*(1/2))(img1)
        plt.imshow(img1)
    else:
        plt.imshow(img)
plt.show()
