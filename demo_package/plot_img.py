
import numpy
from matplotlib import pyplot as plt
import time
import pylab
import torchvision

def plot_img(image):
    image = image.numpy()[0]
    mean=0.1302
    std=0.3081
    image=((mean*image)+std)
    plt.imshow(image,cmap='gray')
    pylab.show()