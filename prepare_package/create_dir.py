
import os

path=r'..'
for t in ['train','valid']:
    os.mkdir(os.path.join(path, t))
    for folder in [r'dog',r'cat']:
        os.mkdir(os.path.join(path,t,folder))