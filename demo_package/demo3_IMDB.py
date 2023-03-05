
import torch
import string,re
import torchtext
from torchtext.datasets import IMDB, imdb
import torchtext

MAX_WORDS = 10000  # 仅考虑最高频的10000个词
MAX_LEN = 200  # 每个样本保留200个词的长度
BATCH_SIZE = 20



train_iter, test_iter = IMDB(split=('train', 'test'))
print(type(train_iter))
print(type(train_iter))

from torchtext.datasets import IMDB

train_iter = IMDB(split='train')

def tokenize(label, line):
    return line.split()

tokens = []
for label, line in train_iter:
    tokens += tokenize(label, line)
