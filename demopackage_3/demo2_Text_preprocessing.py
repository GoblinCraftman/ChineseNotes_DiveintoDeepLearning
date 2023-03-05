# 南京信息工程大学
# 信息工程
# 3班张木梓
# 2022-11-02 17:03
import collections
import re
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        #返回包含每行的列表
        lines = f.readlines()
    # 对lines中的数据line做正则化处理，其中[^A-Za-z]表示所有非字母str，+表示匹配所有一个及以上的目标，这句的操作是将所有非字母的长度从一开始的字符串替换为‘ ’（空格）
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])
print('-'*30)

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    # 一般的文本处理都是用这个，因为为了减小计算量并且使矩阵合理，是对于每个字母进行标号并且进行训练
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])


#构建词表
class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        # 对于非法输出，初始化使其不会报错
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 这一步会将tokens转换为一维向量，使其可以被识别
        counter = count_corpus(tokens)
        # 对于给出了次数的counter类，可以像dict一样操作，但是counter类不是dict且大多数操作不能进行，并且这里进行的是降序排序，最大的在前面、
        # 这个类似字典类，第一个值为类名，第二个为次数
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],reverse=True)
        # 未知词元的索引为0，然后加上保留的词表（这样每个词的索引就和每个词进行了对应，从1开始）
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 对保留词表的词进行索引分配，产生了对应的词表和索引以及值
        # 这个idx从0开始
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        # 这两个值为词和出现次数,  此时词表中只有reverse的单词，还没有对tokens里的继续判断
        for token, freq in self._token_freqs:
            # min_freq为词出现最小次数的阈值
            if freq < min_freq:
                break
            # 对于超过阈值的词并且没有记录的添加到保留词表中并且分配索引
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                # 最后一位的索引等于长度-1
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        # 返回存在多少个词在词表中
        return len(self.idx_to_token)

    # 这一步非常关键，重载了操作符[]，此时使用索引进行取值时会默认self.token_to_idx进行取值
    def __getitem__(self, tokens):
        # 输入不是list也不是tuple
        if not isinstance(tokens, (list, tuple)):
            # 这一步相当于返回tokens对应的value的第一个值
            return self.token_to_idx.get(tokens, self.unk)
        # 返回词表里的数据，如果是tuple或者list就拆开再返回
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        # 返回索引对应的数据，如果是tuple或者list就拆开再返回
        return [self.idx_to_token[index] for index in indices]

    #不知道这两个有什么用
    # 这两个相当于调用self.XXX属性时，返回何种数据
    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表，如果是二维的，将其展开为一维，操作代码如下
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        # 这个写法等同于 for lines in tokens：
        #                                 for token in line：
        #                                                   token
        tokens = [token for line in tokens for token in line]
    # 返回一个cunter类，返回出现的字符串和次数
    return collections.Counter(tokens)

print('-'*30)
vocab = Vocab(tokens)
print(len(vocab))
print(list(vocab.token_to_idx.items())[:10])
print(list(vocab.token_to_idx.items())[-10:-1])

print('-'*30)
for i in [0, 10]:
    print('文本:', tokens[i])
    # 在词表中查询所有的单词对应的索引并且输出
    print('索引:', vocab[tokens[i]])

def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    # 这里的tokens里面全是单词拆开的字母和空格
    tokens = tokenize(lines, 'char')
    # 因此生成的词表也只有27种元素（吧unk未知去掉的话）
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

print('-'*30)
corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))
print(corpus[:10])
# 这两个内容相当于列表内容和索引的转换
print(vocab.idx_to_token)
print(vocab.token_to_idx.items())
