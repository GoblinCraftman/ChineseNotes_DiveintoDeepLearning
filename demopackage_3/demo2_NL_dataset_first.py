
import random
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

# 默认是读取为单词向量，有行数的二维矩阵
tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])

# freqs是不同单词的出现频率
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
plt.show()

print('-'*30)
# 这里浅显的表示了二元组的含义：前后相连的词组
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
# 对二元词组进行计数
bigram_vocab = d2l.Vocab(bigram_tokens)
# 统计出现最多的二元词组
print(bigram_vocab.token_freqs[:10])

print('-'*30)
# 和上方基本相同
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

# 注意构造多元词组会导致整体长度衰减

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
plt.show()

#         二维单词词向量（都是对应索引）  批大小  允许随机进行的缩进量范围
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1(randint最后一位不取)
    # 从位移量（num_step）到最后的切片，因为索引从0开始，所以-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签（在tau个数据之后的那个）（测试会生成多少个一维词向量）
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    # 这个操作用于取出单独的向量来show或者分析
    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    # 一个能生成多少个batch在指定的batchsize下（每一步都不包含多余的部分，向下取整）
    num_batches = num_subseqs // batch_size
    # 开始分配batch，间距为batchsize
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        #                           这个初始化了的索引中，都为长度为step的向量，以其为单位分配给batchsize
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        # Y = crops[j + 1]
        Y = [corpus[j + 1] for j in initial_indices_per_batch]
        # 这里X返回的是词向量，y返回的是label（对应X中每个值的label，长度相同）,每个大小是batchszie*numstep
        # 并且注意这里返回的是两个包含tensor的XY的两个生成器，生成器需要转换类型才能直接使用
        yield torch.tensor(X), torch.tensor(Y)

my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

#             二维单词词向量（都是对应索引）  批大小  允许随机进行的缩进量范围
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    # 原理和上面的类似，但是这里就没有进行random.shuffle的打乱
    offset = random.randint(0, num_steps)
    # -1给label留出位置，然后算出要取多少个（在batch_size下向下取整）
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    # reshape时，数据依次排列，因此返回了batchsize个batchsize分的矩阵
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    # 用有多少个batchszie除以tau获得batch数量（本质是一个batch会有tau*batchsize个数据）
    num_batches = Xs.shape[1] // num_steps
    # 这里让i对应提取出来的Xs，Ys的索引
    for i in range(0, num_steps * num_batches, num_steps):
        # 因为没有提取单个数据，此时返回的生成器中的xy数据仍然是二维的，且大小为[batch_szie,num_steps]
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        # 这里返回的x，y的batch中的数据是从中间分开的，分别向后取
        yield X, Y


for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        # 选择是否使用随机队列，并且开始初始化方法
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        # 获得所有二维单词向量和其的词表(max_tokens==-1,all)
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        # 选择类的此属性时，迭代器开始运行并且返回data和label两个迭代对象
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

