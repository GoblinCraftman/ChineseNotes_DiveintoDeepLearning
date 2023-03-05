
import math
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

def mid():
    print('-'*30)

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    # 无长度要求时，直接计算输出；同时数据格式要求为：1d或者2d对应X的0或者0，1维度（数据最后一维大小会自动广播）
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        # 判断是否为一维向量，以决定是否进行扩展
        if valid_lens.dim() == 1:
            # 对于高维矩阵，valid—len为高维度对应位数的一维向量，此时将每一维度的限制扩展到X的每个样本的大小，从0维度到0，1维度相等
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        # pytorch_deep\demopackage_4\demo3_STS.py 94 valid—len展开为一维与X以遍历顺序对应
        # 计算过程中X变为二维，顺序按照三维遍历顺序，最后会reshape回原形状进行计算
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)# exp(-1e6)计算结果为0
        return nn.functional.softmax(X.reshape(shape), dim=-1)

mid()
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # Linear只会对维度大小进行改变，而不是改变维度
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    # 总览：将查询和键值对划入隐藏层后通过广播全部二维相加，再通过linear消除隐藏层回到原输出大小
    # 值的维度指的是每一个key可能对应多个value，此时value数量即称为维度
    # valid—len限制作用的提取长度
    def forward(self, queries, keys, values, valid_lens):
        # 与权重相乘
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，时间步，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        # (batch_size，查询的个数，“键－值”对的个数，num_hiddens)，此时即获得查询值对应的每个权重的注意力权重
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        # softmax会取出部分对于整体的占比
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        # 输出结果大小：(batch_size，查询的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

# （batch，timestep，queries—size）or （batch，queries—num，queries—height）/（batch，key—value—size，key—num）
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values的小批量，两个值矩阵是相同的
# (batch，key—value—size，value—num)
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    2, 1, 1)
valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention.eval()
mid()
# （batch，timestep，value）or （batch，queries—num，value）
print(attention(queries, keys, values, valid_lens))

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
plt.show()

# 使用点积可以得到计算效率更高的评分函数， 但是点积操作要求查询和键具有相同的长度d
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
mid()
print(attention(queries, keys, values, valid_lens))

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
plt.show()