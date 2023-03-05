import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

def mid():
    print('-'*30)

n_train = 50  # 训练样本数
# 获得50个处于（0，1）的放大5倍的随机数据
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本,这里只获取值而不获取索引
print(x_train)

def f(x):
    return 2 * torch.sin(x) + x**0.8

# 2sin(x) + x**0.8 + 噪音            均值0，方差1的正态分布的噪音
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
# 获得大小为50的均匀样本
x_test = torch.arange(0, 5, 0.1)  # 测试样本 我们会发现样本的所有结果都存在误差，这个误差就是normal参数
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
print(n_test)

def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    plt.legend()
    plt.show()

y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)

# X_repeat的形状:(n_test,n_train),
# 每一行都包含着相同的测试输入（例如：同样的查询）
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
# [n_test,n_train]   得出训练值和查询值的距离，获得每个查询值和所有训练值的注意力参数
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# 在x-train大小上计算，返回大小为n-test[n_train]
# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
# 如果第一个参数是二维的，而第二个参数是一维的，则返回矩阵向量积。最后会求和，相当于加权求和
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)

d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
plt.legend()
plt.show()

X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
print(torch.bmm(X, Y).shape)# torch.Size([2, 1, 6])

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 存储在Parameters（）迭代器中，初始化时设置学习参数
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
    #                   查询、   键、    值
    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数),这一步操作的含义是让每个数都扩展成键值对个数个
        # 扩展大小至[querise，keys]，使每个查询对应所有建，计算所有距离
        # 这里可以增加学习参数数量到queries-keys的数量（train）
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        # 计算注意力参数
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # bmm：批量矩阵乘法
        # values的形状为(查询个数，“键－值”对个数),attention的大小升至（n_test,1,n_train）,value大小变为(查询个数，“键－值”对个数，1)，结果大小会变为（1，1），结果为（查询个数）
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

# 传入网络之前进行扩展（格式修整）
# 此时默认queries的大小和key—value大小相同
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# 这样的作用是：使得每个总体的softmax样本不同，过滤掉自己，进而避免数据在该样本上过拟合，此时听过对角矩阵取反操作，过滤掉了（n，n）的位置的所有元素
# keys的形状:('n_train'，'n_train'-1),？（使用对角矩阵有助于获得对应位置的权重）
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)，同上
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.25)
scheduler = torch.optim.lr_scheduler.StepLR(trainer, 1000, 0.8)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(500):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    scheduler.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))

# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
# 保留数据，消除梯度
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
plt.show()

