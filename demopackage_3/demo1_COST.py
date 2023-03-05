
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from torch.utils import data

T = 1000  # 总共产生1000个点
# 生成1—1000的一维向量
time = torch.arange(1, T + 1, dtype=torch.float32)
# 意思是生成0—10的1000个点（精度0.01）的sin值加上 mean std size(T,和T输出没区别，只是有时候T会报错)
# 这里的X是一个长度为1000的一维向量 [1000]
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))


# 这是马尔科夫假设，即数据和前tau个数据有关
tau = 4
# 从第五个数据开始才能构成tau [996,4]
features = torch.zeros((T - tau, tau))
# 对feature的四个维度（以最后一维来看），取
for i in range(tau):
    # 这里取单个元素，会导致降维，此时大小为[996]，以此加入[0,996],[1,997],[2,998],[3,999]。此时每个数据都是依次增大的四个数据
    features[:, i] = x[i: T - tau + i]
print(features.shape)
# 因为是从
labels = x[tau:].reshape((-1, 1))
print(labels.shape)

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练          作为一个tuple（*）输入
dataset = data.TensorDataset(*(features[:n_train],labels[:n_train]))
train_iter = data.DataLoader(dataset, batch_size, shuffle=True)
print(type(train_iter))
print(type(dataset))
print(len(dataset))
print(type(dataset[0]))
print(type(dataset[0][0]))
print(type(dataset[0][1]))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    #apply在nn.Moudle类使用时，一般都为初始化权重操作，其中操作用如if type(m) == nn.Linear的语句来判断如何和是否初始化
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')#这里引用了绘图类

net = get_net()
train(net, train_iter, loss, 100, 0.01)

result = []
for i in range(len(features)):
    result.append(float(net(features[i]).cpu()))
plt.plot([i for i in range(996)], labels, label='true', color='royalblue', linewidth=1)
plt.plot([i for i in range(996)],result,label='1-pred',color='m',linewidth=1,linestyle='--')
plt.title('true and 1-pred')
plt.grid()
plt.legend()
plt.show()

#后面的算法是使用训练好的网络来进行多次的预测（即预测多次以后的值），但是这个模型效果奇差
result_2 = torch.zeros(996,)
for i in range(600):
    result_2[i] = net(dataset[i][0])
for i in range(600-4,996-4):
    result_2[i+4] = net((result_2[i:i+4]).reshape(-1,4))
plt.plot([i for i in range(996)],result,label='1-pred',color='purple',linewidth=1,linestyle='--')
plt.plot([i for i in range(996)],labels,label='true',color='royalblue',linewidth=1)
plt.plot([i for i in range(996)],[float(i) for i in result_2],label='outrange_pred',color='crimson',linewidth=1,linestyle=':')
plt.title('true and 1-pred and outrange_pred')
plt.grid()
plt.legend()
plt.show()

result_4 = torch.zeros(996,)
result_16 = torch.zeros(996,)
result_64 = torch.zeros(996,)

result_mid = torch.zeros(996,)
for i in range(996-4):
    for a in range(i,i+4):
        result_mid[a] = net(features[a])
    for b in range(i+4,i+5):
        result_mid[b] = net(result_mid[b-4:b])
    result_4[i+4] = result_mid[i+4]

result_mid = torch.zeros(996,)
for i in range(996-16):
    for a in range(i,i+4):
        result_mid[a] = net(features[a])
    for b in range(i+4,i+17):
        result_mid[b] = net(result_mid[b-4:b])
    result_16[i+16] = result_mid[i+16]

result_mid = torch.zeros(996,)
for i in range(996-64):
    for a in range(i,i+4):
        result_mid[a] = net(features[a])
    for b in range(i+4,i+65):
        result_mid[b] = net(result_mid[b-4:b])
    result_64[i+64] = result_mid[i+64]

plt.plot([i for i in range(996)], labels, label='true', color='royalblue', linewidth=1)
plt.plot([i for i in range(996)], [float(i) for i in result_4], label='4-pred', color='crimson', linewidth=1, linestyle='--')
plt.plot([i for i in range(996)], [float(i) for i in result_16], label='16-pred', color='purple', linewidth=1, linestyle=':')
plt.plot([i for i in range(996)], [float(i) for i in result_64], label='64-pred', color='g', linewidth=1, linestyle='-.')
plt.grid()
plt.legend()
plt.show()



