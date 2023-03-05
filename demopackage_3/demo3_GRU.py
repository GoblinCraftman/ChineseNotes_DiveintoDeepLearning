import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        # 更新和重置门均使用sigmoid进行激活（这样操作会使值来到[0，1]，符合更新的需求）
        # [batchsize,vocabsize]X[vocabsize,numhid]->[batchszie,numhid]
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z) # 更新门
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r) # 重置门
        # 候选隐藏状态使用tanh激活
        # [batchsize,vocabsize]X[vocabsize,numhid]  [batchsize,numhid]*[batchsize,numhid]X[numhid,numhid]
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h) # 重置门参数用于获取候选隐藏状态
        H = Z * H + (1 - Z) * H_tilda # 更新门参数用于更新隐藏状态(参数本身决定保存多少原参数的比例)
        # [batchszie,numhid]X[numhid,vocabsize]
        Y = H @ W_hq + b_q # 提取特征进行输出
        outputs.append(Y)
    #                 输出大小为[timestep*batchszie，vocabsize]
    return torch.cat(outputs, dim=0), (H,)

# 自己写的GRU实现
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
#                          词表大小，隐藏层大小，数据类型，权重初始化函数，隐藏层初始化函数，前向传播网络
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# 掉包实现的GRU
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)