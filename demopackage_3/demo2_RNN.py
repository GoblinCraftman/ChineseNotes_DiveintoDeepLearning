import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 将（数据大小-1）//num_step//batch_szie，即可获得batch_num
batch_size, num_steps = 32, 35
# 默认返回字母
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

print(F.one_hot(torch.tensor([0, 2]), len(vocab)))

# 对于字母生成独热编码来说，28包括26个字母，1个未知，1个空格
# 大小为（批量大小，时间步数）
X = torch.arange(10).reshape((2, 5))
# reshape之后的输出（时间步数。批量大小，独热位数），此时遍历就能得到每个步数下的所有批量的输出
print(F.one_hot(X.T, 28).shape)

# 初始化权重，以normal的正态分布为基准
def get_params(vocab_size, num_hiddens, device):
    # 输入就是词表大小，而预测结果的输出也是词表大小（因为输出格式相同）
    num_inputs = num_outputs = vocab_size

    # 初始化函数，0.01倍的正态分布数字
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    # 其中w为更新参数中权重部分，b对应其中的常值部分,大小设置的原因是因为矩阵乘法
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 至此之上的有原本的注释，笔记本查看

# 初始化隐藏层权重（因为在0的时候是没有隐藏状态的）
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 这一步非常关键，说明了训练对象和输入的batchsize的大小无关，batchsize只和初始化大小有关（其中全是0）
def rnn(inputs, state, params):
    # time_step由input传入
    # inputs的形状：(时间步数量，批量大小，词表大小)，state就是初始化的隐藏状态（长为1的含有2个元素的tuple）
    # 将权重依次取出
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    # 这一步的实际是逐步输入每一时间步的对应批大小的数据，即在每一时间步下对所有批量对象的预测结果
    for X in inputs:
        # torch.mm执行矩阵乘法  [batch_size,input]*[input,hid]，注意mm不能广播，必须严格符合定义，即[a,b]*[b,c]->[a,c]
        # 此时的对应关系为：[batch_size,data(len)]*[weight(len),hid_num],b_h会自动广播，因为是加法
        # 结果大小为[batch_size,hid_num]
        # 使用输入数据x和原隐藏层参数h来对H进行更新,并且使用tanh作为激活函数
        # 这里的原理涉及线性代数和机器学习
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # 使用更新的隐藏层进行预测[batch_szie,hid_num]*[num_hidden,input]->[batch_size,input],b_q自动广播
        # 即输出只和更新的隐藏层H有关
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    # 这样写是是因为output里面全是tensor
    # 返回预测结果和更新后的隐藏层权重（错一位进行预测）
    # 其大小为[time_step*batch_size,独热编码位数]，这些数据都是逐步预测的
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    #                  独热编码位数，隐藏层大小，计算数据类型，权重初始化方法，初始化隐藏层方法，前向传播网络
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        # 获得权重初始化参数
        self.params = get_params(vocab_size, num_hiddens, device)
        # 加载隐藏层和网络方法（这里作为方法是因为不会随机变化，state全是0）
        self.init_state, self.forward_fn = init_state, forward_fn

    # 这个__call__相当于forward
    def __call__(self, X, state):
        # 对输入数据进行转化为独热编码
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    # batchsize在这里传入，这个参数会限制输入的数据格式
    def begin_state(self, batch_size, device):
        # 初始化state（隐藏层）
        return self.init_state(batch_size, self.num_hiddens, device)

# 类似linear的中间参数
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
# X = torch.arange(10).reshape((2, 5))
# reshape之后的输出（时间步数。批量大小，独热位数），此时遍历就能得到每个步数下的所有批量的输出
# 此时时间步只有2，意味着只有一个预测结果输出
state = net.begin_state(X.shape[0], d2l.try_gpu())
# new_state是一个含一个tensor的tuple
Y, new_state = net(X.to(d2l.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape)


# vocab is class,
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    # 这一步获取了字母的独热编码序号，但是实际训练时，输入的数据直接就是已经转换的结果
    outputs = [vocab[prefix[0]]]
    # 这一步将输入初始化，因为数据会由input提取，这里相当于将batchsize和timestep均设定为1
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        # 用所有字母初始化隐藏层，然后将所有字母按顺序存入结果
        _, state = net(get_input(), state)
        # 这里加入了字母，因此下一次get的字母就更新了
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        # 我们知道返回值为独热编码类型，大小为[time_step*batch_size,独热编码位数]，此时batchsize为1即所有结果一一相连，返回最有可能的预测结果
        # 这时一步一步计算，返回结果只有一个(不包括隐藏层的话)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
        # vocab,index是存储所有元素的列表
    return ''.join([vocab.idx_to_token[i] for i in outputs])

print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))


#                 网络   阈值
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    # 判断输入是否为网络
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # 对所有梯度的平方进行求和，然后开根号
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    # 这样裁剪之后norm的大小会变为（theta / norm）**2，就为梯度裁剪留开了空间
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# 对于一个batch来说，其中的数据集中训练（一次输出）
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    # X数据链和Y目标链，错位1
    # train_iter里为含有data，label的tuple，大小为（batchszie，timestep）
    for X, Y in train_iter:
        # 初始化或者随机迭代
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state，并且因为随机迭代中前后的样本没有关联，state需要重新初始化（其实不随机的也要初始化，但是可以继承前面的）
            # 因此这里学习了不同batchsize下的初始化权重
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 如果有网络且是否为初始化后的（state，）的隐藏层
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                # 保留数值但是不保留梯度，即继承上一个state的数据，梯度在这个epoch重新计算
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        # 转换为（timestep，batchsize）后展开，此时依次为每个step的所有结果依次
        # 转换是因为y_hat的=顺序是对于batch的每个内容逐个预测然后拼接，就导致了数据顺序变化为（timestep，batchsize）
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        # 获得预测的y和隐藏层
        y_hat, state = net(X, state)
        # 这里本质类似多分类问题，因此loss为crossentropy
        l = loss(y_hat, y.long()).mean()
        # 根据是否拥有需要梯度归零的优化器来决定怎么优化
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        # numel返回其中元素个数（即batchsize*timestep）
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())


# 我们应该明白其中训练的对象只和独热编码位数（分类可能）和隐藏层数量有关，batchsize是无关的变量，只影响隐藏层可以批量操作的数量
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=False)
