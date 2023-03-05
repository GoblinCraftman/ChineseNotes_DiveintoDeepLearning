import collections
import math
import os

import torch
from torch import nn
from d2l import torch as d2l

def mid():
    print('-'*30)

# 该网络由于对于输入数据的输出
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        # 这一层将独热编码的字典大小也就是vocabsize传入，将会输出一个大小相同的enbedsize维数的输出字典
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # dropout–如果非零，则在除最后一层外的每个GRU层的输出上引入一个dropout层，其dropout概率等于dropout。默认值：0
        # 特别注意的模组的GRU只有循环层内容，返回的是处理过的中间状态（即最后一个维度是numhid，没有进行linear）
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步（遍历时优先按照时间步顺序输出每个batchsize的数据）
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0（nn.GRU中state如果没有传入初始化方法默认是torch.zero()
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        # 此时的output和state都是沿着时间步逐步处理的数据，output是concat的，state是自更新的
        return output, state

encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)

# 避免此时即开始积累梯度
encoder.eval()
# 试验大小
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
mid()
print(output.shape)
print(state.shape)

# 对于输入state的生成网络的训练
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 需要注意的是decoder同样也需要编码，这里的编码对应的是目标语言的vocab
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这里假设encoder和decoder的隐藏层大小相同，（？这样写是因为会将上一层的输出的中间状态和隐藏层同时输入）
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        # 从中间状态提取数据的linear
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # 这里就是返回编码器部分输出的第二个值，编码器输出为[中间状态，隐藏层]，即将输入语句最后的提取隐藏层（2层）作为初始化输入
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2) # [num_steps,batch_size,embed_size]
        # 广播context，使其具有与X相同的num_steps。[num_steps，1，1]。这一步操作将提取的隐藏层广播到所有时间步上
        context = state[-1].repeat(X.shape[0], 1, 1)
        # 对于一个[batch_size,num_steps,embed_size]的X来说，这样会自动广播到所有的X上
        # 这一步的意义在于将X的每个numstep的数据传入
        X_and_context = torch.cat((X, context), 2)
        # 这里会自动逐步操作，因为GRU接收到的输入为正确输入加上隐藏层，此时即为使用正确输入进行训练
        output, state = self.rnn(X_and_context, state)
        # 这里使用linear会保留除开最后一个维度的形状
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

#                           词表大小        编码位数        隐藏层大小       隐藏层层数
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
decoder.eval()
# 这里X只是拿来看大小
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
mid()
# [batchsize, timestep, vocabsize] [num_layer, batchsize, numhid]
print(output.shape, state.shape)
print(len(state), state[0].shape)

# X:[batchsize,idxs]
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    # 获得序列x中的长度，这个值很关键
    maxlen = X.size(1)
    # 生成需要替代的位置的索引，后为判断条件（这里面的值均为bool）
    # 这里的操作非常巧妙，意为前面的数据依照后方向高维广播，后方的数据依照前方向低维广播，输出大小会是[validlen(1)，maxlen]
    # 同时注意这个操作的意义是对高维数据进行适配（valid_len的长度是和X一样的）
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # ~:按位取反
    # 同时bool可以决定是否赋值
    X[~mask] = value
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
mid()
#                      对应data/label的maxlen
print(sequence_mask(X, torch.tensor([1, 2])))

# 此函数也可用于屏蔽最后几个维度的数据
# 这是因为数据取1为第二维度，其数据会自动向低维度广播（如果维度大于2）
X = torch.ones(2, 3, 4)
print(sequence_mask(X, torch.tensor([1, 2]), value=-1))

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        # 自动对每个进行截止处理
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        # 计算[batch_size,vocab_size,num_steps]和[batch_size,num_steps]的损失
        # 根据定义，crossentropy计算每种可能的预测可能性和（单一）label的损失，要求将维度放在最后一维
        # 这里给的是每种vocab的预测值的loss，numstep是其可能分布，因为我们的输出是vocab
        # 其他要求为(N, C, d_1, d_2, ..., d_K)(N,C, d_1, d_2, ..., d_K），d为批量，C=number of classes， N=batch size
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

loss = MaskedSoftmaxCELoss()
# loss的大小为[batchsize,numstep]
# 最大长度数量对应维度
print(loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0])))

#
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    # 手动初始化权重，只对GRU和Linear操作
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            # 这是一个nn.Moudle类，对其层param的所有weight进行拆解，获得名称，param是网络层权重具体名称
            # m._flat_weights_names来获得具体的weights的name
            for param in m._flat_weights_names:
                # param的权重会有weight和bias两个参数，此时param参数是权重的str字符串，下面判断其中是否有weight
                if "weight" in param:
                    # m._parameters和m.parameters（）效果大致相同，其为存储为
                    nn.init.xavier_uniform_(m._parameters[param])

    # 应用权重初始化和cuda化
    # apply应用会对net里面的所有对象，即所有层进行初始化操作
    net.apply(xavier_init_weights)
    net.to(device)
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    loss.cuda()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            # 传入参数cuda化，(batch_size,num_steps)，(batch_size）
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 添加开始符号（）                           batchsize
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1) # 这里的返回结果大小为（batchsize，1）
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学（广播）,相同纬度上拼接，其余维度自动广播（这里少取了一位用来添加开始符号）
            # 获得输出的结果，隐藏层不要
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            # 梯度裁剪，意译为梯度总和大于1即裁剪
            d2l.grad_clipping(net, 1)
            # 一共训练了几个token
            num_tokens = Y_valid_len.sum()
            # 反向传播
            optimizer.step()

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
net = d2l.EncoderDecoder(encoder, decoder)
# output的形状:(batch_size,num_steps,vocab_size)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# 此训练没有验证集，此时即为进行验证
#                   网络   翻译目标语句    原语言词表  目标语言词表   预测长度     类型     注意力
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    # 对传入token进行预处理，分为字符串，标点
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    # 获得有效长度（由源语言指向目标语言）
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    # 根据预测长度进行裁剪
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    # 传入获得编码输出
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    # 获得源语言隐藏层。由源语言长度获得目标语言长度，这样做是因为普遍长度存在大小关系（实际上这里初始化只和第一个参数有关）
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴（向上获得大小为1的一个维度），给所有的传入数据添加一个开始符号
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    # 创建获得结果和注意力权重的列表
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # 用开始符号变相初始化解码（输出）网络，获得初始结果和隐藏层
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        # 输出大小为torch.Size([1, 1, 201])(batchsize,num_step,tag_vocab_size)
        dec_X = Y.argmax(dim=2)
        # torch.Size([1, 1])(batchsize,num_step)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        # 保存输出结果
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
mid()
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

