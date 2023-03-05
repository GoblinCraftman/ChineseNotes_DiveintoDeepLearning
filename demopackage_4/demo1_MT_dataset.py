
# 英语到法语
import os
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

def mid():
    print('-'*30)

def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
# 此数据集中，数据格式为：英文 标点 （空格\t） 法语 空格 标点 （空格？） 换行符（空格）
mid()
print(raw_text[:75])
print(len(raw_text))

def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    # 这里返回一个bool判断值，判断标点前有无空格，没有则添加（后续训练会将标点作为一个token来训练）
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    # 这些都是utf-8的代码格式，将半角和全角的空格全部替换为空格
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    # 这一步的写法很关键，含有换行符的str可以进行迭代，其迭代仍然为每个字符(包括换行符)
    # 对其进行判断是否满足条件以确定返回' ' + char或者char，enumerate的i第一个为0
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    # 直接全连接
    return ''.join(out)

text = preprocess_nmt(raw_text)
mid()
print(text[:80])

def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    # 以换行符分割每个对应的训练token
    for i, line in enumerate(text.split('\n')):
        # 设置训练样本上限
        if num_examples and i > num_examples:
            break
        # 此时数据格式为 英文 空格 标点 \t 法语 空格 标点 \n，先分割为data和label部分
        parts = line.split('\t')
        if len(parts) == 2:
        # 此时数据集中的格式大致为 ’go .	va !‘，以空格分割会产生两个个元素，即向列表中加入列表
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
mid()
print(source[:6])
print(target[:6])

print(len(source))
def get_distribution(x):
    data = [max(len(j) for j in i) for i in x]
    data_list = []
    for i in range(20):
        data_mid = len([a for a in data if a == (i+1)])
        exec(f'data_{i} = {data_mid}')
        exec(f'data_list.append(data_{i})')
    return data_list

#num_data = get_distribution(source)
#num_label = get_distribution(target)

def barshow(*args):
    plt.title('The long of tokens in data and label')
    plt.bar([i - 0.5 for i in range(1, 21)], args[0], tick_label = args[0], label='data', width=0.5)
    plt.bar([i for i in range(1, 21)], args[1], tick_label = args[1], label='label', width=0.5)
    plt.xticks([i-0.5 for i in range(1,21)],[i for i in range(1, 21)])
    plt.xlabel('The long of tokens')
    plt.ylabel('The times of each long')
    plt.legend()
    plt.show()

#barshow(num_data,num_label)

# 详细代码见demopackage_3\demo2_Text_preprocessing.py，这里返回了一个vocab类，其[]被重载，keys为单词。values为出现次数，min_freq为抑制阈值，低于其的单词将会被舍弃
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
# 我们还指定了额外的特定词元， 例如在小批量时用于将序列填充到相同长度的填充词元（“<pad>”）， 以及序列的开始词元（“<bos>”）和结束词元（“<eos>”）
mid()
print(len(src_vocab))

# 训练的对象需要具有一定的长度（肯定可以训练长度），因此做裁剪
#               判断序列  时间步数      填充字符
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

mid()
print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))
print([src_vocab.idx_to_token[i] for i in truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])])

def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    # tokens to idx
    lines = [vocab[l] for l in lines]
    # 添加停止符
    lines = [l + [vocab['<eos>']] for l in lines]
    # 转换为大小相同的批量（截断或者填充）
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    # 将判断是否为pad的bool值数值在第二维度相加，第二维度消失，返回第一维度大小的tensor
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    #   转换的批向量  词元数
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    # 获得源文本
    text = preprocess_nmt(read_data_nmt())
    # 获得处理后分开的data和label
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 分别转换data和label为向量
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 合并并且转换为词向量迭代器
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    # 类似的dataset参考demopackage_3/demo1_COST.py的tensor数据集构建
    # 其中直接传入数据即可，本质即返回一个数据流，其具体意义如data和label由人为具体判断
    data_iter = d2l.load_array(data_arrays, batch_size)
    # 返回    迭代器      输入词表    输出词表
    return data_iter, src_vocab, tgt_vocab

# 获得数据
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break

