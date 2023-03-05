
from torch import nn


# 输入数据，输出处理数据
# 在python定义类的时候可以预先留出一个接口不实现，而在后续继承的子类中实现，当我们想要提醒自己这个类的子类一定要实现这个接口时，可以调用NotImplementError
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

# 输入初始处理输入，获得初始状态。 输入输入以及状态进行更新
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

# 获得输入输出，更新状态后返回当前批次输出
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        # 编码输出
        enc_outputs = self.encoder(enc_X, *args)
        # 解码输出
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # 传入编码输出的码和隐藏状态然后输出
        return self.decoder(dec_X, dec_state)