import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt

_NEG_INF = -1e9

# 位置编码
def get_position_encoding(length, hidden_size, 
                          min_timescale=1.0, max_timescale=1.0e4):
    """ Return positional encoding.
        按照论文中公式进行展开. 首先将sin中的内容计算Log，展开，随后计算Log中的每一项
            随后进行exp还原. 最后再计算sin,cos合并. 
        设length=100, hidden_size=20, 则最终输出的是 shape=[100,20] 的位置编码.
    """
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            tf.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal   # shape=(length, hidden_size)

# 可视化位置编码函数. 越靠后面的维度位置的跨度越大.
# signal = get_position_encoding(length=100, hidden_size=20)
# print(signal.shape)
# plt.figure(figsize=(15, 5))
# plt.plot(np.arange(100), signal[:, 0:4].numpy())
# plt.legend(["dim %d"%p for p in [0,1,2,3]])
# plt.show()


# 返回decoder所用的attention权重的mask
def get_decoder_self_attention_bias(length):
    """ 
    返回decoder使用的mask矩阵，shape=(1, 1, length, length).
        主对角元和下三角为0，其余元素为无穷小. 
        该句子在decoder中经过softmax后上三角的权重会被置为接近于0. 
        防止每个时间步的单词atten到后面的词.
    """
    with tf.name_scope("decoder_self_attention_bias"):
        # 返回shape=(length,length)的矩阵，主对角元及下三角元素全为1,其余为0
        valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        # 返回矩阵，主对角及下三角全部为0，其余为 -1e9
        decoder_bias = _NEG_INF * (1.0 - valid_locs)
        # decoder_bias在经过softmax之后，会将上三角元素的权重全部置为0
    return decoder_bias   # shape=(1, 1, length, length)

# decoder_bias = get_decoder_self_attention_bias(length=10)
# print(decoder_bias.shape)   # (1,1,10,10)
# plt.figure(figsize=(5,5))
# plt.imshow(decoder_bias[0,0])
# plt.show()


def get_padding(x, padding_value=0):
    # 检查 x 中的每个元素，如果与padding_value值相等，则为1. , 否则为0.
    # 返回值的 shape = x.shape
    with tf.name_scope("padding"):
        return tf.to_float(tf.equal(x, padding_value))


def get_padding_bias(x):
    """Calculate bias tensor from padding values in tensor.
    将 x 中的元素与 0 进行比较，如果相等，此处为-1e9，如果不等，此处为0.
    此处的目的应该是，padding的元素处不计算损失

    Bias tensor that is added to the pre-softmax multi-headed attention logits,
    which has shape [batch_size, num_heads, length, length]. The tensor is zero at
    non-padding locations, and -1e9 (negative infinity) at padding locations.

    Args:     x: int tensor with shape [batch_size, length]
    Returns:  Attention bias tensor of shape [batch_size, 1, 1, length].
    """
    with tf.name_scope("attention_bias"):
        padding = get_padding(x)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
    # x.shape=(batch_size,length), attention_bias.shape=(batch_size,1,1,length)
    return attention_bias 


# decoder_bias = get_decoder_self_attention_bias(length=10)[0, 0, :, :]
# attention_bias = get_padding_bias(decoder_bias)
# print(attention_bias.shape)     # [10, 1, 1, 10]
# print(attention_bias.numpy()[:, 0, 0, :])
# plt.figure(figsize=(5,5))
# plt.imshow(attention_bias.numpy()[:, 0, 0, :])
# plt.show()