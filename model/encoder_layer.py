
import tensorflow as tf
tf.enable_eager_execution()

import attention_layer, ffn_layer, layer_norm
from post_process import PrePostProcessingWrapper

class EncoderStack(tf.keras.layers.Layer):
    """Transformer encoder stack.
    共计包含 N 个相同的模块，每个模块的处理流程为:
        1. self-attention 层
        2. 两个前向全连接层
    """
    def __init__(self, params, train, name=None):
        super(EncoderStack, self).__init__(name=name)
        # self.layers = []
        # for idx in range(params["num_hidden_layers"]):  # 参数为6
        
        # 0. -------------------------------
        # 初始化 selfAttention 层
        self_attention_layer_0 = attention_layer.SelfAttention(params["hidden_size"], params["num_heads"], params["attention_dropout"], train, n="Encoder self-attention 0", name="enc-selfatt-0")
        # 初始化 前向全连接 层
        feed_forward_network_0 = ffn_layer.FeedFowardNetwork(params["hidden_size"], params["filter_size"], params["relu_dropout"], train, params["allow_ffn_pad"], name="enc-ffn-0")
        
        # PrePostProcessingWrapper的目的是进行包装
        # 具体操作为: layer_norm -> 具体操作 -> dropout -> resdual-connect
        self.self_attention_wrapper_0 = PrePostProcessingWrapper(self_attention_layer_0, params, train, name="enc-selfattwrap-0")
        self.feed_forward_wrapper_0 = PrePostProcessingWrapper(feed_forward_network_0, params, train, name="enc-ffnwrap-0")

        # 1. -------------------------------
        self_attention_layer_1 = attention_layer.SelfAttention(params["hidden_size"], params["num_heads"], params["attention_dropout"], train, n="Encoder self-attention 1", name="enc-selfatt-1")
        feed_forward_network_1 = ffn_layer.FeedFowardNetwork(params["hidden_size"], params["filter_size"], params["relu_dropout"], train, params["allow_ffn_pad"], name="enc-ffn-1")
        self.self_attention_wrapper_1 = PrePostProcessingWrapper(self_attention_layer_1, params, train, name="enc-selfattwrap-1")
        self.feed_forward_wrapper_1 = PrePostProcessingWrapper(feed_forward_network_1, params, train, name="enc-ffnwrap-1")
        
        # 2. -------------------------------
        self_attention_layer_2 = attention_layer.SelfAttention(params["hidden_size"], params["num_heads"], params["attention_dropout"], train, n="Encoder self-attention 2", name="enc-selfatt-2")
        feed_forward_network_2 = ffn_layer.FeedFowardNetwork(params["hidden_size"], params["filter_size"], params["relu_dropout"], train, params["allow_ffn_pad"], name="enc-ffn-2")
        self.self_attention_wrapper_2 = PrePostProcessingWrapper(self_attention_layer_2, params, train, name="enc-selfattwrap-2")
        self.feed_forward_wrapper_2 = PrePostProcessingWrapper(feed_forward_network_2, params, train, name="enc-ffnwrap-2")
        
        # layer-norm 层，用于最终输出时使用
        self.output_normalization = layer_norm.LayerNormalization(params["hidden_size"], name="enc-norm")

    def call(self, encoder_inputs, attention_bias, inputs_padding):
        """Return the output of the encoder layer stacks.
        
        attention_bias: padding的位置标记为-1e9，其余位置标记为0.  shape=(batch,1,1,length)
        inputs_padding: 获取 padding 的位置，标记为1,其余标记为0.  shape=(batch,length)

        输出: shape [batch_size, input_length, hidden_size]
        """
        encoder_inputs = self.self_attention_wrapper_0(encoder_inputs, attention_bias)
        encoder_inputs = self.feed_forward_wrapper_0(encoder_inputs, inputs_padding)

        encoder_inputs = self.self_attention_wrapper_1(encoder_inputs, attention_bias)
        encoder_inputs = self.feed_forward_wrapper_1(encoder_inputs, inputs_padding)
        
        encoder_inputs = self.self_attention_wrapper_2(encoder_inputs, attention_bias)
        encoder_inputs = self.feed_forward_wrapper_2(encoder_inputs, inputs_padding)

        return self.output_normalization(encoder_inputs)


