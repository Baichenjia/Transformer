
import tensorflow as tf
tf.enable_eager_execution()
import attention_layer, layer_norm, ffn_layer
from post_process import PrePostProcessingWrapper

class DecoderStack(tf.keras.layers.Layer):
    """Transformer decoder stack.
    包含 N 个完全相同的模块，每个模块包括:
        1. Self-attention layer
        2. Multi-head attention，其中query是上一步的输出, key和value是encoder的输出
        3. 全连接层
    """
    def __init__(self, params, train, name=None):
        super(DecoderStack, self).__init__(name=name)

        # 0 -----------------------
        self_attention_layer_0 = attention_layer.SelfAttention(params["hidden_size"], params["num_heads"], params["attention_dropout"], train, n="Decoder self-attention 0", name="dec-selfatt-0")
        enc_dec_attention_layer_0 = attention_layer.Attention(params["hidden_size"], params["num_heads"], params["attention_dropout"], train, n="Decoder-encoder attention 0", name="dec-enc-0")
        feed_forward_network_0 = ffn_layer.FeedFowardNetwork(params["hidden_size"], params["filter_size"], params["relu_dropout"], train, params["allow_ffn_pad"], name="dec-ffn-0")
        self.self_attention_wrapper_0 = PrePostProcessingWrapper(self_attention_layer_0, params, train, name="dec-selfattwrap-0")
        self.enc_dec_attention_wrapper_0 = PrePostProcessingWrapper(enc_dec_attention_layer_0, params, train, name="dec-encwrap-0")
        self.feed_forward_wrapper_0 = PrePostProcessingWrapper(feed_forward_network_0, params, train, name="dec-ffnwrap-0")

        # 1 -----------------------
        self_attention_layer_1 = attention_layer.SelfAttention(params["hidden_size"], params["num_heads"], params["attention_dropout"], train, n="Decoder self-attention 1", name="dec-selfatt-1")
        enc_dec_attention_layer_1 = attention_layer.Attention(params["hidden_size"], params["num_heads"], params["attention_dropout"], train, n="Decoder-encoder attention 1", name="dec-enc-1")
        feed_forward_network_1 = ffn_layer.FeedFowardNetwork(params["hidden_size"], params["filter_size"], params["relu_dropout"], train, params["allow_ffn_pad"], name="dec-ffn-1")
        self.self_attention_wrapper_1 = PrePostProcessingWrapper(self_attention_layer_1, params, train, name="dec-selfattwrap-1")
        self.enc_dec_attention_wrapper_1 = PrePostProcessingWrapper(enc_dec_attention_layer_1, params, train, name="dec-encwrap-1")
        self.feed_forward_wrapper_1 = PrePostProcessingWrapper(feed_forward_network_1, params, train, name="dec-ffnwrap-1")

        # 2 -----------------------
        self_attention_layer_2 = attention_layer.SelfAttention(params["hidden_size"], params["num_heads"], params["attention_dropout"], train, n="Decoder self-attention 2", name="dec-selfatt-2")
        enc_dec_attention_layer_2 = attention_layer.Attention(params["hidden_size"], params["num_heads"], params["attention_dropout"], train, n="Decoder-encoder attention 2", name="dec-enc-2")
        feed_forward_network_2 = ffn_layer.FeedFowardNetwork(params["hidden_size"], params["filter_size"], params["relu_dropout"], train, params["allow_ffn_pad"], name="dec-ffn-2")
        self.self_attention_wrapper_2 = PrePostProcessingWrapper(self_attention_layer_2, params, train, name="dec-selfattwrap-2")
        self.enc_dec_attention_wrapper_2 = PrePostProcessingWrapper(enc_dec_attention_layer_2, params, train, name="dec-encwrap-2")
        self.feed_forward_wrapper_2 = PrePostProcessingWrapper(feed_forward_network_2, params, train, name="dec-ffnwrap-2")

        # 
        self.output_normalization = layer_norm.LayerNormalization(params["hidden_size"], name="dec-norm")

    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
                     attention_bias, cache=None):
        """Return the output of the decoder layer stacks.

        decoder_inputs: shape=(batch_size, length, embedding_dim) 为句子向右移动一位，第一位填0的结果
        encoder_outputs: encoder的输出，在decoder中要对其进行attention操作.[batch_size, input_length, hidden_size]
        decoder_self_attention_bias: shape=(1, 1, length, length). 主对角元和下三角为0，其余元素为无穷小.
        attention_bias: padding的位置标记为-1e9，其余位置标记为0. shape=[batch_size, 1, 1, input_length]
        
        cache: (Used for fast decoding) A nested dictionary storing previous
                decoder self-attention values. The items are:
            {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                       "v": tensor with shape [batch_size, i, value_channels]},...}
        Returns: shape=[batch_size, target_length, hidden_size]
        """

        layer_name = "layer_0"
        layer_cache = cache[layer_name] if cache is not None else None
        decoder_inputs = self.self_attention_wrapper_0(decoder_inputs, decoder_self_attention_bias, cache=layer_cache)
        decoder_inputs = self.enc_dec_attention_wrapper_0(decoder_inputs, encoder_outputs, attention_bias)
        decoder_inputs = self.feed_forward_wrapper_0(decoder_inputs)

        layer_name = "layer_1"
        layer_cache = cache[layer_name] if cache is not None else None
        decoder_inputs = self.self_attention_wrapper_1(decoder_inputs, decoder_self_attention_bias, cache=layer_cache)
        decoder_inputs = self.enc_dec_attention_wrapper_1(decoder_inputs, encoder_outputs, attention_bias)
        decoder_inputs = self.feed_forward_wrapper_1(decoder_inputs)

        layer_name = "layer_2"
        layer_cache = cache[layer_name] if cache is not None else None
        decoder_inputs = self.self_attention_wrapper_2(decoder_inputs, decoder_self_attention_bias, cache=layer_cache)
        decoder_inputs = self.enc_dec_attention_wrapper_2(decoder_inputs, encoder_outputs, attention_bias)
        decoder_inputs = self.feed_forward_wrapper_2(decoder_inputs)

        # norm, shape=(batch, length_dicoder, dim)
        return self.output_normalization(decoder_inputs)


