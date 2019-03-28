# Defines the Transformer model, and its encoder and decoder stacks.

import tensorflow as tf
tf.enable_eager_execution()
import attention_layer, embedding_layer, ffn_layer, model_utils, layer_norm
from post_process import  PrePostProcessingWrapper
from encoder_layer import EncoderStack
from decoder_layer import DecoderStack
import sys
sys.path.append("..")

import matplotlib.pyplot as plt

_NEG_INF = -1e9

class Transformer(tf.keras.Model): 
    """Transformer model for sequence to sequence data.
    1. 初始化 encoder 和 decoder 所用的 bias (mask)
    2. 调用相应的 encoder 和 decoder 函数
    """
    def __init__(self, params, train):
        """Initialize layers to build Transformer model.
        params: 初始化在 model_prarms中的对象
        train: 是bool型变量，用于决定是否使用dropout
        """
        super(Transformer, self).__init__()
        self.train = train
        self.params = params
        # encoder embedding
        self.embedding_layer_encoder = embedding_layer.EmbeddingSharedWeights(
                params["vocab_size_input"], params["hidden_size"], name="enc-embed")
        # decoder embedding
        self.embedding_softmax_layer_decoder = embedding_layer.EmbeddingSharedWeights(
                params["vocab_size_output"], params["hidden_size"], name="dec-embed")
        #
        self.encoder_stack = EncoderStack(params, train, name="Enc-stack")
        self.decoder_stack = DecoderStack(params, train, name="Dec-stack")

    def call(self, inputs, targets):
        """Calculate target logits or inferred target sequences.
        
        Args:
            inputs: int tensor with shape [batch_size, input_length].
            targets: None or int tensor with shape [batch_size, target_length].

        Returns:
            If targets is defined, then return logits for each word in the target sequence. 
            float tensor with shape [batch_size, target_length, vocab_size]
            If target is none, then generate output sequence one token at a time.
                returns a dictionary {
                    output: [batch_size, decoded length]
                    score: [batch_size, float]}
        """
        # 此处没有什么实际的解释，总之是有好处的
        # initializer = tf.variance_scaling_initializer(
                # self.params["initializer_gain"], mode="fan_avg", distribution="uniform")
        # with tf.variable_scope("Transformer", initializer=initializer):
            # 所有的padding位置标记为 -1e9， 其余位置为0
        
        attention_bias = model_utils.get_padding_bias(inputs)
        # 经过 encoder 得到输入句子的编码
        encoder_outputs = self.encode(inputs, attention_bias)  # batch,length,hidden_size

        # 用于预测
        logits = self.decode(targets, encoder_outputs, attention_bias)
        return logits

    def encode(self, inputs, attention_bias):
        """Generate continuous representation for inputs.
        
        流程: Embedding -> 位置编码 -> dropout -> encoder_stack

        inputs: 原始的输入句子, shape=[batch_size, input_length].
        attention_bias: padding的位置标记为-1e9，其余位置标记为0. shape=[batch_size, 1, 1, input_length]
        返回encoder提取的特征: shape=[batch_size, input_length, hidden_size]
        """
        with tf.name_scope("encode"):
            # shape=(batch_size, length, embedding_dim)
            # embdding的过程中，padding的位置输出的是全0的向量
            embedded_inputs = self.embedding_layer_encoder(inputs)  # embedding
            length = tf.shape(embedded_inputs)[1]

            # 获取 padding 的位置，标记为1, 其余标记为0. shape=(batchsize, length)
            inputs_padding = model_utils.get_padding(inputs)

            with tf.name_scope("add_pos_encoding"):
                # 位置编码，shape=(length, hidden_size)
                pos_encoding = model_utils.get_position_encoding(length, self.params["hidden_size"])

                # 组合, shape=(length, hidden_size)
                encoder_inputs = embedded_inputs + pos_encoding

            if self.train:
                encoder_inputs = tf.nn.dropout(encoder_inputs, rate=self.params["layer_postprocess_dropout"])

            # 最后一步：调用 encoder_stack处理
            return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)

    def decode(self, targets, encoder_outputs, attention_bias):
        """Generate logits for each value in the target sequence.

        targets: 目标语言. shape=[batch_size, target_length].用于计算损失
        encoder_outputs: encoder的输出，在decoder中要对其进行attention操作.[batch_size, input_length, hidden_size]
        attention_bias: padding的位置标记为-1e9，其余位置标记为0. shape=[batch_size, 1, 1, input_length]

        返回 shape = [batch_size, target_length, vocab_size]. 最后一维与词表长度相等
        """
        with tf.name_scope("decode"):
            # embedding后 shape=(batch_size, length, embedding_dim)
            decoder_inputs = self.embedding_softmax_layer_decoder(targets)
            # print("decoder_inputs.shape =", decoder_inputs)
            
            # 在length中的第一维填充全0向量. 维度不变
            with tf.name_scope("shift_targets"):
                decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            # print("&&", decoder_inputs[0, 0:2, :10])

            # 加入位置编码
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += model_utils.get_position_encoding(length, self.params["hidden_size"])
            
            if self.train:
                decoder_inputs = tf.nn.dropout(decoder_inputs, rate=self.params["layer_postprocess_dropout"])

            # shape=(1, 1, length, length). 主对角元和下三角为0，其余元素为无穷小.
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(length)
            
            # decode. 此处要传入两个bias:
            # decoder_self_attention_bias 是一个三角矩阵，表示self-attention中的依赖关系
            # attention_bias 对encoder源语言中padding的位置标记为-1e9. 输入shape=(batch,length_decoder,dim)
            outputs = self.decoder_stack(decoder_inputs, encoder_outputs, 
                            decoder_self_attention_bias, attention_bias)

            # 该输出层的权重和embedding层共享. shape=(batch,length_decoder,vocab_size)
            logits = self.embedding_softmax_layer_decoder.linear(outputs)
            return logits

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = model_utils.get_position_encoding(max_decode_length + 1, self.params["hidden_size"])
        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(max_decode_length)   # 三角形矩阵 (1,1,length,length)
        
        def symbols_to_logits_fn(ids, i, cache):
            # Set decoder input to the last generated IDs
            if i == 0:
                decoder_input = tf.zeros([ids.shape[0], 1, self.params["hidden_size"]])
            else:
                decoder_input = ids[:, -1:]   # (batch, 1)
                decoder_input = self.embedding_softmax_layer_decoder(decoder_input)  # (batch, 1, 256)
            
            decoder_input += timing_signal[i:i + 1]
            
            # 在翻译中，这里的 bias 是全0向量，长度与当前翻译的长度i相等. 实际上没有任何作用，加入到logits之后，logits不发生变化
            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

            decoder_outputs = self.decoder_stack(
                  decoder_input, cache.get("encoder_outputs"), self_attention_bias,
                  cache.get("encoder_decoder_attention_bias"), cache)
            logits = self.embedding_softmax_layer_decoder.linear(decoder_outputs)
            return logits, cache
        return symbols_to_logits_fn


