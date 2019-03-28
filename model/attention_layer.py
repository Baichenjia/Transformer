"""Implementation of multiheaded attention and self-attention layers."""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()

class Attention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""
    def __init__(self, hidden_size, num_heads, attention_dropout, train, n=None, name=None):
        # train 是 bool变量，决定是否使用dropout
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of heads.")

        super(Attention, self).__init__(name=name)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train
        self.n = n

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name=name+"-q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name=name+"-k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name=name+"-v")

        self.output_dense_layer = tf.layers.Dense(hidden_size, 
                    use_bias=False, name=name+"-output_transform")

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.

        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
            x: A tensor with shape [batch_size, length, hidden_size]

        Returns:
            A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.
        将输入中的第1维和第3维合并，作为最后一维
        Args:
            x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

        Returns:
            A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, bias, cache=None):
        """Apply attention mechanism to x and y.
        
        bias: padding的位置标记为-1e9，其余位置标记为0.  shape=(batch,1,1,length)

        Args:
            x: a tensor with shape [batch_size, length_x, hidden_size]
            y: a tensor with shape [batch_size, length_y, hidden_size]
            bias: attention bias that will be added to the result of the dot product.
            cache: (Used during prediction) dictionary with tensors containing results
                of previous attentions. The dictionary must have the items:
                        {"k": tensor with shape [batch_size, i, key_channels],
                         "v": tensor with shape [batch_size, i, value_channels]}
                where i is the current decoded length.

        Returns:
            Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).

        # key和value总来自于同一个地方，在decoder中，key和value来自于encoder, query来自于decocer
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y) 
        v = self.v_dense_layer(y)   
        # 在translate时，每次解码一个词. x.shape=(batch,1,hidden_size)
        #    q.shape=k.shape=v.shape=(batch,1,hidden_size)

        # cache在translate中才会用到，在翻译第1个词之后, cache["k"].shape=(batch,1,hiddensize)
        #   在翻译第2个词之后，cache["k"].shape=(batch,2,hiddensize)...以此类推
        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v   

        # Split q, k, v into heads.
        # 在训练中, length_x,length_y都是提前定好的.
        # 在翻译中(decoder-self-attention中), q 的维度不变，每次都是(batch, heads, 1, depth)
        #    k,v 的维度每翻译一步都要增加, (batch, heads, {1,2,3,4....}, depth)
        q = self.split_heads(q)    # (batch, num_heads, length_x, depth)
        k = self.split_heads(k)    # (batch, num_heads, length_y, depth)
        v = self.split_heads(v)    # (batch, num_heads, length_y, depth)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5   # 规约，论文中的描述为 Q*T(K)/sqrt(depth)

        # Q*T(K)  Q和K的前两维是batch,num_heads，均不变. 后两维相乘:(length_x,depth)*(depth,length_y)=(length_x,length_y)
        logits = tf.matmul(q, k, transpose_b=True)  # shape=(batch,num_heads,length_x,length_y)
        # bias.shape=(batch,1,1,length_y),加上 bias 之后，padding位置的数值变为负无穷,随后经softmax后会转为0
        # 在encoder-decoder attention中,bias最后一维的长度与encoder长度相等.表示对于输入的句子中是padding的区域不进行attention
        logits += bias 

        # softmax后维度为(batch, num_heads, length_x, length_y). 最后一维加和为1
        weights = tf.nn.softmax(logits, name="attention_weights")

        # plot
        def plot_attention():
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1, 1, 1)
            ax.matshow(weights[0, :, 0, :], cmap='viridis')
            # plt.imshow(weights[0, 0, :, :])            
            # ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
            # ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
            plt.title(self.name+",     shape="+str(weights.shape))
            plt.show()
        # plot_attention()

        if self.train:
            weights = tf.nn.dropout(weights, rate=self.attention_dropout)
        attention_output = tf.matmul(weights, v)  # shape=v.shape=(batch,num_heads,length_x,depth)

        # 将num_heads和depth合并. Recombine heads --> [batch_size, length, hidden_size] 
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        # 输出 shape=(batch, length, hidden_size)
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, cache=None):
        return super(SelfAttention, self).call(x, x, bias, cache)
