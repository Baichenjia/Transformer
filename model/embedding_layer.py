"""Implementation of embedding layer with shared weights."""

import model_utils
import tensorflow as tf  
tf.enable_eager_execution()

# 该层实现了embedding layer，同时embedding的权重与最后一层全连接的权重共享.
# embedding层和最后一层全连接层的权重矩阵规模相等，均为 (vocab_size * hidden_size)

class EmbeddingSharedWeights(tf.keras.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, hidden_size, method="gather", name=None):
        """Specify characteristic parameters of embedding layer.
        Args:
            vocab_size: Number of tokens in the embedding. (Typically ~32,000)
            hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
        """
        super(EmbeddingSharedWeights, self).__init__(name=name)
        assert name is not None
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.shared_weights = tf.get_variable(name+"-weights", 
                [self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(0., self.hidden_size ** -0.5))
    
    def call(self, x):
        """Get token embeddings of x.

        输入x是原始是句子: shape=[batch_size, length], int型

        输出embeddings: shape=[batch_size, length, embedding_size]
            padding: float32 tensor with shape [batch_size, length] indicating the locations of the padding tokens in x.
        """
        # mask中凡是 padding 的位置都为0, 其余位置为1
        # 将其余embedding结果相乘，表示所有padding的位置都不需要进行embedding
        mask = tf.to_float(tf.not_equal(x, 0))
        embeddings = tf.gather(self.shared_weights, x) # shape=(batch, length, embedding_size)
        embeddings *= tf.expand_dims(mask, -1)         # (batch, length, dim) * (batch, length, 1) = (batch, length, dim)

        # Scale embedding by the sqrt of the hidden size 
        # 具体原因不明
        embeddings *= self.hidden_size ** 0.5
        return embeddings


    def linear(self, x):
        """Computes logits by running x through a linear layer.
        liner操作与embedding操作共享权重，乘以的是embedding句子的转置
        x: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns: float32 tensor with shape [batch_size, length, vocab_size].
        """
        
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        x = tf.reshape(x, [-1, self.hidden_size])
        logits = tf.matmul(x, self.shared_weights, transpose_b=True)  # 转置

        return tf.reshape(logits, [batch_size, length, self.vocab_size])
