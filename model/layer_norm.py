import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


class LayerNormalization(tf.keras.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size, name=None):
        super(LayerNormalization, self).__init__(name=name)
        self.hidden_size = hidden_size

        self.scale = tf.get_variable(name+"-layer_norm_scale", [self.hidden_size], 
                initializer=tf.ones_initializer())  # 全1初始化
        self.bias = tf.get_variable(name+"-layer_norm_bias", [self.hidden_size], 
                initializer=tf.zeros_initializer()) # 全0初始化

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)   # (batchsize,length,1)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)  # (batchsize,length,1)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)   # (batchsize,length,dims)
        return norm_x * self.scale + self.bias


# if __name__ == '__main__':
#     batch_size, seq_len, dims = 5, 10, 3
#     x = tf.convert_to_tensor(np.random.random((batch_size, seq_len, dims)), dtype=tf.float32)
#     layer_norm = LayerNormalization(hidden_size=3)
#     y = layer_norm(x)
#     print(y.shape)

