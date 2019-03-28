"""Implementation of fully connected network."""

import tensorflow as tf
tf.enable_eager_execution()

class FeedFowardNetwork(tf.keras.layers.Layer):
    """Fully connected feedforward network.
    处理的难点在于，一个批量内虽然已经padding成为一样的长度，但在计算损失时，只有不是
        padding的位置才计算损失。因此需要将padding的位置提取出来，其余位置置0.
        而批量中句子的原始长度不一致，因此padding的长度也不一样，因此需要一定的处理。
    """

    def __init__(self, hidden_size, filter_size, relu_dropout, train, allow_pad, name=None):
        super(FeedFowardNetwork, self).__init__(name=name)
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.train = train
        self.allow_pad = allow_pad

        self.filter_dense_layer = tf.layers.Dense(
                filter_size, use_bias=True, activation=tf.nn.relu, name=name+"-filter-layer")
        self.output_dense_layer = tf.layers.Dense(
                hidden_size, use_bias=True, name=name+"-output-layer")

    def call(self, x, padding=None):
        """Return outputs of the feedforward network.
        padding: 获取 padding 的位置，标记为1, 其余标记为0.  shape=(batch,length)

        x.shape      = [batch_size, length, hidden_size]
        return.shape = [batch_size, length, hidden_size]
        """
        padding = None if not self.allow_pad else padding

        # Retrieve dynamically known shapes
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        if padding is not None:
            with tf.name_scope("remove_padding"):
                pad_mask = tf.reshape(padding, [-1])   # shape=(batch*length, )
                # 序列中不是padding位置为0，因此可以提取出来. nopad_ids的长度 < batch*length
                nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))  

                # 提取没有padding的元素
                # Reshape x to [batch_size*length, hidden_size] to remove padding
                x = tf.reshape(x, [-1, self.hidden_size])
                x = tf.gather_nd(x, indices=nonpad_ids)

                # Reshape x from 2 dimensions to 3 dimensions.
                x.set_shape([None, self.hidden_size])
                x = tf.expand_dims(x, axis=0)

        # x.shape = (1, None, hidden_size)
        output = self.filter_dense_layer(x)  # output.shape=(1,None,filter_size)
        if self.train:
            output = tf.nn.dropout(output, 1.0 - self.relu_dropout)
        output = self.output_dense_layer(output)

        # ！ 恢复原来的大小，在 nopad_ids 的位置填充 output,其余位置均为0
        # scatter_nd函数的作用是，先全0初始化一个shape大小的tensor,随后根据indices将updates内容填充进去
        if padding is not None:
            with tf.name_scope("re_add_padding"):
                output = tf.squeeze(output, axis=0)
                output = tf.scatter_nd(
                        indices=nonpad_ids,
                        updates=output,
                        shape=[batch_size * length, self.hidden_size]
                )
                output = tf.reshape(output, [batch_size, length, self.hidden_size])
        return output
