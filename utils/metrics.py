
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)."""
    with tf.name_scope("pad_to_same_length"):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]

        max_length = tf.maximum(x_length, y_length)

        # 这里的padding是在length的维度向后padding. 
        x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
        return x, y


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
    """Calculate cross entropy loss while ignoring padding.

    Args:
        logits: Tensor of size [batch_size, length_logits, vocab_size]
        labels: Tensor of size [batch_size, length_labels]
        smoothing: Label smoothing constant, used to determine the on and off values
        vocab_size: int size of the vocabulary
    Returns:
        Returns the cross entropy loss and weight tensors: float32 tensors with
            shape [batch_size, max(length_logits, length_labels)]
    """
    with tf.name_scope("loss", values=[logits, labels]):
        logits, labels = _pad_tensors_to_same_length(logits, labels)

        # Calculate smoothing cross entropy
        with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
            confidence = 1.0 - smoothing
            low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
            # 直观理解: 如果labels中第0个词在词表中的位置为10，词表长度为100
            #   则变化后，one-hot在第0个位置有100维，其中第10维占据约confidence=0.9,其余概率被其他维平分，各占据low_confidence
            soft_targets = tf.one_hot(   # shape=(length, vocab_size)
                    tf.cast(labels, tf.int32),
                    depth=vocab_size,
                    on_value=confidence,
                    off_value=low_confidence)
            # 返回是一个list，与Logits的维度相等 (batchsize, length)
            xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=soft_targets)

            # Calculate the best (lowest) possible value of cross entropy, and subtract from the cross entropy loss.
            # 理解：当取到了正确类时的交叉熵损失. 由于label是soft的,因此不能只在正确label处计算损失
            # 当 smoothing=0.1, vocab_size=100时, normalizing_constant=0.78
            normalizing_constant = -(
                confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
                low_confidence * tf.log(low_confidence + 1e-20))
            # 如果已经分类正确，则 xentropy 减去 normalizer 之后等于0
            xentropy -= normalizing_constant

        # 对 padding 的部分进行处理, padding的部分不计算损失
        weights = tf.to_float(tf.not_equal(labels, 0))
        return xentropy * weights, weights
