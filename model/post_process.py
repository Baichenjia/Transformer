
import tensorflow as tf
tf.enable_eager_execution()
from layer_norm import LayerNormalization

class PrePostProcessingWrapper(tf.keras.layers.Layer):
    """Wrapper class that applies layer pre-processing and post-processing."""
    def __init__(self, layer, params, train, name=None):
        super(PrePostProcessingWrapper, self).__init__(name=name)
        assert name is not None
        
        self.layer = layer
        self.postprocess_dropout = params["layer_postprocess_dropout"]
        self.train = train

        # Create normalization layer
        self.layer_norm = LayerNormalization(params["hidden_size"], name=name+"-norm")

    def call(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
        return x + y

