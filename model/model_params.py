
from collections import defaultdict

BAI_PARAMS = defaultdict(
    lambda: None,               # Set default value to None.

    # Model params
    initializer_gain=1.0,       # Used in trainable variable initialization.
    max_length_input = 16,      # length
    max_length_output = 11, 
    vocab_size_input = 9394,    # vocab
    vocab_size_output = 4918,   # hidden size
    hidden_size = 256,          
    num_hidden_layers = 3,      # hidden layer
    num_heads = 4,
    filter_size = 512,

    # Dropout values (only used when training)
    layer_postprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1,

    # Training params
    learning_rate = 1e-4,
    label_smoothing=0.1,
    learning_rate_decay_rate=1.0,
    learning_rate_warmup_steps=16000,
    default_batch_size = 256,

    # Optimizer params
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.997,
    optimizer_adam_epsilon=1e-09,

    # Default prediction params
    extra_decode_length=50,
    beam_size=4,
    alpha=0.6,  # used to calculate length normalization in beam search
)
