
import sys
sys.path.append("model")
sys.path.append("utils")
sys.path.append("data")
import os
import numpy as np
import model_params, transformer, metrics, load_data, util, model_utils
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

# params
params = model_params.BAI_PARAMS

# init data and dict
print("Load data...")
dataset, inp_lang, targ_lang, input_val, target_val = load_data.prepare_tfdata("data/eng-spa.txt")
print(input_val.dtype)

# init model
test_inputs, test_targets = input_val[:params["default_batch_size"]], target_val[:params["default_batch_size"]]
model = transformer.Transformer(params, train=False)
test_logits = model(test_inputs, test_targets)
print("TEST:", test_logits.shape)
print("--------------")

# load weights
model.load_weights("weights/model_weight49.h5")
print("load weights.")

def predict(encoder_outputs, encoder_decoder_attention_bias):
    """Return predicted sequence."""
    batch_size = tf.shape(encoder_outputs)[0]
    input_length = tf.shape(encoder_outputs)[1]
    max_decode_length = input_length + params["extra_decode_length"]

    # Function to translate one step
    symbols_to_logits_fn = model._get_symbols_to_logits_fn(max_decode_length)

    # Create cache storing decoder attention values for each layer.
    cache = {"layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, params["hidden_size"]]),
                "v": tf.zeros([batch_size, 0, params["hidden_size"]]),
        } for layer in range(params["num_hidden_layers"])}

    # Add encoder output and attention bias to the cache.
    cache["encoder_outputs"] = encoder_outputs
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    # 
    IDS = tf.zeros([batch_size, 1], dtype=tf.int32)
    for i in range(max_decode_length):
        # print("Translation step:", i)
        # logits.shape = (batch_size, 1, vocab_size)
        logits, cache = symbols_to_logits_fn(IDS, i, cache)
        # argmax
        ids_new = tf.argmax(logits, axis=-1, output_type=tf.int32)
        # concat
        IDS = tf.concat([IDS, ids_new], axis=1)

        # test
        # print("---------Cache-----------")
        # print("logits:\n", logits)
        # for name, kv in cache.items():
        #     if name.startswith("layer"):
        #         print("name= ", name)
        #         print("key= ", kv["k"])
        #         print("value= ", kv["v"])
        #         print("-----")
        # # print("IDS:", IDS)
        # print("------------------------")
        # if i > 6:
        #     break

    return IDS[:, 1:].numpy()


def translate():
    source = ["Es un gran honor conocerte aqui.", 
              "Me gustaría hablar contigo sobre lo que ocurrió ayer en la escuela.",
              "Tom tiene una hermana que puede hablar francés.",
              "Soy un estudiante de la Universidad."]

    source = [load_data.preprocess_sentence(s) for s in source]
    input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')] for sp in source]
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                        maxlen=params["max_length_input"], padding='post')    

    # 经过 encoder
    attention_bias = model_utils.get_padding_bias(input_tensor)   # batch, 1, 1, length
    encoder_outputs = model.encode(input_tensor, attention_bias)  # batch,length,hidden_size
    print("---------Decoder-----------")

    # 进入decode
    IDS = predict(encoder_outputs, attention_bias)
        
    for i in range(len(source)):
        word = " ".join([targ_lang.idx2word[w] for w in IDS[i]])
        print("----------")
        print(source[i])
        print(word)
        print("----------\n")

translate()




