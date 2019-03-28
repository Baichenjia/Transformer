import sys
sys.path.append("model")
sys.path.append("utils")
sys.path.append("data")
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
tf.enable_eager_execution()
import model_params, transformer, metrics, load_data, util
import matplotlib.pyplot as plt

# weight
PATH = "weights/model_weight49.h5"
assert os.path.exists(PATH)

# params
params = model_params.BAI_PARAMS

# compute loss
def compute_loss(m, inputs, targets):
    # compute logits
    logits = m(inputs, targets)   # (batch, length, vocab_size_output)
    assert logits.shape[1:] == (params['max_length_output'], params['vocab_size_output'])
    # xentropy中不计算padding位置的损失.  xentropy.shape=(batchsize, length)
    xentropy, weights = metrics.padded_cross_entropy_loss(
            logits, targets, params["label_smoothing"], params["vocab_size_output"])
    # assert xentropy.shape == weights.shape == (params['default_batch_size'], params['max_length_output'])

    # 取平均，对padding的位置不纳入计算
    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
    return loss

# init data and dict
print("Load data...")
dataset, inp_lang, targ_lang, input_val, target_val = load_data.prepare_tfdata("data/eng-spa.txt")

# init model
model = transformer.Transformer(params, train=False)

# Test
test_inputs, test_targets = input_val[:params["default_batch_size"]], target_val[:params["default_batch_size"]]
test_logits = model(test_inputs, test_targets)
print("TEST:", test_logits.shape)

# Load weights
model.load_weights(PATH)
print("load weights.")

# EVAL
for i in range(6):
    print("----------\n Eval: ", i)
    eval_loss = compute_loss(m=model, inputs=input_val[1000*i: 1000*(i+1)], targets=target_val[1000*i: 1000*(i+1)])
    print("EVAL LOSS:", eval_loss.numpy())

