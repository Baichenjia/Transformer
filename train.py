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
model = transformer.Transformer(params, train=True)

# Test
test_inputs, test_targets = input_val[:params["default_batch_size"]], target_val[:params["default_batch_size"]]
test_logits = model(test_inputs, test_targets)
print("TEST:", test_logits.shape)

# Load weights
# model.load_weights("weights/model_weight.h5")
# print("load weights.")

def get_learning_rate(step, init=params["learning_rate"]):
    return tf.convert_to_tensor(init)
    # return tf.convert_to_tensor(init * pow(0.7, (step / 1000.)), dtype=tf.float32)

# Optimizer
learning_rate = tfe.Variable(params["learning_rate"], trainable=False, name="LR")
optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate,
        beta1=params["optimizer_adam_beta1"], beta2=params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

# Train
global_step = tf.train.get_or_create_global_step()
EPOCHS = 50
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    for (batch, (inputs, targets)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            loss = compute_loss(m=model, inputs=inputs, targets=targets)

        gradient = tape.gradient(loss, model.trainable_variables)
        gradient_norm = tf.global_norm(gradient)
        if batch % 10 == 0:
            print("Global step:", global_step.numpy(), ", learning_rate:", learning_rate.numpy())
            print("batch:", batch, ", loss:", loss.numpy(), ", gradient_norm:", gradient_norm.numpy())
            print("")
        # clip and apply
        gradient, _ = tf.clip_by_global_norm(gradient, 1.)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables), global_step=global_step)

        # change lr
        new_lr = get_learning_rate(global_step.numpy())
        learning_rate.assign(new_lr)

    # eval
    for i in range(6):
        eval_loss = compute_loss(m=model, inputs=input_val[1000*i: 1000*(i+1)], targets=target_val[1000*i: 1000*(i+1)])
        print("EVAL LOSS:", eval_loss.numpy())
    
    # save    
    model.save_weights("weights/model_weight"+str(epoch)+".h5")
    print("save model.\n----\n")

    # s1 = [(w.name, w.shape) for w in model.trainable_variables]
    # for s in s1:
    #     print(s)

