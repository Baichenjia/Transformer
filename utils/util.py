import sys
sys.path.append("../model")
sys.path.append("../data")

import numpy as np
import model_params, load_data

# params
params = model_params.BAI_PARAMS

def generate_test_data(dataset, inp_lang, targ_lang):
    # init data and dict
    # print("<start> id:", inp_lang.word2idx['<start>'])
    # print("<end> id:", inp_lang.word2idx['<end>'])

    # 模拟正常的 inputs 
    inputs = np.random.randint(low=1, high=params['vocab_size_input'],
                size=(params['default_batch_size'], params['max_length_input']))
    rand = np.random.randint(low=2, high=8, size=(params['default_batch_size'], ))
    # print("rand:\n", rand)
    inputs[np.arange(0, params['default_batch_size']), -rand] = 0.
    for i in range(params['default_batch_size']):
        inputs[i, 0] = inp_lang.word2idx['<start>']
        for j in range(0, params['max_length_input']):
            if inputs[i, j] == 0:
                inputs[i, j-1] = inp_lang.word2idx['<end>']
                for k in range(j, params['max_length_input']):
                    inputs[i, k] = 0
                break
    # print(inputs)

    # 模拟正常的 targets 
    targets = np.random.randint(low=1, high=params['vocab_size_output'],
                size=(params['default_batch_size'], params['max_length_output']))
    rand = np.random.randint(low=2, high=6, size=(params['default_batch_size'], ))
    # print("rand:\n", rand)
    targets[np.arange(0, params['default_batch_size']), -rand] = 0.
    for i in range(params['default_batch_size']):
        targets[i, 0] = targ_lang.word2idx['<start>']
        for j in range(0, params['max_length_output']):
            if targets[i, j] == 0:
                targets[i, j-1] = targ_lang.word2idx['<end>']
                for k in range(j, params['max_length_output']):
                    targets[i, k] = 0
                break
    # print(targets)
    return inputs, targets

if __name__ == '__main__':
    generate_test_data()

