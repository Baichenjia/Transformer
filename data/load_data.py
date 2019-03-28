# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np 
import os
import time
import sys
sys.path.append("../model")
import model_params
import tensorflow as tf 
tf.enable_eager_execution()

params = model_params.BAI_PARAMS

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)\
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)    # 标点符号和文字之间添加空格
    w = re.sub(r'[" "]+', " ", w)           # 将多个空格转换为1个空格
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)  # 去除非合法的字符
    
    w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


# 读入数据, 返回一个list，其中元素是 (语言1，语言2) 的元组
def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]] 
    return word_pairs


# 构建词和编号之间的对应关系
# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()
    
    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        
        self.vocab = sorted(self.vocab)
        
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1
        
        for word, index in self.word2idx.items():
            self.idx2word[index] = word

def max_length(tensor):
    # 后续需要将所有的句子长度填充值 max_length
    return max(len(t) for t in tensor)


def load_dataset(path, num_examples):
    # creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)

    # index language using the class defined above    
    inp_lang = LanguageIndex(sp for en, sp in pairs)      # 初始化两个不同的类
    targ_lang = LanguageIndex(en for en, sp in pairs)
    
    # Vectorize the input and target languages
    
    # Spanish sentences. 一个 num_examples 长的list,每个元素是一个list，代表一个句子.元素是整形
    input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]
    
    # English sentences
    target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
    
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    
    # 在 input_tensor 中每个元素的后方填充0，扩充到 max_length_inp 长
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=max_length_inp,
                                                                 padding='post')    
    # 在 target_tensor 中每个元素的后方填充0，扩充到 max_length_tar 长
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=max_length_tar, 
                                                                  padding='post')
    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar


def prepare_tfdata(path_to_file):
    # Try experimenting with the size of that dataset
    num_examples = 30000
    input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file, num_examples)

    # Test
    # ------
    # print("Input sequence is Spanish, and output sequence is English.")
    # print("Vocab size of input data:", len(inp_lang.vocab))     # 9393
    # print("Vocab size of output data:", len(targ_lang.vocab))   # 4917
    # print("Max len of input sequence:", max_length_inp)
    # print("Max len of output sequence", max_length_targ)
    # print("Number of input tensor:", len(input_tensor))
    # print("Number of target tensor:", len(target_tensor))
    # print("Example of input and target: ")
    # rad = np.random.randint(low=0, high=num_examples)
    # print("Index =", rad)
    # print("   Input=", input_tensor[rad])
    # print("   Decode to text:", " ".join([inp_lang.idx2word[i] for i in input_tensor[rad]]))
    # print("   \nTarget=", target_tensor[rad])
    # print("   Decode to text: ", " ".join([targ_lang.idx2word[i] for i in target_tensor[rad]]))
    # -------

    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

    # Show length  24000 24000 6000 6000
    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

    # tf.data
    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = params['default_batch_size']
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    vocab_inp_size = len(inp_lang.word2idx)    # 9394
    vocab_tar_size = len(targ_lang.word2idx)   # 4918

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset, inp_lang, targ_lang, input_tensor_val, target_tensor_val

