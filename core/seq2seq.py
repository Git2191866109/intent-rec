# -*- coding: UTF-8 -*-

'''
Created on 2017年4月21日

@author: superhy
'''

import numpy as np

def onehot_tensorization(text, vocab, vocab_indices):
    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 10
    step = 2
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    # vectorization one-hot vector space
    # set up state transfer matrix in tensor space
    print('Onehot_Tensorization...')
    x_train = np.zeros((len(sentences), maxlen, len(vocab)), dtype=np.bool)
    y_train = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            x_train[i, t, vocab_indices[word]] = 1
        y_train[i, vocab_indices[next_chars[i]]] = 1

    return x_train, y_train

def w2v_tensorization(text, vocab, vocab_indices, w2v_model):
    # need input word2vec model for query the word embeddings
    
    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 10
    step = 2
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    # vectorization one-hot vector space
    # set up state transfer matrix in tensor space
    print('Word2vec_Tensorization...')
    x_train = np.zeros((len(sentences), maxlen, len(vocab)), dtype=np.bool)
    y_train = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            x_train[i, t, vocab_indices[word]] = 1
        y_train[i, vocab_indices[next_chars[i]]] = 1

    return x_train, y_train

if __name__ == '__main__':
    pass