# -*- coding: UTF-8 -*-

'''
Created on 2017年4月21日

@author: superhy
'''

from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop

import numpy as np
from recog.embedding import word2Vec


def onehot_tensorization(text, vocab, vocab_indices):
    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 10
    step = 3
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
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    # vectorization one-hot vector space
    # set up state transfer matrix in tensor space
    print('Word2vec_Tensorization...')
    x_train = np.zeros((len(sentences), maxlen, len(vocab)), dtype=np.float32)
    y_train = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            vocab_vector = word2Vec.getWordVec(w2v_model, word)
            # use Chinese wordvec as the training space
            x_train[i, t] = vocab_vector
        y_train[i, vocab_indices[next_chars[i]]] = 1

    return x_train, y_train

def LSTM_core(w2v_dim):
    
    maxlen = 40
    
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, w2v_dim)))
    model.add(Dense(output_dim=w2v_dim))
    model.add(Activation('softmax'))
    
    rms_optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=rms_optimizer)
    
    return model

def sample(preds, temperature=1.0):
    
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    # \log (preds) / temperature
    preds = np.log(preds) / temperature
    # e^preds
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probas)

if __name__ == '__main__':
    pass