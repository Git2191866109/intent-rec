# -*- coding: UTF-8 -*-

'''
Created on 2017年5月3日

@author: superhy
'''

from keras.layers import Input
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, Model

from interface.embedding import word2Vec
import numpy as np


def w2v_seqs_tensorization(corpus, vocab, vocab_indices, w2v_model, ques_token_len, ans_token_len):
    # need input word2vec model for query the word embeddings
    
    # corpus is the list of tuples include pair like: (ques, ans)
    #===========================================================================
    # ques_token_len and ans_token_len are the max length of
    # question sequences and answer sequences, which are setted by system
    # the length of generated answer is expected as ans_token_len
    #===========================================================================
    
    ''' keras need output_length equal as input_length '''
    normalized_token_len = max(ques_token_len, ans_token_len) 
    
    x_train = np.zeros((len(corpus), normalized_token_len, w2v_model.vector_size), dtype=np.float32)
    y_train = np.zeros((len(corpus), normalized_token_len, len(vocab)), dtype=np.bool)
    for qa_index, qa_tuple in enumerate(corpus):
        ques_sentence = qa_tuple[0]
        ans_sentence = qa_tuple[1]
        for ques_t_index, ques_token in enumerate(ques_sentence[ : normalized_token_len]):
            if ques_token in vocab:
                x_train[qa_index, ques_t_index] = word2Vec.getWordVec(w2v_model, ques_token)
        for ans_t_index, ans_token in enumerate(ans_sentence[ : normalized_token_len]):
            if ans_token in vocab:
                y_train[qa_index, ans_t_index, vocab_indices[ans_token]] = 1
                
    return x_train, y_train

def LSTM_core(w2v_dim, indices_dim, token_len):
    ''' build the model: a simple RNN encoder-decoder framework '''
    
    # some parameter
    encoder_dropout = 0.0
    decoder_dropout = 0.0
    
    encoder_hidden_size = 128
    decoder_hidden_size = 128
    
    output_activation = 'softmax'
    
    encoder = Sequential()
    encoder.add(LSTM(output_dim=encoder_hidden_size, input_shape=(token_len, w2v_dim), return_sequences=True, dropout_U=encoder_dropout))
    
    decoder = Sequential()
    decoder.add(LSTM(output_dim=decoder_hidden_size, return_sequences=True, dropout_U=decoder_dropout))
    decoder.add(TimeDistributed(Dense(output_dim=indices_dim, activation=output_activation)))
    
    ques_input = Input(shape=((token_len, w2v_dim)))
    
    encoded = encoder(ques_input)
    decoded = decoder(encoded)
    
    model = Model(input=ques_input, output=decoded)
    return model

if __name__ == '__main__':
    pass
