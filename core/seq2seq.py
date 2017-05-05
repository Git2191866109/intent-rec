# -*- coding: UTF-8 -*-

'''
Created on 2017年5月3日

@author: superhy
'''

from keras.layers import Input
from keras.layers.core import Dense, Masking
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import RMSprop
import sys

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
                
    return x_train, y_train, normalized_token_len

def LSTM_core(w2v_dim, indices_dim, token_len):
    ''' build the model: a simple RNN encoder-decoder framework '''
    
    # some parameter
    encoder_dropout = 0.0
    decoder_dropout = 0.0
    
    encoder_hidden_size = 128
    decoder_hidden_size = 128
    
    output_activation = 'softmax'
    optimizer_lr = 0.001
    
    encoder = Sequential()
    # add masking layer to skip the [0.0, 0.0, ...] part
    encoder.add(Masking(mask_value=0., input_shape=(token_len, w2v_dim)))
    encoder.add(LSTM(output_dim=encoder_hidden_size,
                     return_sequences=True,
                     dropout_U=encoder_dropout))
    decoder = Sequential()
    decoder.add(LSTM(output_dim=decoder_hidden_size,
                     return_sequences=True,
                     dropout_U=decoder_dropout))
    decoder.add(TimeDistributed(Dense(output_dim=indices_dim,
                                      activation=output_activation)))
    
    ques_input = Input(shape=((token_len, w2v_dim)))
    
    encoded = encoder(ques_input)
    decoded = decoder(encoded)
    
    model = Model(input=ques_input, output=decoded)
    
    rms_optimizer = RMSprop(lr=optimizer_lr)
    model.compile(optimizer=rms_optimizer, loss='categorical_crossentropy')
    
    return model

def sample(preds, temperature=0.5):
    
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    # \log (preds) / temperature
    preds = np.log(preds) / temperature
    # e^preds
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, size=1)
    
    return np.argmax(probas)

def trainer(corpus_tuple, vocab, vocab_indices, w2v_model, ques_token_len, ans_token_len):
    '''
    need to pre-load the training data include:
    1. the corpus_tuple include list of question sentences
    2. the vocab include all words
    3,4. the dicts of (word, indicate) and (indicate, word)
    5,6. ques_token_len, ans_token_len are counted in seq2seq interface with qa_corpus
    '''
    
    # some parameter
#     nbIter = 1 # for test
    nbIter = 20
    batch_size = 256
    
    x_train, y_train, token_len = w2v_seqs_tensorization(corpus_tuple, vocab, vocab_indices,
                                              w2v_model, ques_token_len, ans_token_len)
    vocab_dim = len(vocab)
    generator = LSTM_core(w2v_dim=w2v_model.vector_size,
                          indices_dim=vocab_dim,
                          token_len=token_len)
    
    for _iter in range(0, nbIter):
        print('')
        print('-' * 50)
        print('Iteration', _iter)
        
        generator.fit(x_train, y_train, batch_size=256, nb_epoch=1)  # keras 2.0: nb_epoch changed to epochs
        
    return generator

def chatbot(generator, ques_test_input, indices_vocab, w2v_model, token_len):
    
    # some parameters
    diversity = 0.5
#     diversity = 1.0
    
    print('----- diversity:', diversity)
    print('----- Generating with seed: '),
    for question in ques_test_input:
        sys.stdout.write(question.split('/')[0].encode('utf-8'))
    print('')
    
    print('-----Generating text: ')
    x_test = np.zeros((1, token_len, w2v_model.vector_size))
    for t, word in enumerate(ques_test_input):
        word_vector = word2Vec.getWordVec(w2v_model, word)
        x_test[0, t] = word_vector
    
    ansContext = []
    ansPreds = generator.predict(x_test, verbose=0)[0]
    for ans_pred in ansPreds:
        ans_token_index = sample(preds=ans_pred, temperature=diversity)
        ans_token = indices_vocab[ans_token_index]
        
        ansContext.append(ans_token)
        
        sys.stdout.write(ans_token.split('/')[0])
        sys.stdout.flush()
    print('')   
    
    return ansContext

#===============================================================================
# additional operation
#===============================================================================

def storageGenerator(generator, frame_path):
    frameFile = open(frame_path, 'w')
    json_str = generator.to_json()
    frameFile.write(json_str)  # save model's framework file
    frameFile.close()
    
    record_path = frame_path.replace('.json', '.h5')
    generator.save_weights(record_path, overwrite=True)  # save model's data file
        
    return frame_path, record_path

def recompileGenerator(generator):
    
#     optimizer = SGD(lr=0.1, decay=1e-5, nesterov=True)  # only CNNs_Net use SGD
    optimizer = RMSprop(lr=0.002)
    
    # ps: if want use precision, recall and fmeasure, need to add these metrics
    generator.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
    return generator

def loadStoredGenerator(frame_path, record_path, recompile=False):
        
    frameFile = open(frame_path, 'r')
#     yaml_str = frameFile.readline()
    json_str = frameFile.readline()
    generator = model_from_json(json_str)
    if recompile == True:
        generator = recompileGenerator(generator)  # if need to recompile
    generator.load_weights(record_path)
    frameFile.close()
        
    return generator

if __name__ == '__main__':
    pass
