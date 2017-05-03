# -*- coding: UTF-8 -*-

'''
Created on 2017年4月21日

@author: superhy
'''

'''
need to fix into handle the sentences
'''

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
import sys

from interface.embedding import word2Vec
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

def w2v_tensorization(corpus, vocab, vocab_indices, w2v_model, contLength=10):
    # need input word2vec model for query the word embeddings
    
    # cut the text in semi-redundant sequences of contLength characters
    step = 2
    sentences = []
    next_chars = []
    for text in corpus:
        if len(text) + 1 <= contLength:
            continue
        for i in range(0, len(text) - contLength, step):
            if text[i + contLength] not in vocab:
                continue
            sentences.append(text[i: i + contLength])
            next_chars.append(text[i + contLength])
    print('nb sequences:', len(sentences))

    # vectorization one-hot vector space
    # set up state transfer matrix in tensor space
    print('Word2vec_Tensorization...')
    x_train = np.zeros((len(sentences), contLength, w2v_model.vector_size), dtype=np.float32)
    y_train = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            if word in vocab:
                vocab_vector = word2Vec.getWordVec(w2v_model, word)
            else:
                vocab_vector = np.zeros((w2v_model.vector_size), dtype=np.float32)
            # use Chinese wordvec as the training space
            x_train[i, t] = vocab_vector
        y_train[i, vocab_indices[next_chars[i]]] = 1

    return x_train, y_train

def RNN_core(w2v_dim, indices_dim, contLength=10):
    ''' build the model: a single LSTM '''
    
    # some parameter
    dropout_rate = 0.2
    
    print('Build RNN model...')
    model = Sequential()
    model.add(SimpleRNN(output_dim=128, input_shape=(contLength, w2v_dim)))
    if dropout_rate > 0:
        model.add(Dropout(p=dropout_rate))
    
    model.add(Dense(output_dim=indices_dim))
    model.add(Activation('softmax'))
    
    rms_optimizer = RMSprop(lr=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=rms_optimizer)
    
    return model

def LSTM_core(w2v_dim, indices_dim, contLength=10):
    ''' build the model: a single LSTM '''
    
    # some parameter
    dropout_rate = 0.0
    
    print('Build LSTM model...')
    model = Sequential()
    model.add(LSTM(output_dim=128, input_shape=(contLength, w2v_dim)))
    if dropout_rate > 0:
        model.add(Dropout(p=dropout_rate))
    
    model.add(Dense(output_dim=indices_dim))
    model.add(Activation('softmax'))
    
    rms_optimizer = RMSprop(lr=0.001)
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
    probas = np.random.multinomial(1, preds, size=1)
    
    return np.argmax(probas)

def trainer(corpus, vocab, vocab_indices, w2v_model, contLength=10):
    '''
    need to pre-load the training data include:
    1. the corpus include list of question sentences
    2. the vocab include all words
    3,4. the dicts of (word, indicate) and (indicate, word)
    '''
    
    # some parameters
    nbIter = 20
    
    # load tensor data
    x_train, y_train = w2v_tensorization(corpus, vocab, vocab_indices, w2v_model, contLength)
    input_dim = len(vocab)
    generator = LSTM_core(w2v_dim=w2v_model.vector_size, indices_dim=input_dim, contLength=contLength)
    
    for _iter in range(0, nbIter):
        print('')
        print('-' * 50)
        print('Iteration', _iter)
        
        generator.fit(x_train, y_train, batch_size=256, nb_epoch=1)  # keras 2.0: nb_epoch changed to epochs
        
    return generator
        
def generator(generator, prefix_inputs, indices_vocab, w2v_model, contLength=10):
    
    # some parameters
    diversity = 0.5
#     diversity = 1.0
    generateLength = 30
    
    print('----- diversity:', diversity)
    
    generateContext = []
    generateContext.extend(prefix_inputs)
    print('----- Generating with seed: '),
    for prefix in prefix_inputs:
        sys.stdout.write(prefix.split('/')[0].encode('utf-8'))
    print('')
        
    print('-----Generating text: ')
    for word in generateContext:
        sys.stdout.write(word.split('/')[0])
    
    for _i in range(generateLength):
        x = np.zeros((1, contLength, w2v_model.vector_size))
        prefix_inputs = prefix_inputs[:contLength] if len(prefix_inputs) > contLength else prefix_inputs
        for t, word in enumerate(prefix_inputs):
            vocab_vector = word2Vec.getWordVec(w2v_model, word)
            x[0, t + (contLength - len(prefix_inputs))] = vocab_vector
    
        preds = generator.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_vocab[next_index]
    
        generateContext.append(next_char)
        prefix_inputs = prefix_inputs[1:] + [next_char]
        
        sys.stdout.write(next_char.split('/')[0])
        sys.stdout.flush()
    print('')   
    
    return generateContext

#===============================================================================
# 
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
