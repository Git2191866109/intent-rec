# -*- coding: UTF-8 -*-

'''
Created on 2017年4月21日

@author: superhy
'''

from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
import sys

import numpy as np
from interface.embedding import word2Vec

'''
need to fix into handle the sentences
'''
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

def LSTM_core(w2v_dim, indices_dim, contLength=10):
    ''' build the model: a single LSTM '''
    
    # some parameter
    
    print('Build model...')
    model = Sequential()
    model.add(LSTM(output_dim=128, input_shape=(contLength, w2v_dim)))
    model.add(Dense(output_dim=indices_dim))
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
    generator = LSTM_core(w2v_dim=w2v_model.vector_size, indices_dim = input_dim, contLength = contLength)
    
    for iter in range(0, nbIter):
        print('')
        print('-' * 50)
        print('Iteration', iter)
        
        generator.fit(x_train, y_train, batch_size=128, nb_epoch=1)  # keras 2.0: nb_epoch changed to epochs
        
    return generator
        
def generator(generator, prefix_inputs, indices_vocab, w2v_model, contLength=10):
    
    # some parameters
    diversity = 1.0
    generateLength = 100
    
    print('----- diversity:', diversity)
    
    generateContext = []
    generateContext.extend(prefix_inputs)
    print('----- Generating with seed: "' + str(prefix_inputs) + '"')
    
    for word in generateContext:
        sys.stdout.write(word.split('/')[0])
    
    for i in range(generateLength):
        x = np.zeros((1, contLength, w2v_model.vector_size))
        for t, word in enumerate(prefix_inputs):
            vocab_vector = word2Vec.getWordVec(w2v_model, word)
            x[0, t] = vocab_vector
    
        preds = generator.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_vocab[next_index]
    
        generateContext.append(next_char)
        prefix_inputs = prefix_inputs[1:] + [next_char]
        
        sys.stdout.write(next_char.split('/')[0])
        sys.stdout.flush()
    print('')   
    
    return generateContext

#------------------------------------------------------------------------------ #

def run_trainer_generator(corpus, vocab, vocab_indices, indices_vocab,
                  w2v_model, profix_input, contLength=10):
    
    trainFilePath = '/home/superhy/intent-rec-file/exp_mid_data/train_test-2500/sentences_labeled27500.txt'
    
    x_train, y_train = w2v_tensorization(corpus, vocab, vocab_indices)
    
    input_dim = len(vocab)
    
    model = LSTM_core(indices_dim=input_dim)
    
    # train the model, output generated corpus after each iteration
    for iter in range(1, 20):
        print()
        print('-' * 50)
        
        print('Iteration', iter)
        model.fit(x_train, y_train, batch_size=128, nb_epoch=1)  # keras 2.0: nb_epoch changed to epochs
        
#         start_index = random.randint(0, len(corpus) - contLength - 1)
        
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)
            generated = ''
#             sentence = corpus[start_index: start_index + contLength]
            sentence = profix_input
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)
    
            for i in range(100):
                x = np.zeros((1, contLength, len(vocab)))
                for t, word in enumerate(sentence):
                    vocab_vector = word2Vec.getWordVec(w2v_model, word)
                    # use Chinese wordvec as the training space
                    x[i, t] = vocab_vector
    
                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_vocab[next_index]
    
                generated += next_char
                sentence = sentence[1:] + next_char
    
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

if __name__ == '__main__':
    pass
