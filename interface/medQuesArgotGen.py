# -*- coding: UTF-8 -*-

'''
Created on 2017年4月25日

@author: super
'''
from gensim.models import word2vec
import time

from core import charRNN
from embedding import word2Vec


def loadSentenceVocabData(trainFilePath, gensimW2VModelPath):
    # load file data
    fr_train = open(trainFilePath, 'r')
    trainLines = fr_train.readlines()
    fr_train.close()
    del(fr_train)
   
    corpus = []
    for line in trainLines:
        words = list(word.decode('utf-8') for word in line[line.find('[') + 1 : line.find(']')].split(','))
        if len(words) > 0:
            corpus.append(words)
#     print(type(corpus[0][0]))
    
    # load words vocab indices data
    gensimW2VModel = word2Vec.loadModelfromFile(gensimW2VModelPath)
    words_vocab = gensimW2VModel.vocab.keys()
#     print(type(words_vocab[0]))
    vocab_indices = dict((w, i) for i, w in enumerate(words_vocab))
    indices_vocab = dict((i, w) for i, w in enumerate(words_vocab))
    
    return corpus, words_vocab, vocab_indices, indices_vocab, gensimW2VModel
        
def trainTextGenerator(corpus, words_vocab, vocab_indices,
                       w2v_model,
                       network='LSTM_core',
                       frame_path=None):
    '''
    network: the generator neural network core (only LSTM_core now)
    frame_path: the storage path of the neural network framework model
    '''
    
    generator = charRNN.trainer(corpus, words_vocab, vocab_indices, w2v_model)
    if frame_path != None:
        charRNN.storageGenerator(generator, frame_path)
        print('Generator has been stored in path: {0}.'.format(frame_path))
    
    return generator

def runGenerator(generator, prefix_sentence,
                 indices_vocab,
                 w2v_model,
                 res_path=None):
    
    generateContext = charRNN.generator(generator, prefix_sentence, indices_vocab, w2v_model)
    if res_path != None:
        fw = open(res_path, 'a')
        generate_text_str = 'seed: ' + ''.join(prefix_sentence) + '\n' + 'generate: ' + ''.join(generateContext) + '\n'
        fw.write(generate_text_str)
        fw.close()
    
    return generateContext

if __name__ == '__main__':
    pass
