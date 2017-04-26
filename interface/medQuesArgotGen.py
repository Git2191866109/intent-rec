# -*- coding: UTF-8 -*-

'''
Created on 2017年4月25日

@author: super
'''
from embedding import word2Vec
from core import seq2seq
from gensim.models import word2vec


def loadSentenceVocabData(trainFilePath, gensimW2VModelPath):
    # load file data
    fr_train = open(trainFilePath, 'r')
    trainLines = fr_train.readlines()
    fr_train.close()
    del(fr_train)
    corpus = []
    for line in trainLines:
        words = list(word.decode('utf-8') for word in line[line.find('[') + 1 : line.find(']')].split(','))
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
    
    generator = seq2seq.trainer(corpus, words_vocab, vocab_indices, w2v_model)
    return generator

def runGenerator(generator, prefix_sentence,
                 indices_vocab,
                 w2v_model,
                 res_path=None):
    
    generateContext = seq2seq.generator(generator, prefix_sentence, indices_vocab, w2v_model)
    return generateContext

if __name__ == '__main__':
    pass
