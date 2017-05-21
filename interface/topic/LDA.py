# -*- coding: UTF-8 -*-

'''
Created on 2017年5月21日

@author: superhy
'''


from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore

from interface import fileProcess


def trainLDA_Model(sentences, modelPath, nb_topics=500, multicore=False):
    # reload for safe
    fileProcess.reLoadEncoding()
    
    # load doc2bow
    dictionary = corpora.Dictionary(sentences)
    print('finish load dictionary!')
    corpus = [dictionary.doc2bow(text) for text in sentences]
    print('finish load doc2bow corpus!')
    # train lda model
    if multicore == True:
        lda = LdaMulticore(corpus=corpus, num_topics=nb_topics)
    else:
        lda = LdaModel(corpus=corpus, num_topics=nb_topics)
    
    # save lda model on disk
    lda.save(fname=modelPath)
    print('producing lda model ... ok! model store in {0}'.format(modelPath))
    
    return lda

if __name__ == '__main__':
    pass
