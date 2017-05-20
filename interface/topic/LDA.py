# -*- coding: UTF-8 -*-

'''
Created on 2017年5月21日

@author: superhy
'''

from gensim.models.ldamodel import *

from interface import fileProcess


def trainLDA_Model(sentences, modelPath, nb_topics=500):
    # reload for safe
    fileProcess.reLoadEncoding()
    # train lda model
    lda = LdaModel(corpus=sentences, num_topics=nb_topics, alpha='auto')
    
    # save lda model on disk
    lda.save(fname=modelPath)
    print('producing lda model ... ok! model store in {0}'.format(modelPath))
    
    return lda

if __name__ == '__main__':
    pass