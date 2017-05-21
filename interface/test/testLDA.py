# -*- coding: UTF-8 -*-

'''
Created on 2017.5.21

@author: super
'''
import time

from gensim.models import word2vec

from interface import fileProcess
from interface.topic.LDA import trainLDA_Model


def trainLDA_ModelTest():
    
    fileProcess.reLoadEncoding()
    
    # load all file folder path
    trainDir = fileProcess.auto_config_root() + u'fenke/zhongyi5000/'
    
    files = fileProcess.listAllFilePathInDirectory(trainDir)
    totalSentences = []
    for filePath in files:
        totalSentences.extend(word2vec.LineSentence(filePath))
        
    ldaModelPath = fileProcess.auto_config_root() + u'model_cache/gensim/zhongyi-5000.topic'
    
    start_lda = time.clock()
    lda = trainLDA_Model(sentences=totalSentences, modelPath=ldaModelPath, multicore=False)
    end_lda = time.clock()
    print('train gensim lda model finish, use time: {0}'.format(end_lda - start_lda))
    print(lda.num_terms)
    
if __name__ == '__main__':
    trainLDA_ModelTest()