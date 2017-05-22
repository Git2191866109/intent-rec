# -*- coding: UTF-8 -*-

'''
Created on 2017.5.21

@author: super
'''
import time

from interface import fileProcess
from interface.topic.LDA import trainLDA_Model, getAnsweringSentencesfromQAFile,\
    loadModelfromFile
from sklearn import lda


def getAnsweringSentencesTest():
    trainPath = fileProcess.auto_config_root() + u'fenke_org/zhongyi_qa_all50000.txt'
    sentences = getAnsweringSentencesfromQAFile(trainPath)
    
    for s in sentences:
        print(s)

#------------------------------------------------------------------------------ 
    
def trainLDA_ModelTest():
    
    fileProcess.reLoadEncoding()
    
    # load all file folder path
    trainDir = fileProcess.auto_config_root() + u'fenke_org/zhongyi_qa_all50000.txt'
    answeringSentences = getAnsweringSentencesfromQAFile(trainDir)
        
    ldaModelPath = fileProcess.auto_config_root() + u'model_cache/gensim/zhongyi-lda-50000.topic'
    
    start_lda = time.clock()
    lda = trainLDA_Model(sentences=answeringSentences, modelPath=ldaModelPath, multicore=False)
    end_lda = time.clock()
    print('train gensim lda model finish, use time: {0}'.format(end_lda - start_lda))
    print(lda.num_terms)
    
def loadModelfromFileTest():
    
    ldaModelPath = fileProcess.auto_config_root() + u'model_cache/gensim/zhongyi-lda-50000.topic'
    lda = loadModelfromFile(modelPath=ldaModelPath)
    
    print('number of words: {0}'.format(lda.num_terms))
    
    topics = lda.show_topics(num_topics=500, num_words=10, formatted=False)
    
    
if __name__ == '__main__':
    
#     getAnsweringSentencesTest()

#     trainLDA_ModelTest()
    loadModelfromFileTest()
    
    