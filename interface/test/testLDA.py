# -*- coding: UTF-8 -*-

'''
Created on 2017.5.21

@author: super
'''
import time

from interface import fileProcess
from interface.semantic.LDA import getAnsweringSentencesfromQAFile,\
    trainLDA_Model, getTopicsfromLDA, loadModelfromFile


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

    ldaModelPath = fileProcess.auto_config_root(
    ) + u'model_cache/gensim/zhongyi_lda50000.topic'

    start_lda = time.clock()
    lda = trainLDA_Model(sentences=answeringSentences,
                         modelPath=ldaModelPath, multicore=False)
    end_lda = time.clock()
    print('train gensim lda model finish, use time: {0}'.format(
        end_lda - start_lda))
    print(lda.num_terms)


def trainPosLDA_ModelTest():

    fileProcess.reLoadEncoding()

    # load all file folder path
    trainDir = fileProcess.auto_config_root() + u'fenke_org/zhongyi_pos_qa_all50000.txt'
    answeringSentences = getAnsweringSentencesfromQAFile(trainDir)

    ldaModelPath = fileProcess.auto_config_root(
    ) + u'model_cache/gensim/zhongyi_pos_lda50000.topic'

    start_lda = time.clock()
    lda = trainLDA_Model(sentences=answeringSentences,
                         modelPath=ldaModelPath, multicore=True)
    end_lda = time.clock()
    print('train gensim lda model with pos finish, use time: {0}'.format(
        end_lda - start_lda))
    print(lda.num_terms)


def loadModelfromFileTest():

    ldaModelPath = fileProcess.auto_config_root(
    ) + u'model_cache/gensim/zhongyi_pos_lda50000.topic'
    lda = loadModelfromFile(modelPath=ldaModelPath)

    print('number of words: {0}'.format(lda.num_terms))

    topics = lda.show_topics(num_topics=500, num_words=20, formatted=False)
#     print(topics)
    for t in topics:
        print(t)
#         topic_tuples = t[1]
#         for tuple in topic_tuples:
#             print(tuple[0] + u': ' + str(tuple[1])),
#         print('')


def getTopicsfromLDA_Test():

    ldaModelPath = fileProcess.auto_config_root(
    ) + u'model_cache/gensim/zhongyi_pos_lda50000.topic'
    lda = loadModelfromFile(modelPath=ldaModelPath)

    print('number of words: {0}'.format(lda.num_terms))

    indices_topics = getTopicsfromLDA(lda=lda, scan_range=100, res_range=15)
    
    print(len(indices_topics.keys()))
    for topic_index in indices_topics.keys():
#         print(indices_topics[topic_index])
        for t_tuple in indices_topics[topic_index]:
            print(t_tuple[0] + ': ' + str(t_tuple[1]))
        print('-------------------------------------')

if __name__ == '__main__':

    #     getAnsweringSentencesTest()

    #     trainLDA_ModelTest()
    #     trainPosLDA_ModelTest()
#     loadModelfromFileTest()
#     testFiltNumeTopics()

#     print(WORD_POS().adv[0] == 'd')

    getTopicsfromLDA_Test()


