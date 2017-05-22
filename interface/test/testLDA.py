# -*- coding: UTF-8 -*-

'''
Created on 2017.5.21

@author: super
'''
import time

from interface import fileProcess, wordSeg
from interface.topic.LDA import trainLDA_Model, getAnsweringSentencesfromQAFile, loadModelfromFile
from interface.wordSeg import singlePosSegEngine


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
    ) + u'model_cache/gensim/zhongyi-lda-50000.topic'

    start_lda = time.clock()
    lda = trainLDA_Model(sentences=answeringSentences,
                         modelPath=ldaModelPath, multicore=False)
    end_lda = time.clock()
    print('train gensim lda model finish, use time: {0}'.format(
        end_lda - start_lda))
    print(lda.num_terms)


def loadModelfromFileTest():

    ldaModelPath = fileProcess.auto_config_root(
    ) + u'model_cache/gensim/zhongyi-lda-50000.topic'
    lda = loadModelfromFile(modelPath=ldaModelPath)

    print('number of words: {0}'.format(lda.num_terms))

    topics = lda.show_topics(num_topics=500, num_words=20, formatted=False)
    for t in topics:
        topic_tuples = t[1]
        for tuple in topic_tuples:
            print(tuple[0] + u': ' + str(tuple[1])),
        print('')


def testFiltNumeTopics():

    ldaModelPath = fileProcess.auto_config_root(
    ) + u'model_cache/gensim/zhongyi-lda-50000.topic'
    lda = loadModelfromFile(modelPath=ldaModelPath)

    filter_pos = wordSeg.noun + wordSeg.verb + wordSeg.adj + wordSeg.adv

    nb_topics = 500
    scan_range = 100
    res_range = 10

    print(filter_pos)

    topics = lda.show_topics(num_topics=nb_topics,
                             num_words=scan_range, formatted=False)
    filted_topics = []
    for t in topics:
        topic_tuples = []
        for tuple in t[1]:
            if len(singlePosSegEngine(tuple[0])) == 0:
                continue
            if singlePosSegEngine(tuple[0])[0].split('/')[1] in filter_pos:
                print(tuple[0] + ' '),
                topic_tuples.append((tuple[0], tuple[1]))
            if len(topic_tuples) >= res_range:
                break
        print('')
        filted_topics.append(topic_tuples)
        
    f = open('D:/intent-rec-file/fenke_org/zhongyi-lda-500topics.txt', 'a')
    for topic in filted_topics:
        s = ''
        for tuple in topic:
            s += (tuple[0] + ': ' + str(tuple[1]) + '\n')
        print(s),
        f.write(s)
    f.close()

if __name__ == '__main__':

    #     getAnsweringSentencesTest()

    #     trainLDA_ModelTest()
    #     loadModelfromFileTest()
    testFiltNumeTopics()

#     print(WORD_POS().adv[0] == 'd')
