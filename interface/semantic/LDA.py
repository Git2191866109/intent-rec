# -*- coding: UTF-8 -*-

'''
Created on 2017年5月21日

@author: superhy
'''


from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore

from interface import fileProcess
from interface.semantic import WORD_POS, STOP_WORDS


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
        # can just use in linux
        # very hard for CPU, cautiously use it
        lda = LdaMulticore(
            corpus=corpus, num_topics=nb_topics, id2word=dictionary)
    else:
        lda = LdaModel(corpus=corpus, num_topics=nb_topics, id2word=dictionary)

    # save lda model on disk
    lda.save(fname=modelPath)
    print('producing lda model ... ok! model store in {0}'.format(modelPath))

    return lda


def getTopicsfromLDA(lda, nb_topics=500, scan_range=100, res_range=15):

    filter_pos = WORD_POS.noun + WORD_POS.verb + WORD_POS.adj + WORD_POS.adv

    topics = lda.show_topics(num_topics=nb_topics,
                             num_words=scan_range, formatted=False)
    filtted_topics = []
    for i in range(len(topics)):
        topic_tuples = topics[i][1]
        filtted_topic_tuples = []
        for tuple in topic_tuples:
            if tuple[0].split('/')[0] in STOP_WORDS.stop_words:
                continue
            if tuple[0].find('/') != -1 and tuple[0].split('/')[1] in filter_pos:
                filtted_topic_tuples.append(tuple)
            if len(filtted_topic_tuples) >= res_range:
                break
        filtted_topics.append(filtted_topic_tuples)

    indices_topics = dict((i, t) for i, t in enumerate(filtted_topics))

    return indices_topics


def loadModelfromFile(modelPath, readOnly=False):

    if readOnly == True:
        lda = LdaModel.load(fname=modelPath, mmap='r')
    else:
        lda = LdaModel.load(fname=modelPath)
    print('load lda model from {0} ok!'.format(modelPath))

    return lda

#------------------------------------------------------------------------------


def getAnsweringSentencesfromQAFile(qaFilePath):

    qa_file = open(qaFilePath, 'r')
    lines = qa_file.readlines()
    qa_file.close()

    sentences = []
    for line in lines:
        if line.find('-') == -1:
            continue
        ans_str = line.split('-')[1]
        sentences.append(ans_str[ans_str.find(
            '[') + 1: ans_str.find(']')].split(','))

    return sentences


if __name__ == '__main__':
    pass
