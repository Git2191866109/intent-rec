# -*- coding: UTF-8 -*-

'''
Created on 2016年11月20日

@author: super
'''
import random
import time

from recognizer import fileProcess, cacheIndex
from recognizer.embedding import word2Vec


_totalDirPath = fileProcess.auto_config_root() + u'med_question_5000each/'

def prodRandomLabeledData(totalDirPath, writeFilePath=None):    
    # load all sentences to be trained
    totalSentences = fileProcess.loadMedQuesSentences(totalDirPath)
    
    med_qus_categories = cacheIndex.med_question_index.keys()
#     dirPath = []
#     dirPath.extend(totalDirPath + category + '/' for category in med_qus_categories)
    start_label = time.clock()
    classes = []
    for category in med_qus_categories:
        cateDirPath = totalDirPath + category + '/'
        cateFilesPath = fileProcess.listAllFilePathInDirectory(cateDirPath)
        for i in range(len(cateFilesPath)):
            classes.append(cacheIndex.med_question_index[category])
            
    totalSentences_labeled = []
    for i in range(len(totalSentences)):
        words_str = '[' + ','.join(totalSentences[i]) + ']'
        sentence_labeled = words_str + str(classes[i])
        totalSentences_labeled.append(sentence_labeled)
    end_label = time.clock()
    print('finish give labels in {0}s'.format(end_label - start_label))
    
    start_random = time.clock()
    random.shuffle(totalSentences_labeled)
    end_random = time.clock()
    print('finish random data in {0}s'.format(end_random - start_random))
    
    if writeFilePath != None:
        fw = open(writeFilePath, 'w')
        fw.write('\n'.join(totalSentences_labeled))
        fw.close()
    
    return totalSentences_labeled

if __name__ == '__main__':
    
    writeFilePath = fileProcess.auto_config_root() + u'exp_mid_data/sentences_labeled55000.txt'
    
    #===========================================================================
    # prodRandomLabeledData(_totalDirPath, writeFilePath)
    #===========================================================================
    
    '''
    test mid data index in gensim word2vec
    '''
    #===========================================================================
    # fw = open(writeFilePath, 'r')
    # line = fw.readline()
    # test_words = line[line.find('[') + 1:line.find(']')].split(',')
    # print(test_words[len(test_words) - 1])
    #  
    # w2vModelPath = fileProcess.auto_config_root() + 'model_cache/gensim/med_qus-5000.vector'
    # model = word2Vec.loadModelfromFile(w2vModelPath)
    #  
    # vector = word2Vec.getWordVec(model, test_words[len(test_words) - 1])
    # print(vector)
    #===========================================================================
