# -*- coding: UTF-8 -*-

'''
Created on 2016年11月21日

@author: superhy
'''
import numpy

from recognizer.embedding import word2Vec


def loadTrainTestMatData(trainTestFileTuples, gensimW2VModelPath):
    
    fr_train = open(trainTestFileTuples[0])
    fr_test = open(trainTestFileTuples[1])
    trainLines = fr_train.readlines()
    testLines = fr_test.readlines()
    fr_train.close()
    fr_test.close()
    
    gensimW2VModel = word2Vec.loadModelfromFile(gensimW2VModelPath)
    max_len = 0
    trainMatList = []
    trainLabelList = []
    for line in trainLines:
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        label = line[line.find(']') + 1: len(line)]
        lineVecs = []
        for word in words:
            if word.decode('utf-8') in gensimW2VModel.vocab:
                lineVecs.append(word2Vec.getWordVec(gensimW2VModel, word))
        trainMatList.append(numpy.asarray(lineVecs)) 
        trainLabelList.append(int(label))
        
        max_len = len(words) if len(words) > max_len else max_len
    testMatList = []
    testLabelList = []
    for line in testLines:
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        label = line[line.find(']') + 1: len(line)]
        lineVecs = []
        for word in words:
            if word.decode('utf-8') in gensimW2VModel.vocab:
                lineVecs.append(word2Vec.getWordVec(gensimW2VModel, word))
        testMatList.append(numpy.asarray(lineVecs)) 
        testLabelList.append(int(label))
        
        max_len = len(words) if len(words) > max_len else max_len
        
    for i in trainMatList:
        for j in max_len-len(trainMatList[i]):
            trainMatList[i].append(numpy.zeros(gensimW2VModel.vector_size))

if __name__ == '__main__':
    pass
