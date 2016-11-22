# -*- coding: UTF-8 -*-

'''
Created on 2016年11月21日

@author: superhy
'''
import time

import numpy

from recognizer import fileProcess
from recognizer.embedding import word2Vec
from classifier import layer


def loadTrainTestMatData(trainTestFileTuples, gensimW2VModelPath):
    
    fr_train = open(trainTestFileTuples[0], 'r')
    fr_test = open(trainTestFileTuples[1], 'r')
    trainLines = fr_train.readlines()
    testLines = fr_test.readlines()
    fr_train.close()
    fr_test.close()
    
    start_load = time.clock()
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
        
    '''test no input_shape, just input_dim'''
#     for i in trainMatList:
#         for j in range(max_len - len(trainMatList[i])):
#             trainMatList[i].append(numpy.zeros(gensimW2VModel.vector_size))
#     for i in testMatList:
#         for j in range(max_len - len(testMatList[i])):
#             testMatList[i].append(numpy.zeros(gensimW2VModel.vector_size))
    
    x_train = numpy.asarray(trainMatList)
    y_train = numpy.asarray(trainLabelList)
    x_test = numpy.asarray(testMatList)
    y_test = numpy.asarray(testLabelList)
    xy_data = (x_train, y_train, x_test, y_test)
    end_load = time.clock()
    print('finish load train and test numpy array data in {0}s'.format(end_load - start_load))
    
    input_shape = (gensimW2VModel.vector_size,)
    nb_classes = len(set(trainLabelList + testLabelList))
    
    return xy_data, input_shape, nb_classes

def trainNetworkPredictor(x_train, y_train, network = 'CNNs_Net'):
    pass

if __name__ == '__main__':
    
    '''
    test load train and test data
    '''
#===============================================================================
#     trainFilePath = fileProcess.auto_config_root() + u'exp_mid_data/train_test/train0.txt'
#     testFilePath = fileProcess.auto_config_root() + u'exp_mid_data/train_test/test0.txt'
#     trainTestFileTuples = (trainFilePath, testFilePath)
#     gensimW2VModelPath = fileProcess.auto_config_root() + u'model_cache/gensim/med_qus-5000.vector'
#     
#     xy_data, input_shape, nb_classes = loadTrainTestMatData(trainTestFileTuples, gensimW2VModelPath)
#     
#     print('x_train: {0}'.format(xy_data[0]))
#     print('y_train: {0}'.format(xy_data[1]))
# #     print(len(set(xy_data[1]])))
#     print('x_test: {0}'.format(xy_data[2]))
#     print('y_test: {0}'.format(xy_data[3]))
# #     print(len(set(xy_data[3])))
#     print('input_shape: {0}'.format(input_shape))
#     print('nb_classes: {0}'.format(nb_classes))
#===============================================================================

    '''
    '''
    print(dir(layer))
    function = getattr(layer, 'CNNs_Net')
    model = function((100,), 10)
    print(model)
