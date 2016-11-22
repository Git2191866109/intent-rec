# -*- coding: UTF-8 -*-

'''
Created on 2016年11月21日

@author: superhy
'''
import time

import numpy

from classifier import layer
from recognizer import fileProcess
from recognizer.embedding import word2Vec


def loadTrainTestMatData(trainTestFileTuples, gensimW2VModelPath, nb_classes):
    
    fr_train = open(trainTestFileTuples[0], 'r')
    fr_test = open(trainTestFileTuples[1], 'r')
    trainLines = fr_train.readlines()
    testLines = fr_test.readlines()
    fr_train.close()
    fr_test.close()
    
    start_load = time.clock()
    gensimW2VModel = word2Vec.loadModelfromFile(gensimW2VModelPath)
    max_len = 0
    for line in trainLines + testLines:
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        max_len = len(words) if len(words) > max_len else max_len
    
    trainMatList = []
    trainLabelList = []
    for line in trainLines:
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        label = line[line.find(']') + 1: len(line)]
        lineVecs = numpy.zeros([max_len, gensimW2VModel.vector_size])
        for i in range(len(words)):
            if words[i].decode('utf-8') in gensimW2VModel.vocab:
#                 lineVecs.append(word2Vec.getWordVec(gensimW2VModel, word))
                vector = word2Vec.getWordVec(gensimW2VModel, words[i])
                lineVecs[i] = numpy.asarray(vector, dtype='float32')
        trainMatList.append(lineVecs)
        
        classesVec = numpy.zeros(nb_classes)
        classesVec[int(label) - 1] = 1
        trainLabelList.append(classesVec)
        
    testMatList = []
    testLabelList = []
    for line in testLines:
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        label = line[line.find(']') + 1: len(line)]
        lineVecs = numpy.zeros([max_len, gensimW2VModel.vector_size])
        for i in range(len(words)):
            if words[i].decode('utf-8') in gensimW2VModel.vocab:
#                 lineVecs.append(word2Vec.getWordVec(gensimW2VModel, word))
                vector = word2Vec.getWordVec(gensimW2VModel, words[i])
                lineVecs[i] = numpy.asarray(vector, dtype='float32')
        testMatList.append(lineVecs)
        
        classesVec = numpy.zeros(nb_classes)
        classesVec[int(label) - 1] = 1
        testLabelList.append(classesVec)
    
    x_train = numpy.asarray(trainMatList)
    y_train = numpy.asarray(trainLabelList)
    x_test = numpy.asarray(testMatList)
    y_test = numpy.asarray(testLabelList)
    xy_data = (x_train, y_train, x_test, y_test)
    end_load = time.clock()
    print('finish load train and test numpy array data in {0}s'.format(end_load - start_load))
    
    input_shape = (max_len, gensimW2VModel.vector_size)
    
    return xy_data, input_shape

def trainNetworkPredictor(x_train, y_train,
                          input_shape,
                          nb_classes,
                          network='CNNs_Net'):
    
    # reflect produce network model
    model = getattr(layer, network)(input_shape, nb_classes)
    model = layer.trainer(model, x_train, y_train, auto_stop=False)
    
    return model

def runNetworkPredictor(network_model, x_test):
    
    classes, proba = layer.predictor(network_model, x_test)
    return classes, proba

def evaluateNetworkPredictor(network_model, x_test, y_test):
    
    score = layer.evaluator(network_model, x_test, y_test)
    return score

if __name__ == '__main__':
    
    '''
    test load train and test data
    '''
    trainFilePath = fileProcess.auto_config_root() + u'exp_mid_data/train_test/train0.txt'
    testFilePath = fileProcess.auto_config_root() + u'exp_mid_data/train_test/test0.txt'
    trainTestFileTuples = (trainFilePath, testFilePath)
    gensimW2VModelPath = fileProcess.auto_config_root() + u'model_cache/gensim/med_qus-5000.vector'
     
    xy_data, input_shape = loadTrainTestMatData(trainTestFileTuples, gensimW2VModelPath, 11)
     
    print('x_train: {0}'.format(xy_data[0]))
    print(xy_data[0].shape)
    print('y_train: {0}'.format(xy_data[1]))
    print(xy_data[1].shape)
#     print(len(set(xy_data[1]])))
    print('x_test: {0}'.format(xy_data[2]))
    print('y_test: {0}'.format(xy_data[3]))
#     print(len(set(xy_data[3])))
    print('input_shape: {0}'.format(input_shape))

    '''
    '''
#     print(dir(layer))
#     function = getattr(layer, 'LSTM_Net')
#     model = function((100,), 10)
#     print(model)
