# -*- coding: UTF-8 -*-

'''
Created on 2016年11月21日

@author: superhy
'''
import numpy
import time

from classifier import layer
from recognizer import fileProcess
from recognizer.embedding import word2Vec


def loadGensimMatData(trainTestFileTuples, gensimW2VModelPath, nb_classes):
    
    fr_train = open(trainTestFileTuples[0], 'r')
    fr_test = open(trainTestFileTuples[1], 'r')
    trainLines = fr_train.readlines()
    testLines = fr_test.readlines()
    fr_train.close()
    fr_test.close()
    del(fr_train)
    del(fr_test)
    
    
    start_load = time.clock()
    gensimW2VModel = word2Vec.loadModelfromFile(gensimW2VModelPath)
    vector_dim = gensimW2VModel.vector_size
    max_len = 0
    for line in trainLines + testLines:
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        max_len = len(words) if len(words) > max_len else max_len
    
    trainMatList = []
    trainLabelList = []
    for line in trainLines:
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        label = line[line.find(']') + 1: len(line)]
        lineVecs = numpy.zeros([max_len, vector_dim])
        for i in range(len(words)):
            if words[i].decode('utf-8') in gensimW2VModel.vocab:
#                 lineVecs.append(word2Vec.getWordVec(gensimW2VModel, word))
                vector = word2Vec.getWordVec(gensimW2VModel, words[i])
                lineVecs[i] = numpy.asarray(vector, dtype='float32')
        trainMatList.append(lineVecs)
        
        classesVec = numpy.zeros(nb_classes)
        classesVec[int(label) - 1] = 1
        trainLabelList.append(classesVec)
    del(trainLines)
        
    testMatList = []
    testLabelList = []
    for line in testLines:
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        label = line[line.find(']') + 1: len(line)]
        lineVecs = numpy.zeros([max_len, vector_dim])
        for i in range(len(words)):
            if words[i].decode('utf-8') in gensimW2VModel.vocab:
#                 lineVecs.append(word2Vec.getWordVec(gensimW2VModel, word))
                vector = word2Vec.getWordVec(gensimW2VModel, words[i])
                lineVecs[i] = numpy.asarray(vector, dtype='float32')
        testMatList.append(lineVecs)
        
        classesVec = numpy.zeros(nb_classes)
        classesVec[int(label) - 1] = 1
        testLabelList.append(classesVec)
    del(testLines)
    
    del(gensimW2VModel)
    
    x_train = numpy.asarray(trainMatList)
    del(trainMatList)
    y_train = numpy.asarray(trainLabelList)
    del(trainLabelList)
    x_test = numpy.asarray(testMatList)
    del(testMatList)
    y_test = numpy.asarray(testLabelList)
    del(testLabelList)
    xy_data = (x_train, y_train, x_test, y_test)
    end_load = time.clock()
    print('finish load train and test numpy array data in {0}s'.format(end_load - start_load))
    
    input_shape = (max_len, vector_dim)
    
    return xy_data, input_shape

def loadGensimSumVecData(trainTestFileTuples, gensimW2VModelPath, nb_classes):
    pass

def trainNetworkPredictor(x_train, y_train,
                          input_shape,
                          nb_classes,
                          network='CNNs_Net',
                          storagePath=None):
    
    # reflect produce network model
    start_train = time.clock()
    model = getattr(layer, network)(input_shape, nb_classes)
    model = layer.trainer(model, x_train, y_train)
    end_train = time.clock()
    print('finish train layer model in {0}s'.format(end_train - start_train))
    
    if storagePath != None:
        layer.storageModel(model, storagePath)
        print('layer model has been stored in path: {0}.'.format(storagePath))
    
    return model

def showNetworkPredictor(input_shape,
                          nb_classes,
                          network='CNNs_Net'):
    
    picPath = fileProcess.auto_config_root() + 'model_cache/keras/' + network + '.png'
    model = getattr(layer, network)(input_shape, nb_classes)
    layer.ploter(model, pic_path=picPath)
    print('finish generate layer model picture in path: {0}.'.format(picPath))

def loadNetworkPredictor(storagePath):
    '''
    @param @recompile:  if recompile the loaded layer model
        if u just use the model to predict, you need not to recompile the model
        if u want to evaluate the model, u should set the parameter: recompile as True
    '''
    
    model = layer.loadStoredModel(storagePath, recompile=True)
    print('load layer model from path: {0}.'.format(storagePath))
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
     
    xy_data, input_shape = loadGensimMatData(trainTestFileTuples, gensimW2VModelPath, 11)
     
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
