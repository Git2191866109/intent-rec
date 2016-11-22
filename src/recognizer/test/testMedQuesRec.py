# -*- coding: UTF-8 -*-

'''
Created on 2016��11��22��

@author: super
'''
from recognizer import fileProcess
from recognizer import medQuesRec
from classifier import layer

def testLoadData():
    trainFilePath = fileProcess.auto_config_root() + u'exp_mid_data/train_test/train0.txt'
    testFilePath = fileProcess.auto_config_root() + u'exp_mid_data/train_test/test0.txt'
    gensimW2VModelPath = fileProcess.auto_config_root() + u'model_cache/gensim/med_qus-5000.vector'
    
    trainTestFileTuples = (trainFilePath, testFilePath)
    return medQuesRec.loadTrainTestMatData(trainTestFileTuples, gensimW2VModelPath, nb_classes=11)

def testTrainNetPred(xy_data, input_shape):
    x_train = xy_data[0]
    y_train = xy_data[1]
    model = medQuesRec.trainNetworkPredictor(x_train, y_train, input_shape, nb_classes=11, network='LSTM_Net')
    
    return model

def testRunNetPred(xy_data, model):
    x_test = xy_data[2]
    y_test = xy_data[3]
    classes, proba = medQuesRec.runNetworkPredictor(model, x_test)
    
    for i in range(11):
        print('{0} '.format(list(classes).count(i))),
    print('')

def testEvalNetPred(xy_data, model):
    x_test = xy_data[2]
    y_test = xy_data[3]
    score = medQuesRec.evaluateNetworkPredictor(model, x_test, y_test)
    
    print(score)

if __name__ == '__main__':
    
    '''
    1. laod train and test data
    2. train network predictor model
    3. evaluate network predictor model
    *4. run predict by network model
    '''
    xy_data, input_shape = testLoadData()
    model = testTrainNetPred(xy_data, input_shape)
    testEvalNetPred(xy_data, model)
#     testRunNetPred(xy_data, model)
