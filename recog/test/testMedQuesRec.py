# -*- coding: UTF-8 -*-

'''
Created on 2016年11月21日

@author: super
'''
from recog import fileProcess, medQuesRec


def testLoadData(lb_data=0):
    trainFilePath = fileProcess.auto_config_root() + u'exp_mid_data/train_test/train{0}.txt'.format(lb_data)
    testFilePath = fileProcess.auto_config_root() + u'exp_mid_data/train_test/test{0}.txt'.format(lb_data)
    gensimW2VModelPath = fileProcess.auto_config_root() + u'model_cache/gensim/med_qus-5000.vector'
    
    npzPath = fileProcess.auto_config_root() + u'exp_mid_data/npy_data/train_NP{0}.npz'.format(lb_data)
    
    trainTestFileTuples = (trainFilePath, testFilePath)
    xy_data, input_shape = medQuesRec.loadGensimMatData(trainTestFileTuples, gensimW2VModelPath, nb_classes=11)
    medQuesRec.storeExprimentNpzData(npzPath, xy_data)
    print('store the xy_data at {0}!'.format(npzPath))
    
    return xy_data, input_shape

def testGetNpyData(lb_data=0):
    npzPath = fileProcess.auto_config_root() + u'exp_mid_data/npy_data/train_NP{0}.npz'.format(lb_data)
    xy_data, input_shape = medQuesRec.loadExprimentNpzData(npzPath)
    print('get the xy_data at {0}! input shape: {1}'.format(npzPath, input_shape))
    
    return xy_data, input_shape

def testTrainNetPred(xy_data, input_shape, name_net='CNNs_Net', lb_data=None):
    frame_path = fileProcess.auto_config_root() + u'model_cache/keras/{0}5000_{1}.json'.format(name_net, lb_data)
    x_train = xy_data[0]
    y_train = xy_data[1]
    model, history_metrices = medQuesRec.trainNetworkPredictor(x_train, y_train, input_shape,
                                                      nb_classes=11,
                                                      network=name_net,
                                                      frame_path=frame_path)
    print(history_metrices)
    
    return frame_path

def testRunNetPred(xy_data, frame_path):
    model = medQuesRec.loadNetworkPredictor(frame_path)
    x_test = xy_data[2]
    y_test = xy_data[3]
    classes, proba = medQuesRec.runNetworkPredictor(model, x_test)
    
    for i in range(11):
        print('{0} '.format(list(classes).count(i))),
    print('')

def testEvalNetPred(xy_data, frame_path):
    model = medQuesRec.loadNetworkPredictor(frame_path)
    x_test = xy_data[2]
    y_test = xy_data[3]
    score = medQuesRec.evaluateNetworkPredictor(model, x_test, y_test)
    
    print(score)
    return score

def testShowNetPred(input_shape, name_net='CNNs_Net'):
    medQuesRec.showNetworkPredictor(input_shape, nb_classes=11, network=name_net)

if __name__ == '__main__':
    
    '''
    1. laod train and test data
    2. train network predictor model
    3. evaluate network predictor model
    *4. run predict by network model
    '''
    xy_data, input_shape = testLoadData()
    print('org input_shape: {0}'.format(input_shape))
    xy_data, input_shape = testGetNpyData()
#===============================================================================
#     model = testTrainNetPred(xy_data, input_shape)
#     testEvalNetPred(xy_data, model)
# #     testRunNetPred(xy_data, model)
#===============================================================================
