# -*- coding: UTF-8 -*-

'''
Created on 2016年11月22日

@author: super
''' 

import time

from core import layer
from recog.test.testMedQuesRec import testLoadData, testTrainNetPred, \
    testEvalNetPred, testShowNetPred
    

'''
1. laod train and test data
2. train network predictor model
3. evaluate network predictor model
*4. run predict by network model
'''
'''
1. train model
2. evaluate the model
'''
def one_data(lb_data=0, name_net='CNNs_Net'):
    # exp_param
    
        
    xy_data, input_shape = testLoadData(lb_data=lb_data)
    model = testTrainNetPred(xy_data, input_shape, name_net=name_net, lb_data=lb_data)
    score = testEvalNetPred(xy_data, model)
    # testRunNetPred(xy_data, model)
    
one_data()

'''
batch process as above operation from data 0~9
'''
def batch_allData(name_net='CNNs_Net'):
    scores = []     
    for i in range(10):
        lb_data = i
        xy_data, input_shape = testLoadData(lb_data=lb_data)
        model = testTrainNetPred(xy_data, input_shape, name_net=name_net, lb_data=lb_data)
        score = testEvalNetPred(xy_data, model)
        scores.append(score)
    print(scores)
    fw = open(name_net + 'batch_scores.txt', 'w')
    fw.write(name_net + '\n' + '\n'.join(str(s) for s in scores))
    fw.close()
    
# batch_allData()

'''
batch process all model in all data 0~9
'''
def batch_allModel_allData():
#     name_nets = ['CNNs_Net', 'GRU_Net', 'BiDirtGRU_Net', 'LSTM_Net', 'BiDirtLSTM_Net', 'StackLSTMs_Net']
    '''except CNNs_Net'''
    name_nets = ['GRU_Net', 'BiDirtGRU_Net', 'LSTM_Net', 'BiDirtLSTM_Net', 'StackLSTMs_Net']
    for name_net in name_nets:
        scores = []
        for i in range(10):
            lb_data = i
            xy_data, input_shape = testLoadData(lb_data=lb_data)
            model = testTrainNetPred(xy_data, input_shape, name_net=name_net, lb_data=lb_data)
            score = testEvalNetPred(xy_data, model)
            scores.append(score)
        print(scores)
        fw = open(name_net + 'batch_scores.txt', 'w')
        fw.write(name_net + '\n' + '\n'.join(str(s) for s in scores))
        fw.close()

# batch_allModel_allData()

'''
1. fig the model framework picture
(inux only)
'''
#===============================================================================
# # exp_param
# lb_data = 0
# name_net = 'CNNs_Net'
# 
# xy_data, input_shape = testLoadData(lb_data=lb_data)
# testShowNetPred(input_shape=input_shape, name_net=name_net)
#===============================================================================
