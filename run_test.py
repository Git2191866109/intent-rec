# -*- coding: UTF-8 -*-

'''
Created on 2016年11月22日

@author: super
''' 

import time

from classifier import layer
from recognizer.test.testMedQuesRec import testLoadData, testTrainNetPred, \
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
# exp_param
lb_data = 1
name_net = 'MultiLSTM_Net'
  
xy_data, input_shape = testLoadData(lb_data=lb_data)
model = testTrainNetPred(xy_data, input_shape, name_net=name_net)
score = testEvalNetPred(xy_data, model)
# testRunNetPred(xy_data, model)

'''
batch process as above operation from data 0~9
'''
# name_net = 'GRU_Net'
# scores = []
#      
# for i in range(10):
#     lb_data = i
#     xy_data, input_shape = testLoadData(lb_data=lb_data)
#     model = testTrainNetPred(xy_data, input_shape, name_net=name_net)
#     score = testEvalNetPred(xy_data, model)
#     scores.append(score)
# print(scores)
# fw = open(name_net + 'batch_scores.txt', 'w')
# fw.write(name_net + '\n' + '\n'.join(str(s) for s in scores))
# fw.close()

'''
1. plot the model framework picture
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
