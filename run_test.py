# -*- coding: UTF-8 -*-

'''
Created on 2016年11月22日

@author: super
''' 
from classifier import layer
'''
1. laod train and test data
2. train network predictor model
3. evaluate network predictor model
*4. run predict by network model
'''

from recognizer.test.testMedQuesRec import testLoadData, testTrainNetPred, \
    testEvalNetPred, testShowNetPred

'''
1. train model
2. evaluate the model
'''
# exp_param
lb_data = 0
name_net = 'CNNs_Net'
 
xy_data, input_shape = testLoadData(lb_data=lb_data)
model = testTrainNetPred(xy_data, input_shape, name_net=name_net)
testEvalNetPred(xy_data, model)
# testRunNetPred(xy_data, model)

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
