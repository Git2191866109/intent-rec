# -*- coding: UTF-8 -*-

'''
Created on 2016年11月22日

@author: super
''' 
'''
1. laod train and test data
2. train network predictor model
3. evaluate network predictor model
*4. run predict by network model
'''

from recognizer.test.testMedQuesRec import testLoadData, testTrainNetPred, \
    testEvalNetPred


xy_data, input_shape = testLoadData()
model = testTrainNetPred(xy_data, input_shape)
testEvalNetPred(xy_data, model)
# testRunNetPred(xy_data, model)
