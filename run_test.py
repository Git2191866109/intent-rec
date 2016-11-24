# -*- coding: UTF-8 -*-

'''
Created on 2016年11月22日

@author: super
'''
from recognizer.test import testMedQuesRec

'''
1. laod train and test data
2. train network predictor model
3. evaluate network predictor model
*4. run predict by network model
'''
xy_data, input_shape = testMedQuesRec.testLoadData()
model = testMedQuesRec.testTrainNetPred(xy_data, input_shape)
testMedQuesRec.testEvalNetPred(xy_data, model)
# testMedQuesRec.testRunNetPred(xy_data, model)
