# -*- coding: UTF-8 -*-

'''
Created on 2016年11月18日

@author: superhy
'''
from sklearn.svm.classes import SVC


def SVCClassify(x_train, y_train):
    '''
    Basic Support Vector Machine Classifier
    '''
        
    # the parameter can be set
    kernel = 'rbf'
    # init core and train it
    # if need the proba-predict result, parameter probability must be =True
    clf = SVC(kernel=kernel, probability=True)
    clf.fit(x_train, y_train)
        
    return clf

if __name__ == '__main__':
    pass