# -*- coding: UTF-8 -*-

'''
Created on 2016年11月18日

@author: superhy
'''

'''
structure layer-net models
'''

from keras.layers import Input
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Flatten, Dense
from keras.layers.pooling import MaxPooling1D


def CNNs(input_shape,
         nb_classes,
         validation_split=0.15,
         auto_stop=False):
    
    # set some fixed parameter in Convolution layer
    nb_filter = 160  # convolution core num       
    filter_length = 5  # convolution core size
    border_mode = 'valid'
    cnn_activation = 'relu'
    subsample_length = 1
    # set some fixed parameter in MaxPooling layer
    pool_length = 5
    # set some fixed parameter in Dense layer
    hidden_dims = 80
    # set some fixed parameter in Dropout layer
    dropout_rate = 0.05
    # set some fixed parameter in Activation layer
    final_activation = 'softmax'
    # set some fixed parameter in training
    batch_size = 4
    nb_epoch = 50
    
    # produce deep layer model with normally structure(not sequential structure)
    sequence_input = Input(shape=input_shape)
    layer_net = Convolution1D(nb_filter=nb_filter,
                              filter_length=filter_length,
                              border_mode=border_mode,
                              activation=cnn_activation,
                              subsample_length=subsample_length)(sequence_input)
    if pool_length == None:
        pool_length = layer_net.output_shape[1]
    layer_net = MaxPooling1D(pool_length=pool_length)(layer_net)
    layer_net = Flatten()(layer_net)
    layer_net = Dense(hidden_dims, activation=cnn_activation)(layer_net)
    

def LSTM():
    pass

def GRU():
    pass

def CNNs2LSTM():
    pass

def LSTM2CNNs():
    pass

'''
tools function for layer-net model
'''

def trainer():
    pass

def predictor():
    pass

def evaluator():
    pass
    
if __name__ == '__main__':
    pass
