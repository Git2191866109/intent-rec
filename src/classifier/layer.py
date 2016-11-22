# -*- coding: UTF-8 -*-

'''
Created on 2016年11月18日

@author: superhy
'''
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
import warnings


#===============================================================================
# structure layer-net models
#===============================================================================
def CNNs_Net(input_shape, nb_classes):
    
    # set some fixed parameter in Convolution layer
    nb_filter = 160  # convolution core num       
    filter_length = 5  # convolution core size
    border_mode = 'valid'
    cnn_activation = 'relu'
    subsample_length = 1
    # set some fixed parameter in MaxPooling layer
    #===========================================================================
    # use global max-pooling, need not pool_length
    # pool_length = 5
    #===========================================================================
    # set some fixed parameter in Dense layer
    hidden_dims = 80
    # set some fixed parameter in Dropout layer
    dropout_rate = 0.05
    # set some fixed parameter in Activation layer
    final_activation = 'softmax'
    #===========================================================================
    # # set some fixed parameter in training
    # batch_size = 4
    # nb_epoch = 50
    #===========================================================================
    
    # check input_shape
    if len(input_shape) > 2 or len(input_shape) < 1:
        warnings.warn('input_shape is not valid!')
        return None
    
    '''produce deep layer model with sequential structure'''
    model = Sequential()
    # hidden layer
    if len(input_shape) == 1:
        model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode=border_mode,
                                activation=cnn_activation,
                                subsample_length=subsample_length,
                                input_dim=input_shape[0]))
    else:
        model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode=border_mode,
                                activation=cnn_activation,
                                subsample_length=subsample_length,
                                input_shape=input_shape))
#     if pool_length == None:
#         pool_length = model.output_shape[1]
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_dims))
    if dropout_rate > 0:
        model.add(Dropout(p=dropout_rate))
    model.add(Activation(activation=cnn_activation))
    # output layer
    model.add(Dense(nb_classes))
    model.add(Activation(activation=final_activation))
    # compile the layer model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return model

def LSTM_Net(input_shape, nb_classes):
    
    # set some fixed parameter in LSTM layer
    lstm_output_size = 160
    lstm_activation = 'tanh'
    # set some fixed parameter in Dense layer
    hidden_dims = 80
    # set some fixed parameter in Dropout layer
    dropout_rate = 0.05
    # set some fixed parameter in Activation layer
    final_activation = 'softmax'
    
    # produce deep layer model with sequential structure
    sequence_input = Input(shape=input_shape)  # input layer
    # hidden layer
    layer_net = LSTM(output_dim=lstm_output_size)
    layer_net = Dense(hidden_dims)(layer_net)
    if dropout_rate > 0:
        layer_net = Dropout(p=dropout_rate)(layer_net)
    layer_net = Activation(activation=lstm_activation)(layer_net)
    # output layer
    preds = Dense(nb_classes, activation=final_activation)(layer_net)
    
    model = Model(sequence_input, preds)
    # compile the layer model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return model

def GRU_Net():
    pass

def CNNs2LSTM_Net():
    pass

def LSTM2CNNs_Net():
    pass


#===============================================================================
# tools function for layer-net model
#===============================================================================
def trainer(model, x_train, y_train,
            batch_size=4,
            nb_epoch=50,
            validation_split=0.1,
            auto_stop=False):
    
    #===========================================================================
    # set callbacks function for auto early stopping
    # by monitor the loss or val_loss if not change any more
    #===========================================================================
    callbacks = []
    if auto_stop == True:
        monitor = 'val_loss' if validation_split > 0.0 else 'loss'
        patience = 5
        mode = 'min'
        early_stopping = EarlyStopping(monitor=monitor,
                                       patience=patience,
                                       mode=mode)
        callbacks = [early_stopping]
        
    model.fit(x=x_train, y=y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_split=validation_split,
              callbacks=callbacks)
    
    return model

def predictor(model, x_test,
              batch_size=4):
    
    # predict the test data's classes with trained layer model
    classes = model.predict_classes(x_test, batch_size=batch_size)
    proba = model.predict_proba(x_test, batch_size=batch_size)
    
    return classes, proba

def evaluator(model, x_test, y_test,
              batch_size=4):
    
    # evaluate the trained layer model
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    return score
    
if __name__ == '__main__':
    pass
