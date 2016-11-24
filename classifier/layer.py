# -*- coding: UTF-8 -*-

'''
Created on 2016年11月18日

@author: superhy
'''
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, model_from_json
import warnings


#===============================================================================
# structure layer-net models
#===============================================================================
def CNNs_Net(input_shape, nb_classes):
    
    # set some fixed parameter in Convolution layer
    nb_filter = 128  # convolution core num       
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
    hidden_dims = 64
    # set some fixed parameter in Dropout layer
    dropout_rate = 0.25
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

def GRU_Net(input_shape, nb_classes):
    # set some fixed parameter in LSTM layer
    gru_output_size = 64
    gru_activation = 'tanh'
    # set some fixed parameter in Dense layer
    hidden_dims = 40
    # set some fixed parameter in Dropout layer
    dropout_rate = 0.25
    # set some fixed parameter in Activation layer
    final_activation = 'softmax'
    
    # check input_shape
    if len(input_shape) > 2 or len(input_shape) < 1:
        warnings.warn('input_shape is not valid!')
        return None
    
    # produce deep layer model with sequential structure
    model = Sequential()
    # hidden layer
    if len(input_shape) == 1:
        model.add(GRU(output_dim=gru_output_size, input_dim=input_shape[0]))
    else:
        model.add(GRU(output_dim=gru_output_size, input_shape=input_shape))
    model.add(Dense(hidden_dims))
    if dropout_rate > 0:
        model.add(Dropout(p=dropout_rate))
    model.add(Activation(activation=gru_activation))
    # output layer     
    model.add(Dense(nb_classes))
    model.add(Activation(activation=final_activation))
    # compile the layer model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return model

def LSTM_Net(input_shape, nb_classes):
    
    # set some fixed parameter in LSTM layer
    lstm_output_size = 64
    lstm_activation = 'tanh'
    # set some fixed parameter in Dense layer
    hidden_dims = 40
    # set some fixed parameter in Dropout layer
    dropout_rate = 0.25
    # set some fixed parameter in Activation layer
    final_activation = 'softmax'
    
    # check input_shape
    if len(input_shape) > 2 or len(input_shape) < 1:
        warnings.warn('input_shape is not valid!')
        return None
    
    # produce deep layer model with sequential structure
    model = Sequential()
    # hidden layer
    if len(input_shape) == 1:
        model.add(LSTM(output_dim=lstm_output_size, input_dim=input_shape[0]))
    else:
        model.add(LSTM(output_dim=lstm_output_size, input_shape=input_shape))
    model.add(Dense(hidden_dims))
    if dropout_rate > 0:
        model.add(Dropout(p=dropout_rate))
    model.add(Activation(activation=lstm_activation))
    # output layer     
    model.add(Dense(nb_classes))
    model.add(Activation(activation=final_activation))
    # compile the layer model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return model

def CNNs_LSTM_Net():
    pass

def LSTM_CNNs_Net():
    pass

#===============================================================================
# tools function for layer-net model
#===============================================================================
def trainer(model, x_train, y_train,
            batch_size=500,
            nb_epoch=500,
            validation_split=0.2,
            auto_stop=False):
    
    #===========================================================================
    # set callbacks function for auto early stopping
    # by monitor the loss or val_loss if not change any more
    #===========================================================================
    callbacks = []
    if auto_stop == True:
        monitor = 'val_acc' if validation_split > 0.0 else 'acc'
        patience = 2
        mode = 'max'
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
              batch_size=500):
    
    # predict the test data's classes with trained layer model
    classes = model.predict_classes(x_test, batch_size=batch_size)
    proba = model.predict_proba(x_test, batch_size=batch_size)
    
    return classes, proba

def evaluator(model, x_test, y_test,
              batch_size=500):
    
    # evaluate the trained layer model
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    return score

def storageModel(model, storage_path):
    '''
    use json file to store the model's framework (.json), use hdf5 file to store the model's data (.h5)
    storage_path must be with .json or nothing(just filename)
        
    when store the .json framework to storage_path, also create/store the .h5 file 
    on same path automatically .json and .h5 file have same filename
    '''
    storeFileName = storage_path
    if storage_path.find('.json') != -1:
        storeFileName = storage_path[:storage_path.find('.json')]
    storeDataPath = storeFileName + '.h5'
    storeFramePath = storeFileName + '.json'
        
    frameFile = open(storeFramePath, 'w')
    json_str = model.to_json()
    frameFile.write(json_str)  # save model's framework file
    frameFile.close()
    model.save_weights(storeDataPath, overwrite=True)  # save model's data file
        
    return storeFramePath, storeDataPath

def recompileModel(self, model):
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def loadStoredModel(storage_path, recompile=False):
    '''
    note same as previous function
    if u just use the model to predict, you need not to recompile the model
    if u want to evaluate the model, u should set the parameter: recompile as True
    '''
    storeFileName = storage_path
    if storage_path.find('.json') != -1:
        storeFileName = storage_path[:storage_path.find('.json')]
    storeDataPath = storeFileName + '.h5'
    storeFramePath = storeFileName + '.json'
        
    frameFile = open(storeFramePath, 'r')
#     yaml_str = frameFile.readline()
    json_str = frameFile.readline()
    model = model_from_json(json_str)
    if recompile == True:
        model = recompileModel(model)  # if need to recompile
    model.load_weights(storeDataPath)
    frameFile.close()
        
    return model
    
if __name__ == '__main__':
    pass
