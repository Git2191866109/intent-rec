# -*- coding: UTF-8 -*-

'''
Created on 2016年11月18日

@author: superhy
'''
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop

import warnings


#===============================================================================
# structure layer-net models
#===============================================================================
def CNNs_Net(input_shape, nb_classes):
    
    # set some fixed parameter in Convolution layer
    nb_filter = 260  # convolution core num       
    filter_length = 5  # convolution core size
    border_mode = 'valid'
    cnn_activation = 'relu'
    subsample_length = 1
    # set some fixed parameter in MaxPooling layer
#     when pool_length==0: use global max-pooling, need not pool_length
    pool_length = 4
    
    # set some fixed parameter in Dense layer
#     hidden_dims = 64
    # set some fixed parameter in Dropout layer
    dropout_rate = 0.5
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
    if pool_length == 0:
        model.add(GlobalMaxPooling1D())
    else:
        model.add(MaxPooling1D(pool_length=pool_length))
        model.add(Flatten())
#     model.add(Dense(hidden_dims))
    if dropout_rate > 0:
        model.add(Dropout(p=dropout_rate))
    # output layer
    model.add(Dense(nb_classes))
    model.add(Activation(activation=final_activation))
    # compile the layer model
    sgd = SGD(lr=0.05, decay=1e-6, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

def GRU_Net(input_shape, nb_classes):
    # set some fixed parameter in LSTM layer
    gru_output_size = 64
    gru_activation = 'tanh'
    # set some fixed parameter in Dense layer
#     hidden_dims = 40
    # set some fixed parameter in Dropout layer
    dropout_rate = 0.5
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
        model.add(GRU(output_dim=gru_output_size, activation=gru_activation,
                      input_dim=input_shape[0]))
    else:
        model.add(GRU(output_dim=gru_output_size, activation=gru_activation,
                      input_shape=input_shape))
#     model.add(Dense(hidden_dims))
    if dropout_rate > 0:
        model.add(Dropout(p=dropout_rate))
    # output layer     
    model.add(Dense(nb_classes))
    model.add(Activation(activation=final_activation))
    # compile the layer model
    rmsprop = RMSprop(lr=0.002)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
    
    return model

def LSTM_Net(input_shape, nb_classes):
    
    # set some fixed parameter in LSTM layer
    lstm_output_size = 64
    lstm_activation = 'tanh'
    # set some fixed parameter in Dense layer
#     hidden_dims = 40
    # set some fixed parameter in Dropout layer
    dropout_rate = 0.5
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
        model.add(LSTM(output_dim=lstm_output_size, activation=lstm_activation,
                       input_dim=input_shape[0]))
    else:
        model.add(LSTM(output_dim=lstm_output_size, activation=lstm_activation,
                       input_shape=input_shape))
#     model.add(Dense(hidden_dims))
    if dropout_rate > 0:
        model.add(Dropout(p=dropout_rate))
    # output layer     
    model.add(Dense(nb_classes))
    model.add(Activation(activation=final_activation))
    # compile the layer model
    rmsprop = RMSprop(lr=0.002)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
    
    return model

def CNNs_LSTM_Net(input_shape, nb_classes):
    # set some fixed parameter in Convolution layer
    nb_filter = 260  # convolution core num       
    filter_length = 5  # convolution core size
    border_mode = 'valid'
    cnn_activation = 'relu'
    subsample_length = 1
    # set some fixed parameter in MaxPooling layer
    pool_length = 4
    # set some fixed parameter in LSTM layer
    lstm_output_size = 180
    lstm_activation = 'tanh'
    # set some fixed parameter in Dense layer
#     hidden_dims = 400
    # set some fixed parameter in Dropout layer
    dropout_rate = 0.5
    # set some fixed parameter in Activation layer
    final_activation = 'softmax'
    
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
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(output_dim=lstm_output_size, activation=lstm_activation))
#     model.add(Dense(hidden_dims))
#     model.add(Activation(activation=cnn_activation))
    if dropout_rate > 0:
        model.add(Dropout(p=dropout_rate))
    # output layer 
    model.add(Dense(nb_classes))
    model.add(Activation(activation=final_activation))   
    # compile the layer model
#     sgd = SGD(lr=0.05, decay=1e-6, nesterov=True)
    rmsprop = RMSprop(lr=0.002)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
    
    return model

def LSTM_CNNs_Net(input_shape, nb_classes):
    # set some fixed parameter in LSTM layer
    lstm_output_size = 200
    lstm_activation = 'tanh'
    # set some fixed parameter in Convolution layer
    nb_filter = 180  # convolution core num       
    filter_length = 5  # convolution core size
    border_mode = 'valid'
    cnn_activation = 'relu'
    subsample_length = 1
    # set some fixed parameter in MaxPooling layer
    pool_length = 4
    # set some fixed parameter in Dense layer
#     hidden_dims = 400
    # set some fixed parameter in Dropout layer
    dropout_rate = 0.5
    # set some fixed parameter in Activation layer
    final_activation = 'softmax'
    
    # check input_shape
    if len(input_shape) > 2 or len(input_shape) < 1:
        warnings.warn('input_shape is not valid!')
        return None
    
    '''produce deep layer model with sequential structure'''
    model = Sequential()
    # hidden layer
    if len(input_shape) == 1:
        model.add(LSTM(output_dim=lstm_output_size, activation=lstm_activation,
                       return_sequences=True,
                       input_dim=input_shape[0]))
    else:
        model.add(LSTM(output_dim=lstm_output_size, activation=lstm_activation,
                       return_sequences=True,
                       input_shape=input_shape))
    model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode=border_mode,
                                activation=cnn_activation,
                                subsample_length=subsample_length))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(Flatten())
#     model.add(Dense(hidden_dims))
#     model.add(Activation(activation=cnn_activation))
    if dropout_rate > 0:
        model.add(Dropout(p=dropout_rate))
    # output layer 
    model.add(Dense(nb_classes))
    model.add(Activation(activation=final_activation))   
    # compile the layer model
#     sgd = SGD(lr=0.05, decay=1e-6, nesterov=True)
    rmsprop = RMSprop(lr=0.002)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
    
    return model

def MultiLSTM_Net(input_shape, nb_classes):
    # set some fixed parameter in LSTM layer
    lstm_init_size = 128
    lstm_size_01 = 80
    lstm_size_02 = 64
    lstm_out_size = 40
    lstm_activation = 'tanh'
    # set some fixed parameter in Dense layer
#     hidden_dims = 400
    # set some fixed parameter in Dropout layer
    dropout_rate_00 = 0.7
    dropout_rate_01W = 0.55
    dropout_rate_01U = 0.55
    dropout_rate_02W = 0.45
    dropout_rate_02U = 0.45
    dropout_rate_03 = 0.3
    # set some fixed parameter in Activation layer
    final_activation = 'softmax'
    
    # check input_shape
    if len(input_shape) > 2 or len(input_shape) < 1:
        warnings.warn('input_shape is not valid!')
        return None
    '''produce deep layer model with sequential structure'''
    model = Sequential()
    # hidden layer
    if len(input_shape) == 1:
        model.add(LSTM(output_dim=lstm_init_size, activation=lstm_activation,
                       return_sequences=True,
                       input_dim=input_shape[0]))
    else:
        model.add(LSTM(output_dim=lstm_init_size, activation=lstm_activation,
                       return_sequences=True,
                       input_shape=input_shape))
    model.add(Dropout(p=dropout_rate_00))
    model.add(LSTM(output_dim=lstm_size_01, activation=lstm_activation,
                   dropout_W=dropout_rate_01W, dropout_U=dropout_rate_01U,
                   return_sequences=True))
    model.add(LSTM(output_dim=lstm_size_02, activation=lstm_activation,
                   dropout_U=dropout_rate_02U, dropout_W=dropout_rate_02W,
                   return_sequences=True))
    model.add(LSTM(output_dim=lstm_out_size, activation=lstm_activation))
    model.add(Dropout(p=dropout_rate_03))
    # output layer 
    model.add(Dense(nb_classes))
    model.add(Activation(activation=final_activation))
    
    # compile the layer model
    rmsprop = RMSprop(lr=0.002)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
    
    return model

def MultiLSTM_MultiCNNs_Net(input_shape, nb_classes):
    pass
    
#===============================================================================
# tools function for layer-net model
#===============================================================================
def trainer(model, x_train, y_train,
            batch_size=500,
            nb_epoch=80,
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

def ploter(model, pic_path):
    '''
    @attention: this function is Linux only
    '''
    
    from keras.utils.visualize_util import plot
    
    # plot the model framework
    print(model.summary())
    plot(model, to_file=pic_path)

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
