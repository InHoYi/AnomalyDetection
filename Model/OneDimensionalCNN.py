import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping

number_of_features = 2

seed = 42
np.random.seed(seed)       
random.seed(seed)          
tf.random.set_seed(seed)

def get_model(window_size, numberOfFeatures):

    model = keras.Sequential()

    model.add(keras.layers.Conv1D(filters = 64, kernel_size = 3, padding='same', input_shape = (window_size, numberOfFeatures)))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Activation('relu'))
    
    model.add(keras.layers.GRU(20, return_sequences = True))
    model.add(keras.layers.GRU(20, return_sequences = False))
    
    model.add(keras.layers.Dense(1))  

    return model


def train(model, train_X, train_Y, valid_X, valid_Y, set_epochs = None):
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    earlyStopping = EarlyStopping( monitor='val_loss', patience = 30, verbose = 1, restore_best_weights = True)
    history = model.fit(train_X, train_Y, epochs = set_epochs, batch_size=8, validation_data=(valid_X, valid_Y), callbacks=[earlyStopping])
    return history

def print_summary(model):
    result = model.summary()
    print(result)