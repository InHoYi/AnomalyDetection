import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

window_size = 30
number_of_features = 2

seed_value = 42
np.random.seed(seed_value)       
random.seed(seed_value)          
tf.random.set_seed(seed_value)

def get_model(dataset):
    model = keras.Sequential()

    model.add(keras.layers.LSTM(64, input_shape = (dataset.shape[1], number_of_features), activation = 'relu', return_sequences = True)) 
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.LSTM(32, return_sequences = False)) 
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(16))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(8))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(1))

    return model

def train(model, train_X, train_Y, valid_X, valid_Y):
    model.compile(optimizer='adam', loss='mse', metrics=['MSE'])
    earlyStopping = EarlyStopping( monitor='val_loss', patience=30, verbose=1, restore_best_weights = True)
    history = model.fit(train_X, train_Y, epochs=500, batch_size=32, validation_data=(valid_X, valid_Y), callbacks=[earlyStopping])
    return history

def print_summary(model):
    result = model.summary()
    print(result)
