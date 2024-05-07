import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

window_size = 30
number_of_features = 2

seed_value = 42
np.random.seed(seed_value)       
random.seed(seed_value)          
tf.random.set_seed(seed_value)

def get_modelLSTM(dataset):
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