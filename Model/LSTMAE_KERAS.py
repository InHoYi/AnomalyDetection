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

def get_model(dataset):
    model = keras.Sequential()

    model.add(keras.layers.LSTM(64, input_shape = (dataset.shape[1], dataset.shape[2]), activation='relu', return_sequences = True))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(32, return_sequences = True))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(16, return_sequences = False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(8))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.RepeatVector(dataset.shape[1]))

    model.add(keras.layers.LSTM(16, return_sequences = True))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(32, return_sequences = True))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(64, return_sequences = True))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.TimeDistributed(keras.layers.Dense(dataset.shape[2])))

    return model


def train(model, train_X, valid_X, set_epochs = None):
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
    earlyStopping = EarlyStopping( monitor = 'val_loss', patience = 20, verbose = 1, restore_best_weights = True)
    history = model.fit(train_X, train_X, epochs = set_epochs, batch_size = 16, validation_data = (valid_X, valid_X), callbacks = [earlyStopping])
    return history

def print_summary(model):
    result = model.summary()
    print(result)

