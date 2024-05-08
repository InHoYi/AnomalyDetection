import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping

number_of_features = 1

seed = 42
np.random.seed(seed)       
random.seed(seed)          
tf.random.set_seed(seed)


def get_model(window_size, number_of_features):
    model = keras.Sequential()

    # Encoder
    model.add(keras.layers.LSTM(64, input_shape = (window_size, number_of_features), activation='relu', return_sequences=True))
    model.add(keras.layers.LSTM(32, return_sequences=False))
    model.add(keras.layers.Dense(16, activation='relu'))

    # Decoder
    model.add(keras.layers.RepeatVector(window_size)) 
    model.add(keras.layers.LSTM(32, return_sequences=True))
    model.add(keras.layers.LSTM(64, return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(number_of_features)))

    return model


def train(model, train_X, valid_X, set_epochs = None):
    model.compile(optimizer = 'adam', loss = 'mae', metrics = ['mae'])
    earlyStopping = EarlyStopping( monitor = 'val_loss', patience = 50, verbose = 1, restore_best_weights = True)
    # history = model.fit(train_X, train_X, epochs = set_epochs, batch_size = 32, validation_data = (valid_X, valid_X), callbacks = [earlyStopping])
    history = model.fit(train_X, train_X, epochs = set_epochs, batch_size = 32, validation_split = 0.1, callbacks = [earlyStopping])
    return history

def print_summary(model):
    result = model.summary()
    print(result)

