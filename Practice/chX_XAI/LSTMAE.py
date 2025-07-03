import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

class LSTMAE:

    def __init__(self, window_size, number_of_features=1, seed=42):
        self.window_size = window_size
        self.number_of_features = number_of_features
        self.seed = seed
        self.model = None
        self.history = None

        # 시드 고정
        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def build_model(self):
        model = keras.Sequential()

        # Encoder
        model.add(keras.layers.LSTM(64, input_shape=(self.window_size, self.number_of_features),
                                    activation='relu', return_sequences=True))
        model.add(keras.layers.LSTM(32, return_sequences=False))
        model.add(keras.layers.Dense(16, activation='relu'))

        # Decoder
        model.add(keras.layers.RepeatVector(self.window_size))
        model.add(keras.layers.LSTM(32, return_sequences=True))
        model.add(keras.layers.LSTM(64, return_sequences=True))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(self.number_of_features)))

        self.model = model

    def train(self, train_X, set_epochs=100, validation_split=0.1):
        if self.model is None:
            self.build_model()

        self.model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        early_stopping = EarlyStopping(monitor='val_loss', patience = 50, verbose=1, restore_best_weights=True)

        self.history = self.model.fit(
            train_X, train_X,
            epochs=set_epochs,
            batch_size = 32,
            validation_split=validation_split,
            callbacks=[early_stopping]
        )
        return self.history

    def print_summary(self):
        if self.model:
            self.model.summary()
        else:
            print("Model is not built yet.")

    def plot_training_history(self):
        if self.history is None:
            print("No training history available.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MAE)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()