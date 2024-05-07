import pandas as pd
import numpy as np


class LoadDataset:

    def __init__(self, window_size=0, numberOfFeatures=0):
        self.window_size = window_size
        self.numberOfFeatures = numberOfFeatures

    def setWindowSize(self, inputNumber):
        self.window_size = inputNumber

    def setNumberOfFeatures(self, inputNumber):
        self.numberOfFeatures = inputNumber


    def createDataset(self, dataset, shuffle = None):
        X = []
        y = []
        for i in range(len(dataset) - self.window_size):
            X.append(dataset[i:(i + self.window_size)])
            y.append(dataset[i + self.window_size][0])
        
        X = np.array(X)
        y = np.array(y)
        if (shuffle == True):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
        
            return X_shuffled, y_shuffled
        
        else:
            return X, y

    ##############################################################################################

    def load_trainData(self, shuffle = None):
        train_X, train_Y = self.createDataset(self.train, self.window_size, shuffle = shuffle)
        train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], self.numberOfFeatures)
        return train_X, train_Y

    def load_validData(self, shuffle = None):
        valid_X, valid_Y = self.createDataset(self.valid, self.window_size, shuffle = shuffle)
        return valid_X, valid_Y

    def load_testData(self, shuffle = None):
        test_X, test_Y = self.createDataset(self.test, self.window_size, shuffle = shuffle)
        return test_X, test_Y

    ##############################################################################################
    
    def load_OriginalData(self, shuffle = None):
        Original_X, _ = self.createDataset(self.datasample, self.window_size, shuffle = shuffle)
        Original_X = Original_X.reshape(Original_X.shape[0], Original_X.shape[1], 2)
        return Original_X, _
    
    def load_DoorOpenData(self, shuffle = None):
        Opened_X, _ = self.createDataset(self.OpenedSample, self.window_size, shuffle = shuffle)
        Opened_X = Opened_X.reshape(Opened_X.shape[0], Opened_X.shape[1], 2)
        return Opened_X, _