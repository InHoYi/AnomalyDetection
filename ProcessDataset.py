import pandas as pd
import numpy as np

class LoadDataset:

    def __init__(self, filepath, window_size = None, numberOfFeatures = 2):
        self.filepath = filepath
        self.window_size = window_size
        self.numberOfFeatures = numberOfFeatures
        self.data = self.load_data()
        self.data = self.extract_data()
        self.data = self.scale_data()

    def load_data(self):
        data = pd.read_csv(self.filepath)
        data['measure_dtm'] = pd.to_datetime(data['measure_dtm'])
        data.replace({"OFF": 0, "ON": 1}, inplace=True)
        return data

    def get_dataframe(self):
        result = pd.DataFrame(self.data)
        return result

    def scale_data(self):
        scaled_data = self.data.copy()
        min_val = np.min(scaled_data['attribute_1_value'])
        max_val = np.max(scaled_data['attribute_1_value'])
        scaled_data['attribute_1_value'] = (scaled_data['attribute_1_value'] - min_val) / (max_val - min_val)
        return scaled_data

    def extract_data(self):
        extracted_data = self.data[['attribute_1_value', 'attribute_2_value']]
        return extracted_data

    def train_test_split(self, set_train_test_split_ratio = 0.8, set_train_valid_split_ratio = 0.75):
        trainTestSplitLength = int(len(self.data) * set_train_test_split_ratio)
        trainValidSplitLength = int(trainTestSplitLength * set_train_valid_split_ratio)
        train = self.data.iloc[:trainValidSplitLength]
        valid = self.data.iloc[trainValidSplitLength:trainTestSplitLength]
        test = self.data.iloc[trainTestSplitLength:]
        return train, valid, test 

    def create_dataset_for_timeseires(self, shuffle = None):
        X, y = [], []

        for i in range(len(self.data) - self.window_size):
            X.append(self.data.iloc[i: i + self.window_size].values)
            y.append(self.data.iloc[i + self.window_size, 0])
        X, y = np.array(X), np.array(y)

        if shuffle:
            indices = np.random.permutation(len(X))
            X, y = X[indices], y[indices]

        return X.reshape(-1, self.window_size, self.numberOfFeatures), y
