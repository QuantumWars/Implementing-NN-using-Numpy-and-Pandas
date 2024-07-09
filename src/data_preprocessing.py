import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.data = np.array(self.data)
        self.m, self.n = self.data.shape
        np.random.shuffle(self.data)

    def process_data(self, dev_size=1000):
        data_dev = self.data[0:dev_size].T
        Y_dev = data_dev[0]
        X_dev = data_dev[1:self.n]
        X_dev = X_dev / 255.

        data_train = self.data[dev_size:self.m].T
        Y_train = data_train[0]
        X_train = data_train[1:self.n]
        X_train = X_train / 255.
        _, m_train = X_train.shape

        return X_train, Y_train, X_dev, Y_dev, m_train
