import os

import numpy as np

CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/cache'


def load():
    return DataSet(
        np.load(CACHE_DIR + '/features.npy'),
        np.load(CACHE_DIR + '/labels.npy'),
        np.load(CACHE_DIR + '/mask.npy')
    )


class DataSet:
    def __init__(self, features, labels, mask):
        self.features = features
        self.labels = labels
        self.mask = mask

    def split_test_train(self, test_ratio=0.1):
        n_test = round(self.features.shape[0] * test_ratio)

        test = DataSet(self.features[:n_test], self.labels[:n_test], self.mask[:n_test])
        train = DataSet(self.features[n_test:], self.labels[n_test:], self.mask[n_test:])

        return test, train
