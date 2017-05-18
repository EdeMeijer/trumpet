def _split_test_train(data):
    test_examples = round(data.shape[0] * 0.1)
    return data[:test_examples], data[test_examples:]


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
