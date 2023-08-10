import time

import numpy as np
from sklearn.metrics import roc_auc_score

from utilities import commons

class BaseKnn:

    def create_index(self):
        pass

    def get_nearest_neighbors(self, query):
        pass

    def get_index_size(self):
        return commons.get_size(self.index, 0)

    def predict_single(self, query, k=10):
        tik = time.perf_counter()
        results, dists = self.get_nearest_neighbors(query, k)
        results = results[0]
        prediction = np.bincount(self.labels[results]).argmax().item()
        tok = time.perf_counter()
        return prediction, tok - tik

    def predict(self, test_features, k=10):
        predictions = []
        for i, test_feature in enumerate(test_features):
            predictions.append(self.predict_single(test_feature, k))
        return predictions

    def predict_probability(self, query, k):
        results, dists = self.get_nearest_neighbors(query, k)
        results = results[0]
        result_labels = self.labels[results]
        unique_labels = np.unique(self.labels)
        if len(unique_labels) > 2:
            probability = []
            for label in unique_labels:
                probability.append(np.count_nonzero(result_labels == label) / len(result_labels))
        else:
            probability = np.count_nonzero(result_labels == np.max(unique_labels)) / len(result_labels)
        return probability

    def get_auc_score(self, test_features, test_lables, k):
        print(np.unique(test_lables))
        y_scores = []
        for test_feature in test_features:
            probability = self.predict_probability(test_feature, k)
            y_scores.append(probability)
        y_scores = np.array(y_scores)
        print(f'y_scores.shape = {y_scores.shape}')
        roc_auc = roc_auc_score(test_lables, y_scores, multi_class='ovr')
        roc_auc = max(roc_auc, 1 - roc_auc)
        return roc_auc
    def get_avg_distance_computations(self, test_features, test_labels, k):
        pass
