import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors, BallTree, KDTree
import time

from base_knn import BaseKnn
from utilities import commons

class sklearn_knn(BaseKnn):
    def __init__(self, features, labels, algorithm='brute'):
        self.index = None
        if algorithm not in ['brute', 'kd_tree', 'ball_tree']:
            raise Exception('unsupported algorithm: {}'.format(algorithm))
        self.algorithm = algorithm
        self.features = features
        self.labels = np.reshape(labels, (len(labels),))
        self.labels = np.asarray(self.labels, dtype=np.int64)

    def fit(self):
        fit_start_time = time.perf_counter()
        if self.algorithm == 'ball_tree':
            self.index = BallTree(self.features)
        elif self.algorithm == 'kd_tree':
            self.index = KDTree(self.features)
        else:
            self.index = NearestNeighbors(algorithm='brute')
            self.index.fit(self.features)
        fit_end_time = time.perf_counter()
        return {'time_taken': fit_end_time - fit_start_time}

    def get_index_size(self):
        file = open('knn_index.pkl', 'wb')
        pickle.dump(self.index, file)
        file.close()
        size = os.path.getsize('knn_index.pkl')
        os.remove('knn_index.pkl')
        return size

    def get_avg_distance_computations(self, test_features, test_labels, k):
        dist_computations = 0
        self.index.reset_n_calls()
        for test_feature in test_features:
            self.predict_single(test_feature, k)
            dist_computations += self.index.get_n_calls()
            self.index.reset_n_calls()
        return dist_computations / len(test_features)
        # return dist_computations

    def get_nearest_neighbors(self, test_feature, k):
        test_feature = test_feature.reshape((1, -1))
        if self.algorithm == 'brute':
            dist, ind = self.index.kneighbors(test_feature, n_neighbors=k)
            return ind, dist
        else:
            dist, ind = self.index.query(test_feature, k)
            return ind, dist
