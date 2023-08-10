import math
import random
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

import mprt.mprt
from utilities import data_preprocessing, commons
from bplustree.bplustree_index import BPlusTree

def predict(index, test_features, test_labels, k=10):
    # print("b+tree predicting {} queries".format(len(test_features)))
    test_features = np.ascontiguousarray(test_features, dtype=np.float32)
    y_score, y_pred = [], []
    total_data_points = 0
    node_search_time = 0
    pair_index_time = 0
    bplustree_start_time = time.perf_counter()
    for i, feature in enumerate(test_features):
        tik = time.perf_counter()
        result_node = index.find_nearest_neighbors(feature)
        tok = time.perf_counter()
        node_search_time += (tok - tik)
        if index.nodeIndexAlgorithm in ['kd_tree', 'ball_tree', 'pynndescent', 'ngt', 'flann_kdtree', 'flann_kmeans', 'mprt', 'faiss', 'qalsh', 'hnsw']:
            prediction, time_taken = result_node.node_index.predict_single(feature, k)
            pair_index_time += time_taken
        else:
            raise Exception('unknown algorithm {}'.format(index.nodeIndexAlgorithm))
        y_score.append(0)
        y_pred.append(prediction)
    bplustree_end_time = time.perf_counter()
    accuracy = commons.calc_accuracy(test_labels, y_pred)
    roc_auc = 0
    time_taken = bplustree_end_time - bplustree_start_time
    return {
        'error':   1 - accuracy, 'time_taken': time_taken, 'time_taken_per_prediction': time_taken / len(test_features),
        'roc_auc': roc_auc, 'data_points_explored': total_data_points / len(test_features), 'y_score': y_score
        }

def get_auc_score(bplus_index, test_features, test_labels, k=10):
    y_score, y_pred = [], []
    test_features = np.ascontiguousarray(test_features, dtype=np.float32)
    unique_labels = np.unique(test_labels)
    for i, feature in enumerate(test_features):
        result_node = bplus_index.find_nearest_neighbors(feature)
        if bplus_index.nodeIndexAlgorithm in ['kd_tree', 'ball_tree', 'flann_kdtree', 'flann_kmeans', 'ngt', 'pynndescent', 'mprt', 'qalsh', 'faiss', 'hnsw']:
            prediction_prob = result_node.node_index.predict_probability(feature, k)
            prediction, _ = result_node.node_index.predict_single(feature, k)
        else:
            raise Exception('unknown algorithm {}'.format(bplus_index.nodeIndexAlgorithm))
        if type(prediction_prob) == list:
            j = 0
            mod_prediction_prob = []
            for label in unique_labels:
                if label in result_node.unique_labels:
                    mod_prediction_prob.append(prediction_prob[j])
                    j += 1
                else:
                    mod_prediction_prob.append(0.0)
            prediction_prob = mod_prediction_prob
        y_score.append(prediction_prob)
    auc_score = roc_auc_score(test_labels, y_score, multi_class='ovr')
    return max(auc_score, 1 - auc_score)

def get_reference_point(dataset, reference_type):
    train_features, _, train_labels, _ = data_preprocessing.read_dataset(dataset)
    mod_data = np.append(train_features, np.reshape(train_labels, (len(train_labels), 1)), axis=1)
    if reference_type == 'median':
        print('returning median as reference point')
        reference_point = np.median(mod_data, axis=0)
    elif reference_type == 'random':
        print('returning random datapoint as reference point')
        referenceIndex = random.randint(0, len(mod_data))
        reference_point = mod_data[referenceIndex]
    elif reference_type == 'kmeans':
        print('returning kmeans centroid as reference point')
        reference_point = get_kmeans_centroid(mod_data)
    else:
        print('returning mean as reference point')
        reference_point = np.mean(mod_data, axis=0)
    return reference_point

def get_index_size(bplus_index, algorithm):
    size = commons.get_size(bplus_index, 0)
    bplus_index.build_index_at_nodes(algorithm, 50)
    for node in bplus_index.get_all_nodes():
        size += node.node_index.get_index_size()
    return size
def create_btree_index_v2(dataset, order, num_neighbors, reference_type='mean'):
    train_features, _, train_labels, _ = data_preprocessing.read_dataset(dataset)
    tik = time.perf_counter()
    reference_point = get_reference_point(dataset, reference_type)
    bplus_index = BPlusTree(dataset, order, reference_point, num_neighbors)
    for i in range(len(train_features)):
        bplus_index.insert(i)
    print('completed all data points')
    tok = time.perf_counter()
    print(f'built index in {tok - tik} seconds')
    return bplus_index, tok - tik

def get_kmeans_centroid(features):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(features)
    # return kmeans.cluster_centers_[0]
    return np.mean(kmeans.cluster_centers_, axis=0)

def getIndexNum(dist, sd):
    return 0
