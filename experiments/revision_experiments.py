import sys

import numpy as np

from utilities import data_preprocessing
from bplustree import bplustree_knn

def exp_order_of_tree_btree(dataset):
    train_features, test_features, train_labels, test_labels = data_preprocessing.read_dataset(dataset)
    order_list = [i / 100 for i in range(1, 11)]
    num_neighbors = 25
    node_algorithm = 'ball_tree'
    error, idx_time, prediction_time = [], [], []
    for order_percent in order_list:
        order = int(order_percent * len(train_features))
        print(f'creating bplus index for {dataset} with order {order}')
        bplus_index, time_taken = bplustree_knn.create_btree_index_v2(dataset, order, num_neighbors, reference_type='mean')
        bplus_index.build_index_at_nodes(node_algorithm, num_neighbors)
        bst_pred_res = bplustree_knn.predict(bplus_index, test_features, test_labels)
        error.append(bst_pred_res['error'])
        prediction_time.append(round(bst_pred_res['time_taken'], 4))
        idx_time.append(round(time_taken, 4))
    print(f'{dataset} error: {error}')
    print(f'{dataset} prediction_time: {prediction_time}')
    print(f'{dataset} idx_time: {idx_time}')

def experiment_roc_auc(dataset, algorithm):
    train_features, test_features, train_labels, test_labels = data_preprocessing.read_dataset(dataset)
    k = 25
    order = int(0.07 * len(train_features))
    bplus_index, _ = bplustree_knn.create_btree_index_v2(dataset, order, k, reference_type='mean')
    bplus_index.build_index_at_nodes(algorithm, k)
    roc_auc_score = bplustree_knn.get_auc_score(bplus_index, test_features, test_labels, k)
    print(f'{dataset}: Algorithm-{algorithm} roc-auc = {roc_auc_score}')
    return roc_auc_score

def exp_num_neighbors_btree(dataset, node_algorithm='ball_tree'):
    train_features, test_features, train_labels, test_labels = data_preprocessing.read_dataset(dataset)
    order = int(0.07 * len(train_features))
    k_list = [5, 10, 15, 20, 25, 50, 100]
    bplus_index, time_taken = bplustree_knn.create_btree_index_v2(dataset, order, k_list[0], reference_type='mean')
    error, prediction_time = [], []
    for k in k_list:
        if k == 5:
            print(f'building node level indices for k= {k}')
            bplus_index.build_index_at_nodes(node_algorithm, k)
            print('completed building node level indices')
        bst_pred_res = bplustree_knn.predict(bplus_index, test_features, test_labels, k)
        error.append(bst_pred_res['error'])
        prediction_time.append(round(bst_pred_res['time_taken'], 4))
    print(f'{dataset} error: {error}')
    print(f'{dataset} prediction_time: {prediction_time}')

def exp_reference_point(dataset, node_algorithm='ball_tree'):
    train_features, test_features, train_labels, test_labels = data_preprocessing.read_dataset(dataset)
    order = int(0.07 * len(train_features))
    k_list = [5, 10, 15, 20, 25, 50, 100]
    # k_list = [5]
    reference_type_list = ['mean', 'median', 'kmeans', 'random']
    idx_creation_time = []
    # reference_type_list = ['random']
    error_map, prediction_time_map = {}, {}
    for reference_type in reference_type_list:
        bplus_index, time_taken = bplustree_knn.create_btree_index_v2(dataset, order, k_list[0], reference_type=reference_type)
        idx_creation_time.append(time_taken)
        error, prediction_time = [], []
        for k in k_list:
            bplus_index.build_index_at_nodes(node_algorithm, k)
            bst_pred_res = bplustree_knn.predict(bplus_index, test_features, test_labels, k)
            error.append(round(bst_pred_res['error'], 4))
            prediction_time.append(round(bst_pred_res['time_taken'], 4))
        error_map[reference_type] = error
        prediction_time_map[reference_type] = prediction_time
    print(f'{dataset} error: {error_map}')
    print(f'{dataset} prediction_time: {prediction_time_map}')
    print(f'{dataset} idx_creation_time: {idx_creation_time}')

def experiment_memory(dataset, algorithm):
    train_features, test_features, train_labels, test_labels = data_preprocessing.read_dataset(dataset)
    train_features = np.ascontiguousarray(train_features, dtype=np.float32)
    k, precision = 25, 0.9
    order = int(0.05 * len(train_features))
    bplus_index, time_taken = bplustree_knn.create_btree_index_v2(dataset, order, k, reference_type='mean')
    # flann_index.create_index()
    return bplustree_knn.get_index_size(bplus_index, algorithm)

if __name__ == '__main__':
    # exp_order_of_tree_btree('covtype')
    # exp_order_of_tree_btree('poker')
    # exp_num_neighbors_btree('covtype', 'mprt')
    # exp_num_neighbors_btree('poker', 'mprt')
    # experiment_btree('covtype', algorithm='kd_tree')
    # exp_num_neighbors_btree('poker', 'ngt')
    # exp_num_neighbors_btree('mnsit', 'flann_kdtree')
    # exp_num_neighbors_btree('mnsit', 'ball_tree')
    # exp_num_neighbors_btree('census', 'ball_tree')
    # exp_num_neighbors_btree('swarm', 'mprt')
    # exp_num_neighbors_btree('swarm', 'pynndescent')
    datasets = ['census', 'covtype', 'poker', 'mnsit', 'swarm', 'gissete']
    # datasets = ['census']
    # for dataset in datasets:
    #     pair_index_algorithm = 'hnsw'
    #     print(f'running btree+{pair_index_algorithm} for {dataset}')
    #     exp_num_neighbors_btree(dataset, pair_index_algorithm)
    #     print("======================================================================================================")
    #     break

    # memory_used = []
    # for dataset in datasets:
    #     pair_algorithm = 'hnsw'
    #     print(f'running {pair_algorithm} for {dataset}')
    #     memory_used.append(experiment_memory(dataset, pair_algorithm))
    #     print("========================================================================")
    # print(f'algorithm: {pair_algorithm}, memory_used: {memory_used}')


    # for dataset in datasets:
    #     exp_reference_point(dataset)
    #     print("======================================================================================================")


    # datasets = ['census', 'covtype', 'poker', 'mnsit', 'swarm', 'gissete']
    # for algorithm in ['ball_tree', 'flann_kdtree', 'flann_kmeans', 'pynndescent', 'ngt']:
    for algorithm in ['hnsw']:
        auc_scores = []
        for dataset in datasets:
            print(f'running {algorithm} for {dataset}')
            auc_scores.append(experiment_roc_auc(dataset, algorithm))
            print("========================================================================")
        print(f'for bplus+{algorithm}, auc_scores: {auc_scores}')

    # for dataset in datasets:
    #     print(f'running order of tree experiment for {dataset}')
    #     exp_order_of_tree_btree(dataset)
    #     print("===========================================================================================================")