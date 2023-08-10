import math

import numpy as np

from bplustree.node import Node
from c2lsh.c2lsh_impl import FaissLsh
from flann_impl.flann import FlannKnn
from hnsw_impl.hnsw_index import HnswKnn
from mprt import mprt
from ngt_impl.ngt_index import NGTKnn
from pynndescent_impl.pynndescent_index import PynnDescentKnn
from qalsh.qalsh_impl import QALsh
from sklearn_impl.knn import sklearn_knn
from utilities import commons, data_loader

class BPlusTree:
    def __init__(self, dataset, order, reference_point, k=10):
        self.root = Node(dataset, order, reference_point)
        self.root.check_leaf = True
        self.order = order
        self.reference_point = reference_point
        self.entropy = 0
        self.dataset = dataset
        self.nodes_visited = []
        self.all_nodes = None
        self.numNeighbors = k
        self.nodeIndexAlgorithm = 'ball_tree'

    # Insert operation
    def insert(self, key_index):
        key = data_loader.get_data_point(self.dataset, key_index)
        key_to_ref_dist = commons.euclidian(key, self.reference_point)
        # key_to_ref_dist = data_loader.get_ref_distance(self.dataset, self.reference_point, key_index)
        old_node = self.search(key_to_ref_dist)
        old_node.insert_at_leaf(key_index, key_to_ref_dist)
        if len(old_node.keys) > old_node.order:
            node1 = Node(self.dataset, old_node.order, self.reference_point)
            node1.check_leaf = True
            node1.parent = old_node.parent
            mid = int(math.ceil(old_node.order / 2)) - 1
            node1.keys, node1.key_distances, node1.pointers = old_node.keys[mid + 1:], old_node.key_distances[mid + 1:], old_node.pointers[mid + 1:]
            node1.next_key = old_node.next_key
            old_node.keys, old_node.key_distances, old_node.pointers = old_node.keys[:mid + 1], old_node.key_distances[:mid + 1], old_node.pointers[
                                                                                                                                  :mid + 1]
            old_node.next_key = node1
            self.insert_in_parent(old_node, node1.keys[0], node1.key_distances[0], node1)

    def get_closest(self, key_ref_dist, node):
        # node_keys, node_pointers = node.keys, node.pointers
        low, high = 0, len(node.keys) - 1
        lowDiff, highDiff = node.key_distances[low], node.key_distances[high]
        if key_ref_dist < lowDiff:
            return node.pointers[low]
        if key_ref_dist >= highDiff:
            return node.pointers[high + 1]

        while low <= high:
            mid = (low + high) // 2
            midDiff = node.key_distances[mid]
            if midDiff == key_ref_dist:
                return node.pointers[mid + 1]
            elif key_ref_dist < midDiff:
                high = mid - 1
            else:
                low = mid + 1
        lowDiff, highDiff = node.key_distances[low], node.key_distances[high]
        return node.pointers[low] if lowDiff < highDiff else node.pointers[high]

    def search(self, key_ref_dist):
        current_node = self.root
        while not current_node.check_leaf:
            current_node = self.get_closest(key_ref_dist, current_node)
        return current_node

    def find_nearest_neighbors(self, key):
        key = np.append(key, 0)
        key_ref_dist = commons.euclidian(key, self.reference_point)
        current_node = self.search(key_ref_dist)
        return current_node

    def get_starting_leaf_node(self):
        cur = self.root
        while not cur.check_leaf:
            cur = cur.pointers[0]
        return cur

    def get_all_nodes(self):
        all_nodes = []
        node = self.get_starting_leaf_node()
        while node is not None:
            all_nodes.append(node)
            node = node.next_key
        return all_nodes

    def build_index_at_nodes(self, algorithm, num_neighbors):
        node = self.get_starting_leaf_node()
        self.nodeIndexAlgorithm = algorithm
        while node is not None:
            data_points = data_loader.get_data_point(self.dataset, node.keys)
            features = data_points[:, :-1]
            labels = data_points[:, -1]
            node.unique_labels = np.unique(labels)
            if algorithm in ['kd_tree', 'ball_tree']:
                node.node_index = sklearn_knn(features, labels, algorithm='ball_tree')
                node.node_index.fit()
            elif algorithm == 'mprt':
                features = np.ascontiguousarray(features, dtype=np.float32)
                node.node_index = mprt.MprtIndex(features, labels, num_neighbors, recall=1.0)
                node.node_index.create_index()
            elif algorithm == 'pynndescent':
                node.node_index = PynnDescentKnn(features, labels)
                node.node_index.create_index()
            elif algorithm == 'ngt':
                node.node_index = NGTKnn(features, labels)
                node.node_index.create_index()
            elif 'flann' in algorithm:
                features = np.ascontiguousarray(features, dtype=np.float32)
                node.node_index = FlannKnn(features, labels, algorithm, precision=0.9)
                node.node_index.create_index()
            elif algorithm == 'qalsh':
                features = np.ascontiguousarray(features, dtype=np.float32)
                node.node_index = QALsh(self.dataset, features, labels)
                node.node_index.create_index()
            elif algorithm == 'faiss':
                features = np.ascontiguousarray(features, dtype=np.float32)
                node.node_index = FaissLsh(self.dataset, features, labels)
                node.node_index.create_index()
            elif algorithm == 'hnsw':
                features = np.ascontiguousarray(features, dtype=np.float32)
                node.node_index = HnswKnn(features, labels)
                node.node_index.create_index()
            else:
                raise Exception('unknown algorithm')
            node = node.next_key

    # Inserting at the parent
    def insert_in_parent(self, node, key, key_ref_dist, new_node):
        if self.root == node:
            rootNode = Node(self.dataset, node.order, self.reference_point)
            rootNode.keys, rootNode.key_distances, rootNode.pointers = [key], [key_ref_dist], [node, new_node]
            self.root = rootNode
            node.parent = rootNode
            new_node.parent = rootNode
            return

        parent_node = node.parent
        parent_pointers = parent_node.pointers
        for i in range(len(parent_pointers)):
            if parent_pointers[i] == node:
                # parent_node.keys = parent_node.keys[:i] + [key] + parent_node.keys[i:]
                parent_node.keys.insert(i, key)
                # parent_node.key_distances = parent_node.key_distances[:i] + [key_ref_dist] + parent_node.key_distances[i:]
                parent_node.key_distances.insert(i, key_ref_dist)
                # parent_node.pointers = parent_node.pointers[:i + 1] + [new_node] + parent_node.pointers[i + 1:]
                parent_node.pointers.insert(i+1, new_node)
                if len(parent_node.keys) > parent_node.order:
                    new_parent = Node(self.dataset, parent_node.order, self.reference_point)
                    new_parent.parent = parent_node.parent
                    mid = int(math.ceil(parent_node.order / 2)) - 1
                    new_parent.keys = parent_node.keys[mid + 1:]
                    new_parent.key_distances = parent_node.key_distances[mid + 1:]
                    new_parent.pointers = parent_node.pointers[mid + 1:]
                    key_ = parent_node.keys[mid]
                    key_dist = parent_node.key_distances[mid]
                    if mid == 0:
                        parent_node.keys = parent_node.keys[:mid + 1]
                        parent_node.key_distances = parent_node.key_distances[:mid + 1]
                    else:
                        parent_node.keys = parent_node.keys[:mid]
                        parent_node.key_distances = parent_node.key_distances[:mid]
                    parent_node.pointers = parent_node.pointers[:mid + 1]
                    for j in parent_node.pointers:
                        j.parent = parent_node
                    for j in new_parent.pointers:
                        j.parent = new_parent
                    self.insert_in_parent(parent_node, key_, key_dist, new_parent)
