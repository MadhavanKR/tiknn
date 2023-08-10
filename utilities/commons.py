import sys

import numpy as np
import math

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

def pca(data_x, n_components=2):
    pca = PCA(n_components=n_components, random_state=1)
    transformed_data = pca.fit_transform(data_x)
    scaler = StandardScaler()
    return scaler.fit_transform(transformed_data)

def entropy(labels):
    labelCount, entropy = {}, 0
    totalLabelCount = len(labels)
    for label in labels:
        label = label.item()
        if label not in labelCount:
            labelCount[label] = 0
        labelCount[label] += 1
    for label in labelCount:
        prob = labelCount[label] / totalLabelCount
        entropy = -(prob * math.log2(prob))
    return entropy

def nearest_neighbors(query, data, k):
    difference = data - query
    distances = np.linalg.norm(difference, 2, axis=1)
    nearest_indices = np.argpartition(distances, k)
    return distances[nearest_indices[:k]], nearest_indices[:k]

def euclidian(x, y, feature_list=None):
    x, y = x[:len(x) - 1], y[:len(y) - 1]
    if feature_list is not None:
        x, y = x[feature_list], y[feature_list]
    euclid_dist = np.linalg.norm(x - y, 2)
    return euclid_dist.item()

def difference(x, y):
    return np.linalg.norm(x, 2) / np.linalg.norm(y, 2)

def calc_accuracy(actual, predictions):
    actual, predictions = np.asarray(actual), np.asarray(predictions)
    correct = 0
    for i in range(len(actual)):
        if actual[i].item() == predictions[i].item():
            correct += 1
    return correct / len(actual)

def calc_sens_spec(actual, predictions):
    tn, fp, fn, tp = confusion_matrix(actual, predictions).ravel()
    specificity = 1 - tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return sensitivity, specificity

def calc_roc_auc_score(test_labels, y_scores):
    test_labels, y_scores = np.asarray(test_labels), np.asarray(y_scores)
    fpr, tpr, threshold = roc_curve(test_labels, y_scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def binary_search(arr, val):
    low, high = 0, len(arr) - 1
    lowVal, highVal = arr[low], arr[high]
    if val < lowVal:
        return low
    if val >= highVal:
        return high
    while low <= high:
        mid = (low + high) // 2
        midDiff = arr[mid]
        if midDiff == val:
            return mid
        elif val < midDiff:
            high = mid - 1
        else:
            low = mid + 1
    lowVal, highVal = arr[low], arr[high]
    return low if lowVal < highVal else high

def get_size(obj, depth, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if depth >= 10:
        return size
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, depth + 1, seen) for v in obj.values()])
        size += sum([get_size(k, depth + 1, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, depth + 1, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, depth + 1, seen) for i in obj])
    return size
