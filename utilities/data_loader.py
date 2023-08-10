import numpy as np

from scipy import spatial
from utilities.data_preprocessing import read_dataset

train_data_map = {}
test_data_map = {}

train_ref_map, test_ref_map = {}, {}

def get_ref_distance(dataset, reference_point, index, type='train'):
    if dataset not in train_ref_map:
        train_x, test_x, train_y, test_y = read_dataset(dataset)
        mod_data = np.append(train_x, np.reshape(train_y, (len(train_y), 1)), axis=1)
        difference = mod_data - reference_point
        distances = np.linalg.norm(difference, 2, axis=1)
        # distances = []
        # for i in range(len(mod_data)):
        #     distances.append(spatial.distance.jaccard(mod_data[i], reference_point))
        # distances = []
        # for i in range(len(mod_data)):
        #     distances.append(spatial.distance.cosine(mod_data[i], reference_point))
        train_ref_map[dataset] = distances

    return train_ref_map[dataset][index].item() if type == 'train' else test_ref_map[dataset][index]

def get_unique_labels(dataset):
    train_x, test_x, train_y, test_y = read_dataset(dataset)
    return np.unique(train_y)

def get_data_point(dataset, index, type='train'):
    if dataset not in train_data_map:
        train_x, test_x, train_y, test_y = read_dataset(dataset)
        mod_data = np.append(train_x, np.reshape(train_y, (len(train_y), 1)), axis=1)
        train_data_map[dataset] = mod_data
        test_data_map[dataset] = test_x

    return train_data_map[dataset][index] if type == 'train' else test_data_map[dataset][index]
