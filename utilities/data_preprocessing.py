import os.path
import random

import numpy as np
import pandas as pd
import sklearn.model_selection
import skmultilearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# from skmultilearn.model_selection import iterative_train_test_split

random.seed(1)

data_base_dir = '/home/madhavan/learning/knn/data'

def read_dataset(dataset_name, as_numpy=True):
    if dataset_name == 'census':
        return read_census_data(as_numpy)
    elif dataset_name == 'aps':
        return read_aps_data(as_numpy)
    elif dataset_name == 'gissete':
        return read_gissete_data(as_numpy)
    elif dataset_name == 'swarm':
        return read_swarm_data(as_numpy)
    elif dataset_name == 'usps':
        return read_usps_data(as_numpy)
    elif dataset_name == 'A':
        return read_A_data(as_numpy)
    elif dataset_name == 'highdim':
        return read_highdim_data(as_numpy)
    elif dataset_name == 'mnsit':
        return read_mnsit_small(as_numpy)
    elif dataset_name == 'sift_small':
        return read_sift('small', as_numpy)
    elif dataset_name == 'sift_big':
        return read_sift('big', as_numpy)
    elif dataset_name == 'sample':
        return create_sample_data(n=10, d=6)
    elif dataset_name == 'covtype':
        return read_covtype_data(as_numpy)
    elif dataset_name == 'poker':
        return read_poker_data(as_numpy)
    else:
        raise Exception('unknown dataset: {}'.format(dataset_name))

def read_poker_data(as_numpy=True):
    """
        :param as_numpy:
        :return: train_features, test_features, train_labels, test_labels
        data from: https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/
    """
    filepath = '/home/madhavan/learning/approximate-knn/knn/data/poker-hand-testing.data'
    df = pd.read_csv(filepath, header=None)
    # df = df[df.columns[-1] < 7]
    df.drop(df.loc[df[df.columns[-1]] >= 4].index, inplace=True)
    labels = df[df.columns[-1]]
    features = df.iloc[:, :-1]
    scaler = MinMaxScaler()
    features = pd.DataFrame(scaler.fit_transform(features))
    return split_data_x(features, labels, as_numpy)

def read_covtype_data(as_numpy=True):
    """
        :param as_numpy:
        :return: train_features, test_features, train_labels, test_labels
        data from: https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/
    """
    filepath = '/home/madhavan/learning/approximate-knn/knn/data/covtype.csv'
    df = pd.read_csv(filepath)
    cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
    df = df[df.Cover_Type <= 2]
    unique_labels = pd.unique(df.Cover_Type)
    print(f'unique_labels: {unique_labels}')
    scaler = StandardScaler()
    for col in cols:
        df[[col]] = scaler.fit_transform(df[[col]])

    labels = df[df.columns[-1]]
    features = df.iloc[:, :-1]
    scaler = MinMaxScaler()
    features = pd.DataFrame(scaler.fit_transform(features))
    return split_data_x(features, labels, as_numpy)

def create_sample_data(n=5, d=2):
    """
    samples n data points od d dimensions from a multivariate guassian
    """
    np.random.seed(1)
    mean = [1.0 for i in range(d)]
    covariance = [[0.0] * d for i in range(d)]
    for i in range(d):
        covariance[i][i] = 1.0

    train_x = np.random.multivariate_normal(mean=mean, cov=covariance, size=n)
    train_y = np.asarray([0 if i % 2 == 0 else 1 for i in range(n)])
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)

    test_x = np.random.multivariate_normal(mean=mean, cov=covariance, size=2)
    test_y = np.asarray([0 if i % 2 == 0 else 1 for i in range(10)])
    scaler = StandardScaler()
    test_x = scaler.fit_transform(test_x)

    return train_x, test_x, train_y, test_y

def read_highdim_data(as_numpy=True):
    train_data_path = f'{data_base_dir}/high-dim-csv.csv'
    train_labels_path = '../data/labels-high-dim-csv.csv'
    df = pd.read_csv(train_data_path)
    labels = pd.read_csv(train_labels_path)
    df = df.fillna(method='ffill')
    features = df.dropna()
    return split_data(features, labels, as_numpy)

def read_A_data(as_numpy=True):
    train_data_path = '../data/A.csv'
    train_labels_path = '../data/labels-A.csv'
    df = pd.read_csv(train_data_path)
    labels = pd.read_csv(train_labels_path)
    df = df.fillna(method='ffill')
    features = df.dropna()
    return split_data(features, labels, as_numpy)

def read_usps_data(as_numpy=True):
    train_data_path = '../data/usps.csv'
    train_labels_path = '../data/labels-usps.csv'
    df = pd.read_csv(train_data_path)
    labels = pd.read_csv(train_labels_path)
    labels.replace(-1, 0, inplace=True)
    df = df.fillna(method='ffill')
    features = df.dropna()
    return split_data(features, labels, as_numpy)

def read_gissete_data(as_numpy=True):
    train_data_path = f'{data_base_dir}/gisette_train.data'
    train_labels_path = f'{data_base_dir}/gisette_train.labels'
    df = pd.read_csv(train_data_path, delim_whitespace=True)
    labels = pd.read_csv(train_labels_path)
    labels.replace(-1, 0, inplace=True)
    df = df.fillna(method='ffill')
    features = df.dropna()
    min_max_scaler = preprocessing.MinMaxScaler()
    # features = pd.DataFrame(min_max_scaler.fit_transform(features))
    # features = (features - features.mean()) / features.std()
    # scaler = StandardScaler()
    # features = pd.DataFrame(scaler.fit_transform(features))
    scaler = MinMaxScaler()
    features = pd.DataFrame(scaler.fit_transform(features))
    return split_data_x(features, labels, as_numpy)

def read_census_data(as_numpy=True):
    df = pd.read_csv(f'{data_base_dir}/adult.data', header=None,
                     names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

    def convert_to_numbers(dataframe, col_name):
        dataframe[col_name] = pd.Categorical(dataframe[col_name])
        dataframe[col_name] = dataframe[col_name].cat.codes

    categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
    for col in categorical_cols:
        convert_to_numbers(df, col)

    scaler = StandardScaler()
    train_col_scale = df[['age', 'fnlwgt', 'education-num', 'hours-per-week']]
    train_scaler_col = scaler.fit_transform(train_col_scale)
    train_scaler_col = pd.DataFrame(train_scaler_col, columns=train_col_scale.columns)
    df['age'] = train_scaler_col['age']
    df['fnlwgt'] = train_scaler_col['fnlwgt']
    df['education-num'] = train_scaler_col['education-num']
    df['hours-per-week'] = train_scaler_col['hours-per-week']
    df.drop('fnlwgt', axis=1, inplace=True)
    df.drop('education', axis=1, inplace=True)
    labels = pd.DataFrame(df, columns=['income'])
    labels.replace(-1, 0, inplace=True)
    scaler = MinMaxScaler()
    features = df.drop('income', axis=1)
    features = pd.DataFrame(scaler.fit_transform(features))
    return split_data_x(features, labels, as_numpy)

def read_aps_data(as_numpy=True):
    train_data_path = '../data/aps-failure.csv'
    df = pd.read_csv(train_data_path, na_values=['na'])

    def convert_to_numbers(dataframe, col_name):
        dataframe[col_name] = pd.Categorical(dataframe[col_name])
        dataframe[col_name] = dataframe[col_name].cat.codes

    categorical_cols = ['class']
    for col in categorical_cols:
        convert_to_numbers(df, col)

    df = df.fillna(method='ffill')
    df = df.dropna()
    labels = pd.DataFrame(df, columns=['class'])
    labels.replace(-1, 0, inplace=True)
    features = df.drop('class', axis=1)
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features))
    return split_data_x(features, labels, as_numpy)

def read_mnsit_small(as_numpy=True):
    df = pd.read_csv(f'{data_base_dir}/mnsit/train.csv')
    labels = pd.DataFrame(df, columns=['label'])
    features = df.drop('label', axis=1)
    # scaler = StandardScaler()
    # features = pd.DataFrame(scaler.fit_transform(features))
    # features = features / 255
    # features = features - features.mean()
    scaler = MinMaxScaler()
    features = pd.DataFrame(scaler.fit_transform(features))
    return split_data_x(features, labels, as_numpy)

def read_swarm_data(as_numpy=True):
    df = pd.read_csv(f'{data_base_dir}/swarm.csv')
    # df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    labels = pd.DataFrame(df, columns=['Swarm_Behaviour'])
    labels.replace(-1, 0, inplace=True)
    features = df.drop('Swarm_Behaviour', axis=1)
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    features = pd.DataFrame(scaler.fit_transform(features))
    return split_data_x(features, labels, as_numpy)

def read_sift(size, as_numpy=True):
    if size == 'small':
        learningFileName = './data/siftsmall/siftsmall_learn.fvecs'
        queryFileName = './data/siftsmall/siftsmall_query.fvecs'
    else:
        learningFileName = './data/sift/sift_learn.fvecs'
        queryFileName = './data/sift/sift_query.fvecs'

    trainingData = read_fvec_as_numpy(learningFileName)
    testingData = read_fvec_as_numpy(queryFileName)
    scaler = StandardScaler()
    trainingData = scaler.fit_transform(trainingData)
    scaler = StandardScaler()
    testingData = scaler.fit_transform(testingData)
    trainingFeatures, trainingLabels = trainingData[:, 0:128], trainingData[:, 128]
    testingFeatures, testingLabels = testingData[:, 0:128], testingData[:, 128]
    print(trainingFeatures.shape, trainingLabels.shape)
    print(testingFeatures.shape, testingLabels.shape)
    return trainingFeatures, testingFeatures, trainingLabels, testingLabels

def write_to_binary_file(np_arr, filename):
    if os.path.exists(filename):
        os.remove(filename)
    temp_file = 'tmp'
    if os.path.exists(temp_file):
        os.remove(temp_file)
    np.savetxt(temp_file, np_arr.astype(np.float32), delimiter=' ', fmt='%.3f', newline=' ')
    with open(temp_file, 'r') as txtfile:
        mytextstring = txtfile.read()

    binarytxt = str.encode(mytextstring.strip())
    # save the bytes object
    with open(filename, 'wb') as fbinary:
        fbinary.write(binarytxt)
    fbinary.close()

def generate_data_idistance(dataset):
    train_dest_filename = f'{data_base_dir}/idist/{dataset}_idist_train'
    test_dest_filename = f'{data_base_dir}/idist/{dataset}_idist_test'
    ref_filename = f'{data_base_dir}/idist/{dataset}_idist_ref'
    print(f'generating files {train_dest_filename}, and {test_dest_filename}')
    train_x, test_x, train_y, test_y = read_dataset(dataset)
    reference_points = train_x[:64, ]
    print(f'train.shape: {train_x.shape}, test.shape: {test_x.shape}, reference.shape: {reference_points.shape}')
    write_to_binary_file(train_x, train_dest_filename)
    write_to_binary_file(test_x, test_dest_filename)
    write_to_binary_file(reference_points, ref_filename)

def read_fvec_as_numpy(filename):
    fv = np.fromfile(filename, dtype=np.float32)
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, 1 + dim)
    return fv

def split_data(features, labels, as_numpy):
    random.seed(1)
    size = len(features)
    # test_indices = random.sample(range(size), max(int(0.1*size), 1000))
    test_indices = np.random.choice(list(features.index), 1000)
    test_features, test_labels = features.iloc[test_indices], labels.iloc[test_indices]
    train_features, train_labels = features.drop(test_indices), labels.drop(test_indices)
    if as_numpy:
        return train_features.to_numpy(), test_features.to_numpy(), train_labels.to_numpy(), test_labels.to_numpy()
    else:
        return features, test_features, labels, test_labels

def split_data_d(features, labels, as_numpy):
    random.seed(1)
    size = len(features)
    o_size = int(1000 * 0.3)
    o_indices = []
    count = 0
    for i in range(size):
        label = labels.iloc[[i]].to_numpy()[0].item()
        if label == 1:
            o_indices.append(i)
            count += 1
        if count >= o_size:
            break

    test_indices = random.sample(range(size), 100)
    test_indices = test_indices[count:] + o_indices
    test_features, test_labels = features.iloc[test_indices], labels.iloc[test_indices]
    # train_features, train_labels = features.drop(test_indices), labels.drop(test_indices)
    train_features, train_labels = features, labels
    # print(test_labels)
    if as_numpy:
        return train_features.to_numpy(), test_features.to_numpy(), train_labels.to_numpy(), test_labels.to_numpy()
    else:
        return features, test_features, labels, test_labels

def split_data_x(features, labels, as_numpy):
    test_size = 100 / len(features)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=1)
    if as_numpy:
        return train_features.to_numpy(), test_features.to_numpy(), train_labels.to_numpy(), test_labels.to_numpy()
    else:
        return features, test_features, labels, test_labels
