'''
ecg_realtime_abnormal_detection
Created 17/07/18 by Matthew Lee
'''
import os
import random
import numpy as np
from operator import itemgetter


from read_data import read_all_data
import config
import utils


@utils.timer()
def slice_annotations(file_dicts, sample_range=config.data['sample_range']):
    '''
    Loads in a dictionary object that contains MIT-BIH dictionary data (use read_all_data in read_data.py to obtain)
    if the annotation type is in the normal or abnormal list (found in config), slice the data based on the sample_range
    using the annotation sample as the center point.

    Use this to create a feature list and a label list and return both as numpy arrays.

    :param file_dicts: A dictionary of dictionarys referencing the MIT-BIH data. (use read_all_data in read_data.py to obtain)
    :param sample_range: The range that the data will be sliced with annotation sample as the center point
    :return: (numpy) features, (numpy) labels
    '''
    features, labels = [], []
    key_values = file_dicts.keys()
    # Key references a filename found in the dictionary
    for key in key_values:
        file_data = file_dicts[key]
        # Confirm that the desired lead (found in config) is present in this file
        if config.data['lead'] in file_data['fields']['sig_name']:
            lead_index = file_data['fields']['sig_name'].index(config.data['lead'])
            # Loop through the annotation samples and corresponding types
            for sample, type in zip(file_data['annotation'].sample, file_data['annotation'].symbol):
                labeled_data = False
                # Ensure we do not slice out of the bounds of the list
                if sample > sample_range/2 and sample + sample_range < len(file_data['signal']):
                    # Check that data is in the normal or abnormal annotations list
                    if type in config.data['normal_annotations']:
                        label = 0
                        labeled_data = True
                    elif type in config.data['abnormal_annotations']:
                        label = 1
                        labeled_data = True
                    # If the label type matches the lists, perform slice and add to features and labels
                    if labeled_data:
                        start_index = sample - int(sample_range/2)
                        feature = file_data['signal'][start_index:start_index + sample_range]
                        feature = [x[lead_index] for x in feature]
                        features.append(feature)
                        labels.append(label)
    return features, labels


@utils.timer()
def stratify_data(X, y):
    '''
    Converts two lists of equal size into strata based
    on the y list
    Note: This does not sample or balance the data. It simply groups
    the data based on the labels. See 'random_sample' for how to balance
    the data
    :param X: List of features
    :param y: List of labels (will be used to stratify)
    :return: feature_list (2D list [unique_labels, features])
             label_list (2D list [unique_labels, labels])
    '''
    unique_labels = list(set(y))
    feature_list = []
    label_list = []

    # Create subgroups
    for _ in unique_labels:
        label_list.append([])
        feature_list.append([])
    for i, feature in enumerate(X):
        index = unique_labels.index(y[i])
        label_list[index].append(y[i])
        feature_list[index].append(feature)

    return feature_list, label_list


@utils.timer()
def random_sample(strata_features, strata_labels, total=None):
    '''
    Converts unbalanced feature and label pairs to balanced based on
    the smallest category.
    :param strata_features: stratified features (see stratify_data)
    :param strata_labels: stratified labels (see stratify_data)
    :param total: the total sample to take. Defaults to smallest category
    :return:
    '''
    for y_list in strata_labels:
        if total is None or len(y_list) < total:
            total = len(y_list)

    sampled_features = []
    sampled_labels = []
    for X_list, y_list in zip(strata_features, strata_labels):
        # Generate indexes
        shuf_indexes = list(range(len(y_list)))
        # Shuffle indexes
        random.shuffle(shuf_indexes)
        # Sample indexes based on total
        shuf_indexes = shuf_indexes[:total]
        # Use indexes to get items from the two lists
        feature_list = list(itemgetter(*shuf_indexes)(X_list))
        label_list = list(itemgetter(*shuf_indexes)(y_list))
        # Append to output lists
        sampled_features.append(feature_list)
        sampled_labels.append(label_list)
    X, y = np.array(sampled_features), np.array(sampled_labels)
    X, y = X.reshape(X.shape[0] * X.shape[1], X.shape[2]), y.reshape(y.shape[0] * y.shape[1])
    return X, y


@utils.timer(verbose_only=True)
def one_hot_encode(x):
    '''
    Takes a list of label indexes and one-hot-encodes them
    :param x: e.g: [1, 0, 2]
    :return: [[0, 1, 0],
              [1, 0, 0],
              [0, 0, 1]]
    '''
    n_values = np.max(x) + 1
    return np.eye(n_values)[x]


@utils.timer()
def setup_data(name="temp", directory=config.data['npy_loc']):
    '''
    Loads data and saves it as two numpy arrays X and y
    :param name: The numpy arrays are saved as {name}_X and {name}_y
    :param directory: The directory to save the data
    :return: X, y
    '''
    data_dicts = read_all_data()
    X, y = slice_annotations(data_dicts)
    X, y = stratify_data(X, y)
    X, y = random_sample(X, y)
    y = one_hot_encode(y)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save("{}/{}_X.npy".format(directory, name), X)
    np.save("{}/{}_y.npy".format(directory, name), y)
    return X, y


@utils.timer(verbose_only=True)
def get_data(name="temp", directory=config.data['npy_loc']):
    '''
    Loads the numpy arrays saved
    :param name: The numpy arrays are saved as {name}_X and {name}_y
    :param directory: The directory to load the data from
    :return: X, y
    '''
    X = np.load("{}/{}_X.npy".format(directory, name))
    y = np.load("{}/{}_y.npy".format(directory, name))
    return X, y


if __name__ == "__main__":
    setup_data()