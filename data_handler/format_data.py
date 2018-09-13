'''
ecg_realtime_abnormal_detection
Created 17/07/18 by Matthew Lee
'''
import os
import random
import numpy as np
from operator import itemgetter
from sklearn.model_selection import train_test_split

from read_data import read_all_data
import process_data
import config
import utils


def slice_signal(signal, index, before=config.data['slice_before'], after=config.data['slice_after']):
    '''
    Returns a slice of the signal list based on the config
    data dict slice before and slice after
    :param signal: list of signal elements
    :param index: the index that the slice will occur at
    :param before: How many elements before index to slice (defaults to config.data['slice_before']
    :param after: How many elements after index to slice (defaults to config.data['slice_after']
    :return: The sliced signal list
    '''
    if index-before < 0:
        raise IndexError("Starting index for slice is out of bounds: Index {}".format(index))
    if index+after+1 > len(signal):
        raise IndexError("Ending index for slice is out of bounds: Index {} with a signal length {}".format(index, len(signal)))
    return signal[index-before:index+after+1]


@utils.timer()
def slice_based_on_annotations(file_dicts, annotations=config.data['annotations']):
    '''
    Loads in a dictionary object that contains MIT-BIH dictionary data (use read_all_data in read_data.py to obtain)
    if the annotation type is in the annotations list, slice the data using the slice_signal function

    Use this to create a feature list and a label list and return both as numpy arrays.

    :param file_dicts: A dictionary of dictionarys referencing the MIT-BIH data. (use read_all_data in read_data.py to obtain)
    :param annotations: A list of characters related to the annotations that will be used (the labels) https://www.physionet.org/physiobank/annotations.shtml
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
                # Label index is the index related to the annotations
                label_index = utils.find_index(annotations, type)
                if label_index is not False:
                    try:
                        if config.processing['difference'] and config.processing['average_difference']:
                            difference = np.array(slice_signal(file_data['difference'], sample))
                            average_difference = np.array(slice_signal(file_data['average_difference'], sample))
                            feature = np.reshape(np.concatenate((difference, average_difference), axis=0), (2, difference.shape[0])).T
                        elif config.processing['difference']:
                            feature = np.array(slice_signal(file_data['difference'], sample))
                        elif config.processing['average_difference']:
                            feature = np.array(slice_signal(file_data['average_difference'], sample))
                        else:
                            feature = slice_signal(file_data['signal'], sample)
                        features.append(feature)
                        labels.append(label_index)
                    except IndexError:
                        utils.w_log("Annotation {} has an index {} outside of signal range".format(type, sample))
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
    X, y = X.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3]), y.reshape(y.shape[0] * y.shape[1])
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
def setup_data(name=config.data['npy_name'], directory=config.data['npy_loc']):
    '''
    Loads data and saves it as two numpy arrays X and y
    :param name: The numpy arrays are saved as {name}_X and {name}_y
    :param directory: The directory to save the data
    :return: X, y
    '''
    data_dicts = read_all_data()
    key_values = data_dicts.keys()
    # Key references a filename found in the dictionary
    for key in key_values:
        signal = data_dicts[key]['signal']
        if config.processing['difference']:
            data_dicts[key]['difference'] = process_data.difference_signal(signal)
        if config.processing['average_difference']:
            data_dicts[key]['average_difference'] = process_data.difference_signal(
                process_data.average_signal(signal)
            )

    X, y = slice_based_on_annotations(data_dicts)
    X, y = stratify_data(X, y)
    X, y = random_sample(X, y)
    y = one_hot_encode(y)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save("{}/{}_X.npy".format(directory, name), X)
    np.save("{}/{}_y.npy".format(directory, name), y)
    return X, y


@utils.timer(verbose_only=True)
def get_data(name=config.data['npy_name'], directory=config.data['npy_loc']):
    '''
    Loads the numpy arrays saved
    :param name: The numpy arrays are saved as {name}_X and {name}_y
    :param directory: The directory to load the data from
    :return: X, y
    '''
    X = np.load("{}/{}_X.npy".format(directory, name))
    y = np.load("{}/{}_y.npy".format(directory, name))
    return X, y


def get_train_test(X, y, test_size=config.data['test_size']):
    '''
    Splits data into train and testing files based on test_size config
    :param X: The feature data
    :param y: The label data
    :param test_size: the float value of the test size between 0 and 1 (default config file)
    :return: (X_train, X_test, y_train, y_test)
    '''
    return train_test_split(X, y, test_size=test_size, stratify=y)


if __name__ == "__main__":
    setup_data()
    X, y = get_data()
    X_train, X_test, y_train, y_test = get_train_test(X, y)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)