'''
ecg_realtime_abnormal_detection
Created 17/07/18 by Matthew Lee
'''
import numpy as np

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
        if config.data['lead'] in file_data['fields']['signame']:
            lead_index = file_data['fields']['signame'].index(config.data['lead'])
            # Loop through the annotation samples and corresponding types
            for sample, type in zip(file_data['annotation'].annsamp, file_data['annotation'].anntype):
                labeled_data = False
                # Ensure we do not slice out of the bounds of the list
                if sample > sample_range/2 and sample + sample_range < len(file_data['signal']):
                    # Check that data is in the normal or abnormal annotations list
                    if type in config.data['normal_annotations']:
                        label = [1, 0]
                        labeled_data = True
                    elif type in config.data['abnormal_annotations']:
                        label = [0, 1]
                        labeled_data = True
                    # If the label type matches the lists, perform slice and add to features and labels
                    if labeled_data:
                        start_index = sample - int(sample_range/2)
                        feature = file_data['signal'][start_index:start_index + sample_range]
                        feature = [x[lead_index] for x in feature]
                        features.append(feature)
                        labels.append(label)
    return np.array(features), np.array(labels)


X, y = slice_annotations(read_all_data())
print(X)
print(y)
print(X.shape)
print(y.shape)