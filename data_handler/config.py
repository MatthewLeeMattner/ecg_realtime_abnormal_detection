'''
ecg_realtime_abnormal_detection
Created 17/07/18 by Matthew Lee
'''

code = {
    'timer': True,
    'verbose': False,
    'testing': False,
    'warnings': True
}

train = {
    'name': 'standard-701',
    'epochs': 100,
    'batch': 32,
    'learning_rate': 0.01,
    'dropout': 0.9,
    'tensorboard': "../data/tensorboard"
}

data = {
    'mit-bih': "/media/matthewlee/DATA/data/MIT-BIH",
    'npy_loc': '../data',
    'npy_name': 'data',
    'hz': 900,
    'slice_before': 500,
    'slice_after': 200,
    'lead': 'MLII',
    'annotations': ['N', 'A'],
    'test_size': 0.1,
    'kernal_size': 5
}

processing = {
    'resample': False,
    'normalize': False,
    'vertical_center': False,
    'noise': False,
    'difference': True,
    'average_difference': True
}

network = {
    'feature_size': data['slice_before'] + data['slice_after'] + 1,
    'feature_channels': 2,
    'labels': 2
}