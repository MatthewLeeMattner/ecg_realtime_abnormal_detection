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
    'epochs': 10,
    'learning_rate': 0.002
}

data = {
    'mit-bih': "/media/matthewlee/DATA/data/MIT-BIH",
    'npy_loc': '../data',
    'npy_name': 'data',
    'hz': 260,
    'slice_before': 22,
    'slice_after': 33,
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