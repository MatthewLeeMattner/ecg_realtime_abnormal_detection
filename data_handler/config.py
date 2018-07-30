'''
ecg_realtime_abnormal_detection
Created 17/07/18 by Matthew Lee
'''

code = {
    'timer': False,
    'verbose': False,
    'testing': False
}

data = {
    'mit-bih': "/media/matthewlee/DATA/data/MIT-BIH",
    'npy_loc': '../data',
    'sample_range': 900,
    'hz': 260,
    'lead': 'MLII',
    'normal_annotations': ['N'],
    'abnormal_annotations': ['A']
}

processing = {
    'resample': False,
    'normalize': True,
    'vertical_center': True,
    'noise': False
}