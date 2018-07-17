'''
ecg_realtime_abnormal_detection
Created 17/07/18 by Matthew Lee
'''

code = {
    'timer': True,
    'verbose': True,
    'testing': False
}

data = {
    'location': "/media/matthewlee/DATA/data/MIT-BIH",
    'sample_range': 900,
    'lead': 'MLII',
    'normal_annotations': ['N'],
    'abnormal_annotations': ['A']
}