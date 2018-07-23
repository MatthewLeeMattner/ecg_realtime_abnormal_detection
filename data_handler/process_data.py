'''
ecg_realtime_abnormal_detection
Created 23/07/18 by Matthew Lee
'''
import wfdb
import numpy as np

import config


def resample_signal(signals, annotations, current_hz=360, output_hz=config.data['hz']):
    '''
    Resamples the signals/annotations from current hz to output hz.
    This is useful for training models at different hz for different distributions of data
    :param signals: The original array of signals to resample
    :param annotations: The original array of annotations to resample
    :param current_hz: The current hz that the data represents
    :param output_hz: The hz that you'd like the data to represent
    :return: resampled_sig: The new signal array. resampled_ann: The new annotation array
    '''
    resampled_sig, loc = wfdb.processing.resample_sig(signals, current_hz, output_hz)
    resampled_ann = wfdb.processing.resample_ann(loc, annotations)
    return resampled_sig, resampled_ann


if __name__ == "__main__":
    x = np.arange(0, 1000)
    y = np.arange(0, 1000, 100)
    hz = 300
    hz_target = 150
    sig, ann = resample_signal(x, y, hz, hz_target)
    print(sig)
    print(y)
    print(ann)