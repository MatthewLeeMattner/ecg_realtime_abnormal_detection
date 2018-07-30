'''
ecg_realtime_abnormal_detection
Created 23/07/18 by Matthew Lee
'''
import wfdb
import numpy as np
import math

import config
import utils


class ProcessingPipeline:
    operations = []

    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def add_op(self, func, *args, **kwargs):
        '''
        Adds the operation to the operations list in the form
        of a dictionary object
        :param func: The function to be called
        :param args: The arguments to pass
        :param kwargs: The keyword arguments to pass
        '''
        self.operations.append({
            'function': func,
            'args': args,
            'kwargs': kwargs
        })

    def get_batch(self):
        '''
        Retrieves a batch of data and formats it based on the pipeline operations
        :yield: batch_X, batch_y
        '''
        for i in range(int(math.ceil(len(self.X)/self.batch_size))):
            try:
                batch_X = self.X[i*self.batch_size:i*self.batch_size+self.batch_size]
                batch_y = self.y[i*self.batch_size:i*self.batch_size+self.batch_size]
            except IndexError:
                batch_X = self.X[i*self.batch_size:]
                batch_y = self.y[i*self.batch_size:]
            for op in self.operations:
                temp_batch = []
                for sig in batch_X:
                    temp_batch.append(op['function'](sig))
                batch_X = np.array(temp_batch)
            yield batch_X, batch_y


@utils.timer(verbose_only=True)
def resample_signal(signal, current_hz=360, output_hz=config.data['hz']):
    '''
    Resamples the signals/annotations from current hz to output hz.
    This is useful for training models at different hz for different distributions of data
    :param signals: The original array of signals to resample
    :param annotations: The original array of annotations to resample
    :param current_hz: The current hz that the data represents
    :param output_hz: The hz that you'd like the data to represent
    :return: resampled_sig: The new signal array. resampled_ann: The new annotation array
    '''
    sig, loc = wfdb.processing.resample_sig(signal, current_hz, output_hz)
    return sig


@utils.timer(verbose_only=True)
def vertically_center_signal(signal):
    mean = np.mean(signal)
    return signal - mean


@utils.timer(verbose_only=True)
def normalize_signal(signal):
    max_val = max(signal)
    min_val = min(signal)
    return (signal - min_val)/(max_val - min_val)


@utils.timer(verbose_only=True)
def gaussian_noise(signal):
    noise = np.random.normal(0, 0.01, len(signal))
    return signal + noise


@utils.timer(verbose_only=True)
def expand_dims(signal):
    return np.expand_dims(signal, axis=2)


@utils.timer(verbose_only=True)
def difference_signal(signal):
    return [n - signal[i-1] for i, n in enumerate(signal) if i > 0]


@utils.timer(verbose_only=True)
def average_signal(signal, kernal_size):
    signal_avg = []
    for i in range(len(signal)):
        if i + kernal_size + kernal_size < len(signal):
            kernal_sum = sum(signal[i+kernal_size:i+kernal_size+kernal_size])
            kernal_avg = kernal_sum / kernal_size
            signal_avg.append(kernal_avg)
    return signal_avg

if __name__ == "__main__":
    import read_data
    import matplotlib.pyplot as plt

    data = read_data.read_data("100")
    x = [x[0] for x in data['signal'][:1000]]

    x_dif = difference_signal(x)
    x_avg_dif = difference_signal(average_signal(x, 5))
    index_max = len(x_avg_dif)
    plt.plot(x[:index_max])
    plt.plot(x_dif[:index_max])
    plt.plot(x_avg_dif[:index_max])
    plt.show()

    print(x)