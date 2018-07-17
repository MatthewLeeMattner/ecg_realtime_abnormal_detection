'''
ecg_realtime_abnormal_detection
Created 17/07/18 by Matthew Lee
'''
import wfdb
import os
import config
import utils


@utils.timer(verbose_only=True)
def read_data(filename, directory=config.data['location']):
    '''
    Gathers all the data for a .dat file
    :param filename: name of the file to load. Should just be base name (no extention)
    :param directory: the location of the data. Defaults to configuration location
    :return: dictionary object containing the following elements
        record: Information regarding the signal type
        annotation: The annotation of the signal
            annotation.annsamp[index] -> the sampling point
            annotation.anntype[index] -> the type of annotation
            see: https://www.physionet.org/physiobank/annotations.shtml
        signal: the entire signals. 2 dimentional array
            axis 0 -> each element contains the signals recorded at that time
            axis 1 -> each element is a signal from a different lead. Corresponds with fields['signame']
        fields: dict object {fs, units, signame, comments}.
            fs -> signal sampling rate. Also referred to as hertz
            units -> unknown
            signame -> the name of the lead used. Corresponds to signal axis 1
            comments -> general comments made by annotator
    '''
    utils.v_log("Reading data files related to {}.".format(filename))
    full_path = "{}/{}".format(directory, filename)
    record = wfdb.rdsamp(full_path)
    annotation = wfdb.rdann(full_path, 'atr')
    sig, fields = wfdb.srdsamp(full_path)
    data = {
        'record': record,
        'annotation': annotation,
        'signal': sig,
        'fields': fields
    }
    return data


@utils.timer()
def read_all_data(directory=config.data['location']):
    '''
    Loads all .dat files in a directory. Defaults to config directory
    :param directory: the location of the data. Defaults to configuration location
    :return:
    '''
    data_files = {}
    for file in os.listdir(directory):
        if file.endswith(".dat"):
            filename = os.path.splitext(file)[0]
            data_files[filename] = read_data(filename)
    return data_files

