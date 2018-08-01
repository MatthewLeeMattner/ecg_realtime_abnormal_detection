'''
ecg_realtime_abnormal_detection
Created 17/07/18 by Matthew Lee
'''
import wfdb
import os
import config
import utils
import process_data



@utils.timer(verbose_only=True)
def read_data(filename, directory=config.data['mit-bih']):
    '''
    Gathers all the data for a .dat file
    :param filename: name of the file to load. Should just be base name (no extention)
    :param directory: the location of the data. Defaults to configuration mit-bih
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
    annotation = wfdb.rdann(full_path, 'atr')
    sig, fields = wfdb.rdsamp(full_path)
    data = {
        'annotation': annotation,
        'signal': sig,
        'fields': fields
    }
    data['signal'] = get_lead(data, lead=config.data['lead'])
    return data



@utils.timer()
def read_all_data(directory=config.data['mit-bih']):
    '''
    Loads all .dat files in a directory. Defaults to config directory
    :param directory: the location of the data. Defaults to configuration mit-bih
    :return:
    '''
    data_files = {}
    for file in os.listdir(directory):
        if file.endswith(".dat"):
            filename = os.path.splitext(file)[0]
            try:
                data_files[filename] = read_data(filename)
            except ValueError:
                utils.w_log("Lead {} not found in data file {}".format(config.data['lead'], filename))
    return data_files


@utils.timer(verbose_only=True)
def get_lead(data_dict, lead=config.data['lead']):
    '''
    :param data_dict: The dictionary of the data (see read data)
    :param lead: The lead to find (defaults to config.data['lead'])
    :return: The signal of just the lead specified else a value error
    '''
    if lead in data_dict['fields']['sig_name']:
        lead_index = data_dict['fields']['sig_name'].index(lead)
        return [x[lead_index] for x in data_dict['signal']]
    else:
        raise ValueError("Lead {} not found in data files signal".format(lead))

