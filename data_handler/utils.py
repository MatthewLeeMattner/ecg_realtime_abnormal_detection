'''
ecg_realtime_abnormal_detection
Created 17/07/18 by Matthew Lee
'''
import time

import config


'''
    Decorators
'''
def timer(timer_flag=config.code['timer'], verbose_only=False):
    '''
    This is a decorator that prints out the total time that a function takes to run.
    :param timer_flag: Only print or return time taken to run if flag is True
    :param verbose_only: Only run this timer if the verbose flag is True
    :return: output or tuple (output, time) if in testing mode
    '''
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            output = func(*args, **kwargs)
            end = time.time()
            if timer_flag:
                if verbose_only is False or (verbose_only is True and config.code['verbose']):
                    print("Function {0.__name__} ran for {1}".format(func, end - start))
                    if config.code['testing']:
                        return output, time
            return output
        return wrapper
    return decorator


def verbose_only(func):
    '''
    Decorator that only runs function if verbose flag is True
    :return: output of func (only if verbose flag is True
    '''
    def wrapper(*args, **kwargs):
        if config.code['verbose']:
            return func(*args, **kwargs)
    return wrapper


def warning_only(func):
    '''
    Decorator that only runs function if warning flag is True
    :return: output of func (only if warning flag is True
    '''
    def wrapper(*args, **kwargs):
        if config.code['warnings']:
            return func(*args, **kwargs)
    return wrapper

'''
    Logging
'''


def log(log_text):
    print(log_text)

@verbose_only
def v_log(log_text):
    log("V: " + log_text)

@warning_only
def w_log(log_text):
    log("W: " + log_text)

'''
    Helpers
'''


def find_index(haystack, needle):
    '''
    The index function for list objects returns a ValueError
    if the object is not in the list. This function simple changes that so that
    a False boolean value is returned instead. Python doesn't do -1 as a return
    like most languages because -1 is an index that can be used to access the last element
    '''
    try:
        index = haystack.index(needle)
    except ValueError:
        index = False
    return index