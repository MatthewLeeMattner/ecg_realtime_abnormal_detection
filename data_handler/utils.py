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


'''
    Logging
'''
def log(log_text):
    print(log_text)

@verbose_only
def v_log(log_text):
    log("V: " + log_text)