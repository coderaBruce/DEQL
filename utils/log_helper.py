"""
some of the code is taken from
https://github.com/LunaBlack/KGAT-pytorch/blob/master/utils/log_helper.py
"""
import os
import logging
import csv
from collections import OrderedDict
import yaml
import json
from os.path import exists

def create_save_checkpoint(name, dir_path):
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    checkpoint = os.path.join(dir_path, name + '.pt')
    return checkpoint

def create_log_id(name, dir_path):
    
    log_count = 0
    file_path = os.path.join(dir_path, name + '-log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, name + '-log{:d}.log'.format(log_count))
        
    return name + '-log{:d}'.format(log_count)


def configure_logger(config_dict):
    # Create a logger here
    folder = config_dict.get('log_dir', 'logs')
    name = config_dict.get('log_name', 'mylogger') + config_dict.get('pre_process')
    level = config_dict.get('log_level', logging.DEBUG)
    console_level = config_dict.get('console_log_level', logging.DEBUG)
    no_console = config_dict.get('no_console_log', False)

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create a logger instance
    logger = logging.getLogger(name)
    key = config_dict.get('loop', '')
    idx = '-' + str(config_dict.get(key, ''))
    # log_save_id = create_log_id(name + idx, folder)
    log_save_id = create_log_id(config_dict['log_name'] + '-' + config_dict['note'] + '-' + idx, config_dict['log_dir'])
    logpath = os.path.join(folder, log_save_id + ".log")
    print("All logs will be saved to %s" % logpath)

    # Remove any existing handlers from the logger
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # Set the logger's level
    logger.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)

    if not no_console:
        # Create a console handler
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logger.addHandler(logconsole)
    
    logger.propagate = False

    return logger, folder

def logger_dict(logger, dicts, indent='', log_level=logging.INFO):
    cur_indent = '' + indent
    for k, v in dicts.items():
        if isinstance(v, dict):
            logger.log(log_level, cur_indent + '|-' + k)
            logger_dict(logger, v, indent=cur_indent + '    ', log_level=log_level)
        else:
            logger.log(log_level, cur_indent + '|-{0} : {1}'.format(k, v))




def append_to_csv(file_name, data_dict):
    """
    Used to keep track of data as 
    Appends a row of data to a CSV file. Creates the file with headers if it doesn't exist.

    Parameters:
    file_name (str): The name of the CSV file.
    data_dict (dict): A dictionary containing the fieldnames as keys and corresponding data as values.

    # Example of how to use the function
    # data = {'t1': 1.5, 't2': 0.5, 'mu': 150.0, 'r20': 0.2516, 'r50': 0.3688, 'n20': 0.2061, 'n100': 0.2799}
    # append_to_csv('output_data.csv', data)
    """
    # Check if the file exists to decide whether to write headers
    file_exists = exists(file_name)

    # Open the file in append mode
    with open(file_name, 'a', newline='') as csvfile:
        # Use the keys of the data_dict as fieldnames
        headers = data_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        # Write the header only if the file did not exist
        if not file_exists:
            writer.writeheader()

        # Append the new row of data
        writer.writerow(data_dict)