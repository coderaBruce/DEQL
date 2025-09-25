import argparse
import os
import numpy as np
import random
import pandas as pd

import torch
import torch.nn as nn
import yaml

from utils.log_helper import *

itemId='sid'  
userId='uid'


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='./')
parser.add_argument('--gpu', type = int, default = '0')
parser.add_argument('--key', type = float, default = 1)
parser.add_argument('--t', type = float, default = 0.0)
parser.add_argument('--mu', type = float, default = 1.0)
parser.add_argument('--note', type=str, default='')


args = parser.parse_args()

with open(args.config_path, 'r') as stream:
    config_dict = yaml.safe_load(stream)
for arg, value in vars(args).items():
    config_dict[arg]= value

key = config_dict['loop']
config_dict[key] = args.key
config_dict['arg'] = args.key 

config_dict['hyper_params'][key] = [args.key] 
config_dict.update(vars(args)) 
device = torch.device("cuda:%d"%(args.gpu) if torch.cuda.is_available() else "cpu")
config_dict['device'] = device

config_dict['seed'] = int(config_dict['seed'])

logger, _ = configure_logger(config_dict)
logger.info(args)
logger_dict(logger, config_dict)
