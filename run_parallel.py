import argparse
import os
import numpy as np
import random
import pandas as pd
import sys
import itertools


import torch
import torch.nn as nn
import yaml

from utils.log_helper import *
from utils.eval_helper import *
from utils.data_helper import *
from models.model import *
from utils.train_helper import *

from world import *
import csv

def run(args, config_dict):
    """_summary_

    Args:
        args (_type_): _description_
        config_dict (_type_): _description_
    """

    # Create Saved model
    checkpoint = create_save_checkpoint('model_weights', config_dict['save_dir'])

    # Get the current process ID
    logger.info("Current Process ID: {}".format(os.getpid()))

    # Initialize Random Seed
    def set_seed(seed: int = 98765) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")
        
    set_seed(config_dict['seed'])  
    g = torch.Generator()
    g.manual_seed(0)
    
    
    data_reader = Data_Reader_close_form(config_dict)
    


    ml = eval(config_dict['model'])
    if not os.path.exists(config_dict['save_dir']):
        os.makedirs(config_dict['save_dir'])
   

    best_monitor = 0
    best_epoch = 0


    hyper_params = config_dict.get('hyper_params')
    combinations = list(itertools.product(*hyper_params.values()))

    r20_ll, r50_ll, n20_ll, n100_ll = [], [], [], []
    for i, combo in enumerate(combinations):
        hyper_param_combo_with_name = {list_name:item for item, list_name in zip(combo, hyper_params.keys())} 
        for hyper_param_val, hyper_param_name in zip(combo, hyper_params.keys()):
            config_dict[hyper_param_name] = hyper_param_val

        model = ml(config_dict, data_reader)

        r20, r50, n20, n100 = evaluate_close_form(model, config_dict, data_reader)
        r20_ll.append(r20)
        r50_ll.append(r50)
        n20_ll.append(n20)
        n100_ll.append(n100)



    
        logger.info(f'max r20 value is {max(r20_ll)}, it come from {r20_ll.index(max(r20_ll))}-th option')
        logger.info(f'max r50 value is {max(r50_ll)}, it come from {r50_ll.index(max(r50_ll))}-th option')
        logger.info(f'max n20 value is {max(n20_ll)}, it come from {n20_ll.index(max(n20_ll))}-th option')
        logger.info(f'max n100 value is {max(n100_ll)}, it come from {n100_ll.index(max(n100_ll))}-th option')

if __name__ == '__main__':
    
    run(args, config_dict)
    logger.info('All done!')

