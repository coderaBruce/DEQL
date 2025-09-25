import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pandas as pd
from copy import deepcopy

import numpy as np
from scipy import sparse
import pandas as pd
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import scipy
import scipy.sparse as sp
from sparsesvd import sparsesvd
import world
from world import *
from utils.compute_helper import torch_partitionMM
import os

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def compute(self):
        raise NotImplementedError
    
    


class EDLAE_b_geq_0(BasicModel):

    def __init__(self, config_dict, data_reader):

        super(EDLAE_b_geq_0, self).__init__()
        import copy

        
        a2 = config_dict['a']**2
        b2 = config_dict['b']**2
        p = config_dict['p']


        def load_train_data(csv_file):
            
            tp = pd.read_csv(csv_file)
            n_users = tp['uid'].max() + 1

            rows, cols = tp['uid'], tp['sid']
            data = sparse.csr_matrix((np.ones_like(rows),
                                    (rows, cols)), dtype='float64',
                                    shape=(n_users, data_reader.n_items)) # csr matrix, entries are all one, indics are (row, cols)
            return data
        
        train_data = load_train_data(data_reader.train_mat_path)


        RTR = np.asarray(train_data.T.dot(train_data).todense(), dtype = np.float32)  
        
        G0 = ((1 - p)**2 * p * a2 + (1 - p) ** 3 * b2) * np.ones(RTR.shape) 
        np.fill_diagonal(G0, ((1 - p) * p * a2 + (1 - p) ** 2 * b2))
        H0 = G0 * RTR
        H0_inv = np.linalg.inv(H0)
        self.B = np.zeros(RTR.shape)

        V = ((1 - p) * p * a2 + (1 - p) ** 2 * b2)* np.ones(RTR.shape)
        np.fill_diagonal(V, (1 - p) * b2)
        V = V * RTR

        E1 = ((1 - p) ** 2 * p * (b2 - a2))* np.ones(RTR.shape)
        np.fill_diagonal(E1, (1 - p) * p * (b2 - a2))
        E1 = E1 * RTR

        E2 = copy.deepcopy(E1)
        np.fill_diagonal(E2, 0)

        S1 = H0_inv @ V
        S2 = H0_inv @ E1

        for i in tqdm(range(0, RTR.shape[1])):
            s1 = S1[:, i]
            s2 = S2[:, i]
            s = s1 - 1 / (1 + s2[i]) * s1[i] * s2
            t = H0_inv[:, i] - H0_inv[i, i] / (1 + s2[i]) * s2
            self.B[:, i] = s - 1 / (1 + E2[:, i] @ t) * (E2[i, :] @ s) * t

    def compute(self, X):
        pass



class EDLAE_b_geq_0_L2const(BasicModel):

    def __init__(self, config_dict, data_reader):
        super(EDLAE_b_geq_0_L2const, self).__init__()
        import copy

        
        a2 = config_dict['a']**2
        b2 = config_dict['b']**2
        p = config_dict['p']
        L2_const = config_dict['L2const']

        def load_train_data(csv_file):
            
            tp = pd.read_csv(csv_file)
            n_users = tp['uid'].max() + 1

            rows, cols = tp['uid'], tp['sid']
            data = sparse.csr_matrix((np.ones_like(rows),
                                    (rows, cols)), dtype='float64',
                                    shape=(n_users, data_reader.n_items)) # csr matrix, entries are all one, indics are (row, cols)
            return data
        
        train_data = load_train_data(data_reader.train_mat_path)


        RTR = np.asarray(train_data.T.dot(train_data).todense(), dtype = np.float32)  
        
        G0 = ((1 - p)**2 * p * a2 + (1 - p) ** 3 * b2) * np.ones(RTR.shape) 
        np.fill_diagonal(G0, ((1 - p) * p * a2 + (1 - p) ** 2 * b2))
        
        H0 = G0 * RTR
        np.fill_diagonal(H0, np.diag(H0) + L2_const)
        H0_inv = np.linalg.inv(H0)
        
        self.B = np.zeros(RTR.shape)

        V = ((1 - p) * p * a2 + (1 - p) ** 2 * b2)* np.ones(RTR.shape)
        np.fill_diagonal(V, (1 - p) * b2)
        V = V * RTR

        E1 = ((1 - p) ** 2 * p * (b2 - a2))* np.ones(RTR.shape)
        np.fill_diagonal(E1, (1 - p) * p * (b2 - a2))
        E1 = E1 * RTR

        E2 = copy.deepcopy(E1)
        np.fill_diagonal(E2, 0)

        S1 = H0_inv @ V
        S2 = H0_inv @ E1

        for i in tqdm(range(0, RTR.shape[1])):

            s1 = S1[:, i]
            s2 = S2[:, i]
            s = s1 - 1 / (1 + s2[i]) * s1[i] * s2
            t = H0_inv[:, i] - H0_inv[i, i] / (1 + s2[i]) * s2
            self.B[:, i] = s - 1 / (1 + E2[:, i] @ t) * (E2[i, :] @ s) * t


    def compute(self, X):
        pass


class EDLAE_b_geq_0_diag0_L2const(BasicModel):

    def __init__(self, config_dict, data_reader):
        """
            Close-form solution, train at init
        """
        super(EDLAE_b_geq_0_diag0_L2const, self).__init__()
        import copy

        
        a2 = config_dict['a']**2
        b2 = config_dict['b']**2
        p = config_dict['p']
        L2_const = config_dict['L2const']


        def load_train_data(csv_file):
            
            tp = pd.read_csv(csv_file)
            n_users = tp['uid'].max() + 1

            rows, cols = tp['uid'], tp['sid']
            data = sparse.csr_matrix((np.ones_like(rows),
                                    (rows, cols)), dtype='float64',
                                    shape=(n_users, data_reader.n_items)) # csr matrix, entries are all one, indics are (row, cols)
            return data
        
        train_data = load_train_data(data_reader.train_mat_path)


        RTR = np.asarray(train_data.T.dot(train_data).todense(), dtype = np.float32)  
        
        G0 = ((1 - p)**2 * p * a2 + (1 - p) ** 3 * b2) * np.ones(RTR.shape) 
        np.fill_diagonal(G0, ((1 - p) * p * a2 + (1 - p) ** 2 * b2))
        # H0+L2_const
        H0 = G0 * RTR
        np.fill_diagonal(H0, np.diag(H0) + L2_const)

        H0_inv = np.linalg.inv(H0)
        self.B = np.zeros(RTR.shape)

        V = ((1 - p) * p * a2 + (1 - p) ** 2 * b2)* np.ones(RTR.shape)
        np.fill_diagonal(V, (1 - p) * b2)
        V = V * RTR

        E1 = ((1 - p) ** 2 * p * (b2 - a2))* np.ones(RTR.shape)
        np.fill_diagonal(E1, (1 - p) * p * (b2 - a2))
        E1 = E1 * RTR
        
        E2 = copy.deepcopy(E1)
        np.fill_diagonal(E2, 0)

        S1 = H0_inv @ V
        S2 = H0_inv @ E1


        for i in tqdm(range(0, RTR.shape[1])):

            s1 = S1[:, i]
            s2 = S2[:, i]
            sl = H0_inv[:, i]
            
            s = s1 - 1 / (1 + s2[i]) * s1[i] * s2
            s_corr = sl - 1 / (1 + s2[i]) * sl[i] * s2
            
            t = H0_inv[:, i] - H0_inv[i, i] / (1 + s2[i]) * s2
            
            self.B[:, i] = s - 1 / (1 + E2[:, i] @ t) * (E2[i, :] @ s) * t
            corr_term = s_corr - 1 / (1 + E2[:, i] @ t) * (E2[i, :] @ s_corr) * t
            corr_term *= (self.B[i, i]/corr_term[i])
            self.B[:, i] = self.B[:, i] - corr_term
        



    def compute(self, X):
        pass


