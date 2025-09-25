import pandas as pd
from scipy import sparse
import numpy as np


import torch
from torch.utils.data import Dataset, DataLoader



class Data_Reader_close_form(object):
    
    def __init__(self, config_dict):
        
        self.config_dict = config_dict
        dataset = config_dict['dataset']
        method = config_dict['pre_process']
        path_dict = config_dict['dataset_config'][method][dataset]


        if config_dict['pre_process'] == 'leave_out':
            self.train_mat_path = path_dict['train']
            self.valid_mat_path = path_dict['valid']
            self.test_mat_path = path_dict['test']
            self.sid_path = path_dict['sid']

            unique_sid = list()
            with open(self.sid_path, 'r') as f:
                for line in f:
                    unique_sid.append(line.strip())

            self.n_items = len(unique_sid)

        if config_dict['pre_process'] == 'strong_generalization':
            self.train_mat_path = path_dict['train']
            self.valid_tr_mat_path = path_dict['valid_tr']
            self.valid_te_mat_path = path_dict['valid_te']
            self.test_tr_path = path_dict['test_tr']
            self.test_te_path = path_dict['test_te']
            self.sid_path = path_dict['sid']
            
            unique_sid = list()
            with open(self.sid_path, 'r') as f:
                for line in f:
                    unique_sid.append(line.strip())

            self.n_items = len(unique_sid)
    
    def load_tr_te_data1(self):

        if self.config_dict['pre_process']=='strong_generalization':
            def create_unique_uid_map(tp):

                unique_uids = pd.unique(tp['uid'])
                return dict((pid, i) for (i, pid) in enumerate(unique_uids))
            
            if self.config_dict['isValidationPhase']==True:
                if self.config_dict['dataset'] in {'ML20M', 'MSD', 'Netflix', 'ML100K'}:
                    tp_tr = pd.read_csv(self.valid_tr_mat_path)
                    tp_te = pd.read_csv(self.valid_te_mat_path)
                else:
                    tp_tr = pd.read_csv(self.test_tr_path)
                    tp_te = pd.read_csv(self.test_te_path)

            else:
                tp_tr = pd.read_csv(self.test_tr_path)
                tp_te = pd.read_csv(self.test_te_path)
            
            if self.config_dict['dataset'] in {'Netflix_Frequent'}: # frequent dataset会出现有的user 在te出现, 不在tr中很烦
                n_users = tp_tr.shape[0]
                rows_tr, cols_tr = tp_tr['uid'], tp_tr['sid']
                rows_te, cols_te = tp_te['uid'], tp_te['sid']


                data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                        (rows_tr, cols_tr)), dtype='float64', shape=(n_users, self.n_items))
                data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                        (rows_te, cols_te)), dtype='float64', shape=(n_users, self.n_items))
                return data_tr, data_te
                
            else:
                profile2id = create_unique_uid_map(tp_tr)
                n_users = len(profile2id)
                rows_tr, cols_tr = tp_tr['uid'].apply(lambda x: profile2id[x]), tp_tr['sid']
                rows_te, cols_te = tp_te['uid'].apply(lambda x: profile2id[x]), tp_te['sid']


                data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                        (rows_tr, cols_tr)), dtype='float64', shape=(n_users, self.n_items))
                data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                        (rows_te, cols_te)), dtype='float64', shape=(n_users, self.n_items))
                return data_tr, data_te
        
        elif self.config_dict['pre_process']=='leave_out':
            unique_sid = list()
            with open(self.sid_path) as f:
                for line in f:
                    unique_sid.append(line.strip())

            n_items = len(unique_sid)

            def load_train_data(csv_file):
                tp = pd.read_csv(csv_file)
                n_users = tp['uid'].max() + 1

                rows, cols = tp['uid'], tp['sid']
                data = sparse.csr_matrix((np.ones_like(rows),
                                            (rows, cols)), dtype='float64',
                                            shape=(n_users, n_items))  # csr matrix, entries are all one, indics are (row, cols)
                return data
            
            return load_train_data(self.train_mat_path), load_train_data(self.test_mat_path)
        


class Dataset_close_form(Dataset):

    def __init__(self, X, batch_size):
        self.X = X.todense()
        self.batch_size = batch_size

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        batch = self.X[idx:idx + 1]
        return torch.tensor(batch, dtype=torch.float32)