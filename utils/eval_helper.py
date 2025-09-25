import os
import shutil
import sys
import time
from copy import deepcopy

import numpy as np
from scipy import sparse
import pandas as pd
import bottleneck as bn
import json
from tqdm import tqdm


from torch.utils.data import Dataset, DataLoader
from world import logger
from utils.compute_helper import torch_partitionMM
import scipy.sparse as sp

from world import config_dict

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):

    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)  
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True  

    X_true_binary = (heldout_batch > 0).toarray() 
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32) 
    if config_dict['pre_process']=='strong_generalization':
        recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))  
    if config_dict['pre_process']=='leave_out':
        recall = tmp / X_true_binary.sum(axis=1)  
    return recall



def evaluate_close_form(model, config_dict, data_reader):
    logger.info("evaluating " + config_dict['model'] + " original-version " + "...")

    B = model.B
    test_data_tr, test_data_te = data_reader.load_tr_te_data1()
    N_test = test_data_tr.shape[0]
    idxlist_test = range(N_test)

    batch_size_test = 10000
    n20_list, n100_list, r20_list, r50_list = [], [], [], []


    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)): #bnum: batch number
        end_idx = min(st_idx + batch_size_test, N_test)
        X = test_data_tr[idxlist_test[st_idx:end_idx]]

        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')

        pred_val = torch_partitionMM(X, B)
        
        pred_val[X.nonzero()] = -np.inf
        r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
        r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))
        n20_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
        n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))

    n20_list = np.concatenate(n20_list)
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)


    r20_list = r20_list[~np.isnan(r20_list)]
    r50_list = r50_list[~np.isnan(r50_list)]
    n20_list = n20_list[~np.isnan(n20_list)]
    n100_list = n100_list[~np.isnan(n100_list)]



    if not (r20_list.shape[0] == r50_list.shape[0] == n100_list.shape[0] == n20_list.shape[0]):
        logger.error("r20, r50, n20, n100 list should have same length!")
        raise AssertionError("r20, r50, n20, n100 list should have same length!")

    
    logger.info("Test Recall@20=%.4f (%.4f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    logger.info("Test Recall@50=%.4f (%.4f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
    logger.info("Test NDCG@20=%.4f (%.4f)" % (np.mean(n20_list), np.std(n20_list) / np.sqrt(len(n20_list))))
    logger.info("Test NDCG@100=%.4f (%.4f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    
    return np.mean(r20_list), np.mean(r50_list), np.mean(n20_list), np.mean(n100_list)




