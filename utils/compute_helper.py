import numpy as np
import torch
import scipy.sparse as sp
from world import config_dict


def torchMM(X, BB, device = config_dict['device']):
    BB = torch.tensor(BB).to(torch.float32).to(device)
    XX = torch.tensor(X).to(torch.float32).to(device)
    BBB = (BB).clone().detach().to(device)
    Y = torch.matmul(XX, BBB)
    
    return Y.detach().cpu().numpy()


def torch_partitionMM(matrix_a, matrix_b, block_size=10000):

    if sp.issparse(matrix_a):
        matrix_a = matrix_a.todense()
    if sp.issparse(matrix_b):
        matrix_b = matrix_b.todense()
    
    rows_a, cols_a = matrix_a.shape
    rows_b, cols_b = matrix_b.shape
    
    if cols_a != rows_b:
        raise ValueError("Matrix dimensions are not compatible for multiplication.")

    result_matrix = np.zeros((rows_a, cols_b))

    for i in range(0, rows_a, block_size):
        for j in range(0, cols_b, block_size):
            for k in range(0, cols_a, block_size):
                block_a = matrix_a[i:i+block_size, k:k+block_size]
                block_b = matrix_b[k:k+block_size, j:j+block_size]

                result_matrix[i:i+block_size, j:j+block_size] += torchMM(block_a, block_b)
    return result_matrix