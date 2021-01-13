import numpy as np
import pandas as pd
from typing import List
import torch
from torch import Tensor, rand
#from torch.nn.parameter import Parameter
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from tqdm import tqdm
from scipy.io import loadmat
from utility import aa2index
from contact_mapper import ContactMapper  


class KernelLoader:
    def __init__(self, sub_matrices: list=["BLOSUM62", "BLOSUM50", "BLOSUM45", "BLOSUM80"], 
                        sub_mat_ids: list=[]):
        """
        Interface to MatrixKernel that encapsulates the collection of substitution matrices
        used. 
        Has list of kernels as class property.
        sub_mat_ids takes IDs from SubMat Matlab
        """
        matrices = loadmat("./data/subMats.mat").get('subMats')
        s_mat = []
        s_mat_id = []
        # check for provided sub_matrices in data subMat
        for m_vals, m_id, m_info in matrices:
            # TODO make sub-matrices selectable
            #for s in sub_matrices:
                #if m_id in sub_mat_ids or s in str(m_info):
            s_mat_id.append(m_id[0])
            s_mat.append(m_vals)
        self.kernels: list = [MatrixKernel(matrix=s, matrix_id=m_id) for s, m_id in zip(s_mat, s_mat_id)]
        self.sub_matrices_ids = s_mat_id
        assert len(self.kernels) == len(self.sub_matrices_ids)
    

class MatrixKernel:
    def __init__(self, matrix: np.array, matrix_id: str, depth: int=1):
        """
        Matrix Kernel class takes substitution matrix with which to compute the kernel.
        Takes sequences and list of adjacencies over which to compute the kernel value.
        """
        self.depth: int = depth # not used downstream
        self.matrix = matrix
        self.matrix_id = matrix_id

    def k(self, sequences: np.ndarray, adjacencies: List[tuple]) -> np.ndarray:
        """
        Eq. 7
        Compute kernel value w.r.t. neighborhood normalized.
        N = num of mutational variants
        D = sequence length
        Input: sequences: NxD, 
            adjacencies: NxD as list of tuples with residues and AA neighbors as integers

        return NxN Matrix 
        """
        N = sequences.shape[0]
        k = np.zeros([N, N])
        neighborhoods = np.array([contacts for res, contacts in adjacencies])
        neighborhood_iterator = tqdm(enumerate(neighborhoods))
        for idx, neighbors in neighborhood_iterator:
            neighborhood_iterator.set_description(f"Matrix: {self.matrix_id}")
            for contacts in neighbors:
                # WARN: assumption is that neighborhood does NOT change
                k += self.matrix[sequences[:, contacts], :][:, sequences[:, contacts]]
            k *= self.matrix[sequences[:, idx], :][:, sequences[:, idx]]
        norm = np.sqrt(np.diag(k))[:, np.newaxis]
        k_hat = k / norm.dot(norm.T)

            # TODO imperformant and deprecated
            # k_xy = self.matrix[x_idx][y_idx] * self.averaged_neighborhood(p_res=res_x, q_res=res_y, 
            #                 p_seq=p_sequence, q_seq=q_sequence, 
            #                 p_adj=p_adjacency[idx], q_adj=q_adjacency[idx])
            # TODO this is how normalization is described in the paper, but not how it is done in practice!
            # # compute for normalization
            # k_xx = self.matrix[x_idx][x_idx] * self.averaged_neighborhood(p_res=res_x, q_res=res_x, 
            #                     p_seq=p_sequence, q_seq=p_sequence,
            #                     p_adj=p_adjacency[idx], q_adj=p_adjacency[idx])
            # k_yy = self.matrix[y_idx][y_idx] * self.averaged_neighborhood(p_res=res_y, q_res=res_y, 
            #                     p_seq=q_sequence, q_seq=q_sequence, 
            #                     p_adj=q_adjacency[idx], q_adj=q_adjacency[idx])
            # normalized_k = k_xy / np.sqrt(k_xx*k_yy)
            # k += normalized_k
        return k_hat

