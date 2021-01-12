import numpy as np
import pandas as pd
import torch
from torch import Tensor, rand
from torch.nn.parameter import Parameter
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from tqdm import tqdm
from scipy.io import loadmat
from data_utility import aa2index
from contact_mapper import ContactMapper  

class WeightedDecompositionKernel:
    def __init__(self, kernels: pd.DataFrame, t=0.0, w=None, γ=1.0):
        self.kernels = kernels
        self.kernel_dim = len(self.kernels.mat[0])
        self.w = Parameter(w)
        self.γ = γ
        self.t = t
    
    def K_ϕ(self) -> torch.Tensor:
        """
        Compute weighted Kernel Matrix Values
        """
        assert self.w.shape[0] == len(self.kernels)
        mwdk = torch.zeros([self.kernel_dim, self.kernel_dim])
        for i, mat in enumerate(self.kernels.mat):
            mwdk += self.w[i] * torch.Tensor(mat.to_numpy())**self.γ
        return mwdk


class KernelLoader:
    def __init__(self, sub_matrices: list=["BLOSUM62", "BLOSUM50", "BLOSUM45", "BLOSUM80"], sub_mat_ids:list = []):
        """
        Interface to MatrixKernel that encapsulates the collection of substitution matrices
        used. 
        Has list of kernels as class property.
        sub_mat_ids takes IDs from SubMat Matlab
        """
        matrices = loadmat("./data/subMats.mat").get('subMats')
        s_mat = []
        # check for provided sub_matrices in data subMat
        for m_vals, m_id, m_info in matrices:
            for s in sub_matrices:
                if m_id in sub_mat_ids or s in str(m_info):
                    s_mat.append(m_vals)
        self.kernels: list = [MatrixKernel(matrix=s, matrix_id=m_id) for s, m_id in zip(s_mat, sub_matrices)]
        self.sub_matrices_names: list = sub_matrices
        assert len(self.kernels) == len(self.sub_matrices_names)
    
class MatrixKernel:
    def __init__(self, matrix: np.array, matrix_id: str, depth: int=1):
        """
        Matrix Kernel class takes substitution matrix with which to compute the kernel.
        Takes sequences and list of adjacencies over which to compute the kernel value.
        """
        self.depth: int = depth # not used downstream
        self.matrix = matrix
        self.matrix_id = matrix_id

    def averaged_neighborhood(self, p_res, q_res, p_seq, q_seq, p_adj, q_adj) -> float:
        """computes sum over neighborhood (Eq. 7) for both residue chains"""
        # Point for Improvement: Eq. 7 assumes neighborhoods are equal
        # implication graph structure stays the same during mutations 
        # test residue is the correct in adjacency
        # TODO conflicting only working with 1 sequence - should work with both ??
        p, p_neighborhood = p_adj
        q, q_neighborhood = q_adj
        # assert(np.all(p_neighborhood == q_neighborhood))
        # assert(p_res == p and q_res == q)
        n_sum = np.sum([self.matrix[p_seq[n]][q_seq[n]] for n in p_neighborhood])
        return n_sum

    #def k(self, p_sequence, q_sequence, p_adjacency, q_adjacency) -> float:
    def k(self, sequences, adjacencies) -> float:
        """
        Eq. 7
        Compute kernel value w.r.t. neighborhood normalized.
        N = num of mutational variants
        D = sequence length
        Input: sequences: NxD, 
            adjacencies: NxD as list of AA integers
        p_adjacency, q_adjacency list of neighbors per position

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

            # TODO inperformant and deprecated
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
        print(k_hat)
        return k_hat

