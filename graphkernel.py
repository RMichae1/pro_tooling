import os
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
from reference_alphabet import IUPAC_SEQ2IDX


class KernelLoader:
    def __init__(self, sub_matrices: list=[], sub_mat_ids: list=[]):
        """
        Interface to MatrixKernel that encapsulates the collection of substitution matrices
        used. 
        Has list of kernels as class property.
        sub_mat_ids takes IDs from SubMat Matlab
        """
        matrices = loadmat(f"{os.path.dirname(__file__)}/data/mgp/subMats.mat").get('subMats')
        s_mat = []
        s_mat_id = []
        # check for provided sub_matrices in data subMat
        for m_vals, m_id, m_info in matrices:
            if sub_matrices or sub_mat_ids:
                if self.select_sub_matrices(m_id[0], m_info[0], sub_matrices, sub_mat_ids):
                    s_mat_id.append(m_id[0])
                    s_mat.append(m_vals)
                else:
                    continue
            else:
                s_mat_id.append(m_id[0])
                s_mat.append(m_vals)
        self.kernels: list = [MatrixKernel(matrix=s, matrix_id=m_id) for s, m_id in zip(s_mat, s_mat_id)]
        self.sub_matrices_ids = s_mat_id
        assert len(self.kernels) == len(self.sub_matrices_ids)
    
    @staticmethod
    def select_sub_matrices(matrix_id, matrix_info, sub_matrices, s_mat_ids) -> bool:
        if matrix_id in s_mat_ids: # check with IDs
            return True
        elif any([bool(s in matrix_info) for s in sub_matrices]):
            return True # check with info IDs
        else:
            False
    

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
        temp_k = np.zeros([N, N])
        neighborhoods = adjacencies
        if isinstance(adjacencies[0], tuple):
            neighborhoods = np.array([contacts for res, contacts in adjacencies])
        neighborhood_iterator = tqdm(enumerate(neighborhoods))
        for idx, neighbors in neighborhood_iterator:
            neighborhood_iterator.set_description(f"Matrix: {self.matrix_id}")
            temp_k.fill(0.)
            for n in neighbors:
                # WARN: assumption is that neighborhood does NOT change
                temp_k += self.matrix[sequences[:, n], :][:, sequences[:, n]]
            temp_k *= self.matrix[sequences[:, idx], :][:, sequences[:, idx]]
            k += temp_k
        norm = np.sqrt(np.diag(k))[:, np.newaxis]
        k_hat = k / norm.dot(norm.T)
        return torch.Tensor(k_hat).type(torch.float64)


class VaeKernel:
    def __init__(self, vae, alphabet=IUPAC_SEQ2IDX, n_samples=100, block_size=1024) -> None:
        self.vae = vae
        self.latent_dim = vae.z_dim
        self.alphabet = alphabet
        self.latent_random = torch.normal(0, 1, size=(n_samples, 
                                        self.latent_dim)).float()
        self.importance = Normal(loc=torch.zeros(1), 
                                scale=torch.ones(1)).log_prob(self.latent_random)
        self.p_x_z = self.vae.decoder(self.latent_random).detach().numpy()
        self.block = block_size
        assert(type(self.p_x_z) == np.ndarray)
        assert(self.importance.shape == [n_samples, self.latent_dim])

    def p_i(self, x: np.ndarray, i: int) -> torch.Tensor:
        """
        Marginalize over latent representationm. Σ over all residues.
        : input: sequence x, position i
        : return: likelihood ...
        """
        _x = x[:, i].copy()
        p_z_i = np.zeros([x.shape[0], self.latent_dim])
        for aa in self.alphabet.keys():
            x[:, i] = aa
            z_loc, z_scale = self.vae.encoder(x)
            z_dist = Normal(z_loc, z_scale)
            # TODO investigate Importance Sampling
            p_z_i += torch.exp(torch.sum(z_dist.log_prob(self.latent_random) - self.importance, axis=-1))
        x[:, i] = _x
        p_x_i_z = self.p_x_z[:, i, _x].reshape(1, -1)
        p = torch.sum(p_x_i_z*p_z_i, axis=-1)
        # TODO assert and test stepwise
        return p

    def k_vec(self, x, i) -> torch.Tensor:
        k_x = np.zeros([x.shape[0], 1])
        for n in range(0, int(np.ceil(x.shape[0] / self.block))):
            p = n * self.block
            q = min(p + self.block, x.shape[0])
            _x = x.numpy()[p:q, :]
            k_x[p:q] = self.p_i(_x, i)
        return torch.Tensor(k_x)
    
    def k(self, x_p, x_q=None) -> torch.Tensor:
        N = x_p.shape[0]
        k = torch.zeros([N, N])
        assert x_p.shape[1] == x_q.shape[1]
        for i in range(0, x_p.shape[1]):
            k_x_p = self.k_vec(x_p, i)
            if x_q is None:
                k_x_q = k_x_p
            else:
                k_x_q = self.k_vec(x_q, i)
            k += torch.matmul(k_x_p, k_x_q.T)
            # TODO test p.s.d.
        return k

    # def k(self, sequences: np.ndarray, adjacencies: List[tuple]) -> np.ndarray:
    #     N = sequences.shape[0]
    #     k = np.zeros([N, N])
    #     temp_k = np.zeros([N, N])
    #     neighborhoods = adjacencies
    #     if isinstance(adjacencies[0], tuple):
    #         neighborhoods = np.array([contacts for _, contacts in adjacencies])

    #     for idx, neighbors in neighborhoods:
    #         temp_k.fill(0.)
    #         for n in neighbors:
    #             p_x = self.vae.encoder(x)
    #             p_y = self.vae.encoder(y)
    #             temp_k += (self.p_i(x, n)*self.p_i(y, n)/(p_x*p_y))
    #         p_x = self.vae.encoder(x)
    #         p_y = self.vae.encoder(y)
    #         temp_k *= (self.p_i(x, idx)*self.p_i(y, idx)/(p_x*p_y))
    #         k += temp_k

    #     # TODO compute p_x_i
    #     # TODO compute p(x)
    #     # TODO compute p(y)

    #     norm = np.sqrt(np.diag(k))[:, np.newaxis]
    #     k_hat = k / norm.dot(norm.T)
    #     return torch.Tensor(k_hat).type(torch.float64)

