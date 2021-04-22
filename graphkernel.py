import os
import numpy as np
import pandas as pd
from typing import List
import torch
from torch import Tensor, rand
# from torch.nn.parameter import Parameter
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from scipy.io import loadmat
from reference_alphabet import IUPAC_SEQ2IDX
from vae import VAE
from torch.nn.functional import one_hot


class KernelLoader:
    def __init__(self, sub_matrices: list = [], sub_mat_ids: list = [], vae: VAE = None):
        """
        Interface to MatrixKernel that encapsulates the collection of substitution matrices
        used. 
        Has list of kernels as class property.
        sub_mat_ids takes IDs from SubMat Matlab
        """
        if isinstance(vae, VAE):
            self.kernels: list = [VaeKernel(vae)]
            s_mat_id = ["VAE-kernel"]
        else:
            s_mat, s_mat_id = self.load_sub_matrices(sub_matrices, sub_mat_ids)
            self.kernels: list = [MatrixKernel(matrix=s, matrix_id=m_id)
                                  for s, m_id in zip(s_mat, s_mat_id)]
        self.sub_matrices_ids = s_mat_id
        assert len(self.kernels) == len(self.sub_matrices_ids)

    def load_sub_matrices(self, sub_matrices, sub_mat_ids, ):
        matrices = loadmat(f"./data/mgp/subMats.mat").get('subMats')
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
        return s_mat, s_mat_id

    @staticmethod
    def select_sub_matrices(matrix_id, matrix_info, sub_matrices, s_mat_ids) -> bool:
        if matrix_id in s_mat_ids:  # check with IDs
            return True
        elif any([bool(s in matrix_info) for s in sub_matrices]):
            return True  # check with info IDs
        else:
            False


class MatrixKernel:
    def __init__(self, matrix: np.array, matrix_id: str, depth: int = 1):
        """
        Matrix Kernel class takes substitution matrix with which to compute the kernel.
        Takes sequences and list of adjacencies over which to compute the kernel value.
        """
        self.depth: int = depth  # not used downstream
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
    def __init__(self, vae, alphabet=IUPAC_SEQ2IDX, block_size=1024) -> None:
        self.vae = vae
        self.latent_dim = vae.z_dim
        self.alphabet = alphabet
        self.latent_sample = torch.normal(0, 1, size=(block_size, self.latent_dim)).float()
        # importance sampling via p(z)
        self.p_z = Normal(loc=torch.zeros(1), scale=torch.ones(1)).log_prob(self.latent_sample).sum(1)
        self.n = block_size
        self.block = block_size
        #   assert (self.importance.shape == [n_samples, self.latent_dim])

    def p_x_not_i(self, x: np.ndarray, i: int) -> torch.Tensor:
        """
        Marginalize over latent representation with respect to residue at 
        position i. Σ over all residues.
        : input: sequence x, position i
        : return: likelihood ...
        """
        _x = x[:, i].copy()
        sum_p_x_not_i = np.zeros([x.shape[0], self.latent_dim])
        for aa in self.alphabet.values():  # marginalize over all possible residues
            x[:, i] = aa
            z_loc, z_scale = self.vae.encoder(torch.Tensor(x))
            z_dist = Normal(z_loc, z_scale)
            sum_p_x_not_i += torch.exp(torch.sum(z_dist.log_prob(x), axis=-1))
        x[:, i] = _x
        return sum_p_x_not_i

    def convert_one_hot(self, x):
        x = torch.Tensor(x).to(torch.int64)
        return one_hot(x, num_classes=self.vae.num_categories).to(torch.float)

    def p_x_i_x_not_i(self, x):
        oh_x = self.convert_one_hot(x)
        z_x_loc, z_x_scale = self.vae.encoder(oh_x)
        q_z_x = Normal(z_x_loc, z_x_scale).log_prob(self.latent_sample).sum(1)
        p_x_z = Categorical(self.vae.decoder(self.latent_sample).exp()).log_prob(torch.Tensor(x)).sum(1)
        return torch.sum((p_x_z * self.p_z) / q_z_x, axis=-1) / self.n

    def k_vec(self, x) -> torch.Tensor:
        """
        block-wise computation of k-vector
        """
        x = x.numpy() if isinstance(x, torch.Tensor) else x
        k_x = np.zeros([x.shape[0], 1])
        for n in range(0, int(np.ceil(x.shape[0] / self.block))):
            p = n * self.block
            q = min(p + self.block, x.shape[0])
            _x = x[p:q, :]
            k_x[p:q] = self.p_x_i_x_not_i(_x)
        return torch.Tensor(k_x)

    def k(self, x_p, x_q=None, adjacencies: List[tuple] = None) -> torch.Tensor:
        N = x_p.shape[0]
        k = torch.zeros([N, N])
        # log_p_x = self.vae.log_p(self.convert_one_hot(x_p).flatten())
        # log_p_y = log_p_x if x_q is None else self.vae.log_p(self.convert_one_hot(x_q).flatten())
        temp_k = np.zeros([N, N])
        neighborhoods = np.array([contact for _, contact in
                                  adjacencies]) if isinstance(adjacencies[0], tuple) else adjacencies
        neighborhood_iterator = tqdm(enumerate(neighborhoods))
        for idx, neighbors in neighborhood_iterator:
            temp_k.fill(0.)
            for n in neighbors:
                k_x_p = self.k_vec(x_p)
                k_x_q = k_x_p if x_q is None else self.k_vec(x_q)
                assert x_p.shape[1] == x_q.shape[1]
                temp_k += torch.sum(k_x_p, k_x_q) # - log_p_x - log_p_y # sum of log normalization by subtracting log
            k_x_p = self.k_vec(x_p, idx)
            k_x_q = k_x_p if x_q is None else self.k_vec(x_q, idx)
            temp_k *= torch.sum(k_x_p, k_x_q) 
            k += temp_k
            # TODO normalize by -log(p(x))-log(p(y))
        norm = np.sqrt(np.diag(k))[:, np.newaxis]
        k_hat = k / norm.dot(norm.T)
        return torch.Tensor(k_hat).type(torch.float64)
