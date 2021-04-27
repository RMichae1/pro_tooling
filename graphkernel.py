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
    def __init__(self, vae, alphabet=IUPAC_SEQ2IDX, sample_size=10000, block_size=1024) -> None:
        self.vae = vae
        self.latent_dim = vae.z_dim
        self.alphabet = alphabet
        self.sample_size = sample_size
        self.n = None
        self.block = block_size
        self.latent_sample = torch.normal(0, 1, size=(sample_size, self.latent_dim)).float()
        self.p_z = Normal(loc=torch.zeros(1), scale=torch.ones(1)).log_prob(self.latent_sample).sum(1)

    def convert_one_hot(self, x):
        x = torch.Tensor(x).to(torch.int64)
        return one_hot(x, num_classes=self.vae.num_categories).to(torch.float)

    @torch.no_grad()
    def likelihood(self, x, i) -> np.array:
        N = x.shape[0]
        oh_x = self.convert_one_hot(x)
        z_x_loc, z_x_scale = self.vae.encoder(oh_x)
        q_z_x = Normal(z_x_loc, z_x_scale).log_prob(self.latent_sample[:N]).sum(1)
        p_x_z = Categorical(self.vae.decoder(self.latent_sample[:N]).exp()).log_prob(torch.Tensor(x))
        p_x_z_not_i = p_x_z[:, :i].sum(-1) + p_x_z[:, (i+1):].sum(-1)
        p_x_i_x_not_i = (1/self.n) * p_x_z[:, i] * (p_x_z_not_i * self.p_z[:N]) / q_z_x
        return np.array(p_x_i_x_not_i)[:, np.newaxis]

    def k_vec(self, x, i) -> np.array:
        """
        block-wise computation of k-vector
        """
        k_x = np.zeros([x.shape[0], 1])
        for n in range(0, int(np.ceil(x.shape[0] / self.block))):
            p = n * self.block
            q = min(p + self.block, x.shape[0])
            _x = x[p:q, :]
            k_x[p:q] = self.likelihood(_x, i)
        return k_x

    def compute_encoder_dist(self, x):
        z_loc, z_scale = self.vae.encoder(self.convert_one_hot(x))
        z_dist = Normal(z_loc, z_scale)
        return z_dist

    @torch.no_grad()
    def k(self, x_p, x_q=None, adjacencies: List[tuple] = None) -> torch.Tensor:
        # torch no-grad
        x_p = np.array(x_p)
        N = x_p.shape[0]
        self.n = N
        k = np.zeros([N, N])
        x_q = x_p if x_q is None else x_q
        z_x_dist = self.compute_encoder_dist(x_p)
        z_y_dist = self.compute_encoder_dist(x_q)
        # compute x and y likelihoods
        log_p_x = Categorical(self.vae.decoder(z_x_dist.loc).exp()).log_prob(torch.Tensor(x_p)).detach().numpy()
        log_p_y = Categorical(self.vae.decoder(z_y_dist.loc).exp()).log_prob(torch.Tensor(x_q)).detach().numpy()
        neighborhoods = np.array([contact for _, contact in adjacencies]) if isinstance(adjacencies[0], tuple) \
            else adjacencies
        neighborhood_iterator = tqdm(enumerate(neighborhoods))
        temp_k = np.zeros([N, N])
        for idx, neighbors in neighborhood_iterator:
            temp_k.fill(0.)
            for n in neighbors:
                k_x_p = self.k_vec(x_p, n)
                k_x_q = self.k_vec(x_q, n)
                assert x_p.shape[1] == x_q.shape[1]
                temp_k += (k_x_p - log_p_x[:, n]) + (k_x_q - log_p_y[:, n])  # subtract ll for normalization
            k_x_p = self.k_vec(x_p, idx)
            k_x_q = self.k_vec(x_q, idx)
            temp_k *= (k_x_p - log_p_x[:, idx]) + (k_x_q - log_p_y[:, idx])
            k += temp_k
        norm = np.sqrt(np.diag(k))
        k_hat = k / norm.dot(norm.T)
        return torch.Tensor(k_hat).to(torch.float64)
