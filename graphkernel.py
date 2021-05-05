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
    def __init__(self, vae, alphabet=IUPAC_SEQ2IDX, sample_size=100, block_size=1024, fixed_sample=False,
                 normalize_S=False, marginal_not_i=False) -> None:
        """
        Kernel, which derives likelihoods from provided VAE, given an AA alphabet
        fixed sampling sets latent samples to 1 - FOR TESTING AND DEBUG ONLY!

        To compute a Substitution Matrix equivalent, provide a sequence, or list of sequences for `S_mat_sequence`.
        """
        self.vae = vae
        self.latent_dim = vae.z_dim
        self.alphabet = alphabet
        self.sample_size = sample_size
        self.block = block_size
        self.latent_sample = torch.normal(0, 1, size=(sample_size, self.latent_dim)).float()
        self.s_min = 0.
        self.s_max = 0.
        self.normalize_S = normalize_S
        self.marginal_p_x_not_i = marginal_not_i
        if fixed_sample:
            self.latent_sample = torch.ones((sample_size, self.latent_dim)).float()
        self.p_z = Normal(loc=torch.zeros(vae.z_dim), scale=torch.ones(vae.z_dim)).log_prob(self.latent_sample).sum(1)

    def convert_one_hot(self, x):
        x = torch.Tensor(x).to(torch.int64)
        return one_hot(x, num_classes=self.vae.num_categories).to(torch.float)

    def compute_S_matrix(self, sequences, normalize=True):
        """
        Normalization as conducted in Eq. 8 of Jokinen et al paper.
        """
        AAs = 21
        N, L = sequences.shape
        s = torch.zeros([AAs])
        # marginalize over residues
        for idx, _ in tqdm(enumerate(range(L))):
            ll_S = np.exp(self.log_likelihood(sequences, idx))
            # marginal - sum probs per AA across sequences
            s += ll_S.sum(0)[:AAs]
        s = np.log(s[:, np.newaxis] @ s[:, np.newaxis].T)
        if normalize:
            s = (s-torch.min(s)+1)/(torch.max(s)-torch.min(s)+1)
        return s

    @torch.no_grad()
    def log_likelihood(self, x: torch.Tensor, i: int) -> np.array:
        """
        returns log likelihood of sequence at index i
        shape: N x 1
        internal shape: N sequences x n samples x AA dims
        """
        N, L = x.shape
        oh_x = self.convert_one_hot(x)
        # compute x and y likelihoods
        z_x_loc, z_x_scale = self.vae.encoder(oh_x)
        z = z_x_loc + self.latent_sample[:, np.newaxis] * torch.sqrt(z_x_scale)
        q_z_x = Normal(z_x_loc, z_x_scale).log_prob(z).sum(-1)
        p_z = torch.mean(self.p_z, axis=0)  # prior mean across samples
        # decoder can only evaluate one z at a time # TODO: refactor VAE to enable batched latent processing
        categoricals = [Categorical(self.vae.decoder(z_i).exp()) for z_i in z]
        p_x_z_vec = torch.stack([cat.probs.log() for cat in categoricals])
        p_x_z = torch.stack([cat.log_prob(torch.Tensor(x)) for cat in categoricals])
        if self.marginal_p_x_not_i:
            p_x_not_i = self.p_x_not_i_marginal(p_x_z, q_z_x, i, L)
        else:
            p_x_not_i = self.p_x_not_i_joint(p_x_z, q_z_x, i, L)
        ll_x_i_x_not_i = torch.mean(p_x_z_vec[:, :, i] + p_x_not_i[:, :, np.newaxis] + p_z - (q_z_x/L)[:, :, np.newaxis],
                                    axis=0)
        return ll_x_i_x_not_i

    @torch.no_grad()
    def p_x_not_i_joint(self, p_x_z: torch.Tensor, q_z_x: torch.Tensor, i: int, L: int) -> torch.Tensor:
        """
        As defined in equation, sum over log-likelihoods
        => p(X_1=x_1, X_2=x_2, ... ,X_L=x_L) not inluding X_i
        """
        p_x_not_i_lower_idx = torch.sum(p_x_z[:, :, :i]-(q_z_x/L)[:, :, np.newaxis], axis=-1)  # sum per sequence
        p_x_not_i_higher_idx = torch.sum(p_x_z[:, :, (i+1):]-(q_z_x/L)[:, :, np.newaxis], axis=-1)
        return p_x_not_i_lower_idx + p_x_not_i_higher_idx

    @torch.no_grad()
    def p_x_not_i_marginal(self, p_x_z: torch.Tensor, q_z_x: torch.Tensor, i: int, L: int) -> torch.Tensor:
        """
        Sum over probabilities, to treat sequence likelihood as marginal
        => sum_j p(X_j=x_j) with j!=i
        Attempt to fix very small values
        """
        p_x_not_i_lower_idx = torch.sum(torch.exp(p_x_z[:, :, :i]-(q_z_x/L)[:, :, np.newaxis]), axis=-1)  # sum per sequence
        p_x_not_i_higher_idx = torch.sum(torch.exp(p_x_z[:, :, (i+1):]-(q_z_x/L)[:, :, np.newaxis]), axis=-1)
        return torch.log(p_x_not_i_lower_idx + p_x_not_i_higher_idx)

    @torch.no_grad()
    def log_likelihood_idx(self, x: torch.Tensor, i: int):
        p_x = Categorical(self.vae.decoder(self.compute_encoder_dist(x).loc).exp()).log_prob(torch.Tensor(x))
        ll = self.log_likelihood(x, i)
        p_x_not_i = torch.log(torch.sum(torch.exp(ll), axis=-1))
        normalized_p_x_i_x_not_i = torch.diag(ll[:, x[:, i]]) - p_x[:, i] - p_x_not_i
        return normalized_p_x_i_x_not_i.detach().numpy()[:, np.newaxis]

    def k_vec(self, x, i) -> np.array:
        """
        block-wise computation of k-vector
        """
        k_x = np.zeros([x.shape[0], 1])
        for n in range(0, int(np.ceil(x.shape[0] / self.block))):
            p = n * self.block
            q = min(p + self.block, x.shape[0])
            _x = x[p:q, :]
            k_x[p:q] = self.log_likelihood_idx(_x, i)
        return k_x

    def compute_encoder_dist(self, x):
        z_loc, z_scale = self.vae.encoder(self.convert_one_hot(x))
        z_dist = Normal(z_loc, z_scale)
        return z_dist

    def compute_normalized_S(self, s):
        return (s-self.s_min+1)/(self.s_max-self.s_min+1)

    def set_min_max_S(self, x_p, x_q) -> None:
        assert x_p.shape == x_q.shape
        print("Computing min, max S-values...")
        for idx in range(x_p.shape[1]):
            s_mat = np.log(np.matmul(np.exp(self.k_vec(x_p, idx)), np.exp(self.k_vec(x_q, idx).T)))
            self.s_min = np.min(s_mat) if np.min(s_mat) <= self.s_min else self.s_min
            self.s_max = np.max(s_mat[s_mat!=np.inf]) if np.max(s_mat) >= self.s_max else self.s_max
        return

    def S_val(self, x_p, x_q, idx):
        s_val = np.log(np.matmul(np.exp(self.k_vec(x_p, idx)), np.exp(self.k_vec(x_q, idx).T)))
        return self.compute_normalized_S(s_val) if self.normalize_S else s_val

    @torch.no_grad()
    def k(self, x_p, x_q=None, adjacencies: List[tuple] = None, normalize_k=True, eigen=False) -> torch.Tensor:
        """
        Numerically stable implementation of the proposed kernel function.
        """
        # torch no-grad
        x_p = np.array(x_p)
        N = x_p.shape[0]
        k = np.zeros([N, N])
        eig_values = []
        x_q = x_p if x_q is None else x_q
        if self.s_min == self.s_max == 0. and self.normalize_S:
            self.set_min_max_S(x_p, x_q)
        neighborhoods = np.array([contact for _, contact in adjacencies]) if isinstance(adjacencies[0], tuple) \
            else adjacencies
        neighborhood_iterator = tqdm(enumerate(neighborhoods))
        temp_k = np.zeros([N, N])
        for idx, neighbors in neighborhood_iterator:
            temp_k.fill(0.)
            for n in neighbors:
                s_val = self.S_val(x_p, x_q, n)
                if eigen:
                    s_val[s_val == np.inf] = 0
                    eig_values.append(np.linalg.eigvals(s_val))
                temp_k += s_val
            s_val = self.S_val(x_p, x_q, idx)
            if eigen:
                s_val[s_val == np.inf] = 0
                eig_values.append(np.linalg.eigvals(s_val))
            temp_k *= s_val
            k += temp_k
        print("VECT KERNEL:")
        print(k)
        if not normalize_k:
            return torch.Tensor(k).to(torch.float64), eig_values
        norm = np.sqrt(np.diag(k))
        k_hat = k / norm.dot(norm.T)
        print("VECT NORMALIZED")
        print(k_hat)
        return torch.Tensor(k_hat).to(torch.float64), eig_values
