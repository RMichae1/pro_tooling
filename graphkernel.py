import numpy as np
import torch
from torch import Tensor, rand
from torch.nn.parameter import Parameter
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from Bio.Align import substitution_matrices
from Bio.Align.substitution_matrices import Array as SArray

from contact_mapper import ContactMapper  

class WeightedDecompositionKernel:
    def __init__(self, kernels, t=0.0, w=0.0, γ=1.0):
        self.kernels = kernels
        self.w = Parameter(rand(len(self.kernels.keys()))) if not w else w
        assert len(self.w) == len(self.kernels.keys())
        self.γ = γ
        self.kernel_parameters: dict = {"t": t, "w": self.w, "γ": self.γ}
        self.K_ϕ = self.compute_MKL()

    def compute_MKL(self) -> Tensor:
        K_ = Tensor([0.])
        # weighted kernel by n of internal matrices
        for i, m in enumerate(self.kernels.keys()):
            K_ += self.w[i] * self.kernels[m]**self.γ
        return K_

class KernelFactory:
    def __init__(self, sub_matrices=["BLOSUM62", "BLOSUM50", "BLOSUM45", "BLOSUM80"]):
        """
        Interface to MatrixKernel that encapsulates the collection of substitution matrices
        used. 
        Has list of kernels as class property.
        """
        self.elements = len(sub_matrices)
        self.sub_matrices = sub_matrices
        self.kernels = [self._construct_kernel(mat) for mat in self.sub_matrices]

    def _construct_kernel(self, matrix):
        return MatrixKernel(sub_matrix=matrix)
    
    
class MatrixKernel:
    def __init__(self,  sub_matrix: str, p_sequence: list=None, p_adjacency:tuple=None,
        q_sequence: list=None, q_adjacency: tuple=None, depth: int=1):
        """
        Matrix Kernel class takes substitution matrix with which to compute the kernel.
        Takes sequences and list of adjacencies over which to compute the kernel value.
        """
        self.p_sequence = p_sequence
        self.p_adjacency = p_adjacency
        self.q_sequence = q_sequence
        self.q_adjacency = q_adjacency
        self.depth: int = depth # not used downstream
        self.sub_matrix_str: str = sub_matrix
        self.sub_matrix: SArray = self.get_substitution_matrix()
        self.kernel_value = self.k()

    def get_substitution_matrix(self) -> SArray:
        return self.scale_substitution_matrix(substitution_matrices.load(self.sub_matrix_str)) 

    @staticmethod
    def scale_substitution_matrix(mat: SArray) -> SArray:
        """scale input substitution matrix between zero and one"""
        # TODO test to apply exp instead, since it is a log Likelihood
        mat_val = np.array(mat.values())
        normalized_vals = (mat_val - np.min(mat_val) + 1)/(np.max(mat_val)-np.min(mat_val) + 1)
        # substitution matrix object requires iterable for update
        mat.update(zip(mat.keys(), normalized_vals))
        return mat

    def averaged_neighborhood(self, p, q, idx: int) -> float:
        """computes sum over neighborhood (Eq. 7) for both residue chains"""
        # self.p_adjacency = [self.p_adjacency] if len(self.p_adjacency) == 1 else self.p_adjacency
        res_p, neighbors_p = self.p_adjacency[idx]
        res_q, neighbors_q = self.q_adjacency[idx]
        # check if retrived residues are the same
        # TODO this check is false for k_pp and k_qq where checks are different @!!!
        # assert(p == res_p and q == res_q)
        # Point for Improvement: Eq. 7 assumes neighborhoods are equal
        # implication graph structure stays the same during mutations 
        assert(np.all(neighbors_p == neighbors_q))
        n_sum = 0.
        for n_res in neighbors_p:
            # convert indices of neighbors to Sequence string
            n_res = self.p_sequence[n_res]
            n_sum += self.sub_matrix.get(n_res).get(n_res)
        return n_sum

    def k(self) -> float:
        k = 0.
        # test if parameters have been set
        if self.p_sequence is None or self.q_sequence is None:
            return k
        assert(self.p_sequence.shape[0] == self.q_sequence.shape[0])
        for idx, (res_x, res_y) in enumerate(zip(self.p_sequence, self.q_sequence)):
            s_val = self.sub_matrix.get(res_x).get(res_y)
            k_val = s_val * self.averaged_neighborhood(p=res_x, q=res_y, idx=idx)
            # compute for normalization
            k_pp = self.sub_matrix.get(res_x).get(res_x) 
            k_qq = self.sub_matrix.get(res_y).get(res_y)
            k_pp *= self.averaged_neighborhood(p=res_x, q=res_x, idx=idx)
            k_qq *= self.averaged_neighborhood(p=res_y, q=res_y, idx=idx)
            # normalized_k = k_val / np.sqrt(k_pp*k_qq)
            # TODO normalized value always 1 - since there are no mutations yet
            # k += normalized_k
            k += k_val
        return k

