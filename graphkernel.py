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
    def __init__(self, σ_E=0.0, σ_S=0.0, t=0.0, w=0.0, γ=1.0):
        self.w = Parameter(rand(len(self.kernel.keys()))) if not w else w
        self.γ = γ
        self.kernel_parameters: dict = {"σ_E": σ_E, "σ_S": σ_S, "t": t, "w": self.w, "γ": self.γ}
        self.K_ϕ = self.compute_MKL()

    def compute_MKL(self) -> Tensor:
        K_ = Tensor([0.])
        # weighted kernel by n of internal matrices
        for i, m in enumerate(self.kernel.keys()):
            K_ += self.w[i] * self.kernel[m]**self.γ
        return K_
    
    
class MatrixKernel:
    def __init__(self, p_sequence: list, p_adjacency:tuple,
                    q_sequence: list, q_adjacency: tuple,
                    depth: int=1):
        self.p_sequence = p_sequence
        self.p_adjacency = p_adjacency
        self.q_sequence = q_sequence
        self.q_adjacency = q_adjacency
        self.depth = depth # not used downstream - define neighbors of neighbors
        self.sub_mat_list = ["BLOSUM62", "BLOSUM50", "BLOSUM45", "BLOSUM80"]
        self.substitution_matrices: dict = self.get_substitution_matrices()
        self.kernel = {}
        for s_matrix in self.sub_mat_list:
            self.kernel[s_matrix] = self.k(self.p_sequence, self.q_sequence, s_matrix)

    def get_substitution_matrices(self) -> dict:
        s_dict = {mat: self.scale_substitution_matrix(substitution_matrices.load(mat)) 
                        for mat in self.sub_mat_list}
        return s_dict

    @staticmethod
    def scale_substitution_matrix(mat: SArray) -> SArray:
        """scale input substitution matrix between zero and one"""
        # TODO test to apply exp instead, since it is a log Likelihood
        mat_val = np.array(mat.values())
        normalized_vals = (mat_val - np.min(mat_val) + 1)/(np.max(mat_val)-np.min(mat_val) + 1)
        # substitution matrix object requires iterable for update
        mat.update(zip(mat.keys(), normalized_vals))
        return mat

    def averaged_neighborhood(self, p, q, idx: int, s_matrix: str) -> Tensor:
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
            n_sum += self.substitution_matrices.get(s_matrix).get(n_res).get(n_res)
        return Tensor([n_sum])

    def k(self, p_seq: str, q_seq: str, s_matrix: str) -> Tensor:
        k = 0.
        assert(p_seq.shape[0] == q_seq.shape[0])
        for idx, (res_x, res_y) in enumerate(zip(p_seq, q_seq)):
            s_val = self.substitution_matrices.get(s_matrix).get(res_x).get(res_y)
            k_val = s_val * self.averaged_neighborhood(p=res_x, q=res_y, idx=idx, s_matrix=s_matrix)
            # compute for normalization
            k_pp = self.substitution_matrices.get(s_matrix).get(res_x).get(res_x)
            k_qq = self.substitution_matrices.get(s_matrix).get(res_y).get(res_y)
            k_pp *= self.averaged_neighborhood(p=res_x, q=res_x, idx=idx, s_matrix=s_matrix)
            k_qq *= self.averaged_neighborhood(p=res_y, q=res_y, idx=idx, s_matrix=s_matrix)
            # normalized_k = k_val / np.sqrt(k_pp*k_qq)
            # TODO normalized value always 1 - since there are no mutations yet
            # k += normalized_k
            k += k_val
        print(f"{s_matrix} k value: {k}")
        return Tensor([k])



    def parameter_inference(self):
        α_S = Normal(0, self.kernel_parameters.get("σ_S"))
        β_S = Normal(0, self.kernel_parameters.get("σ_S"))
        α_E = Normal(0, self.kernel_parameters.get("σ_E"))
        β_E = Normal(0, self.kernel_parameters.get("σ_E"))
        σ = np.array([0]) # joint variance parameterized by (σ_S, σ_E, t) - TODO how to put this together?

        # TODO no marg. log likelihood here - this is for GP inference
        # TODO how do I pass around these parameters?
        #marginal_log_likelihood = α - 0.5*y.T*(self.K_ϕ + σ.diagonal()) - 0.5*np.log(K_ϕ + σ.diagonal()) + np.log(Gamma(α_E, β_E).sample()) + np.log(Gamma(α_S, β_S).sample())
        return #marginal_log_likelihood

