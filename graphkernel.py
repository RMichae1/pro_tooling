import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from Bio.Align import substitution_matrices
from Bio.Align.substitution_matrices import Array as SArray
from contactmapper import ContactMapper  
    
    
class WeightedDecompositionKernel:
    def __init__(self, p: ContactMapper, q: ContactMapper,
                    depth: int=1, default_matrix="BLOSUM62", σ_E=0.0, σ_S=0.0, t=0.0, w=0.0, γ=1.0):
        self.p = p
        self.q = q
        self.depth = depth # not used downstream - define neighbors of neighbors
        self.default_matrix = default_matrix
        self.sub_mat_list = ["BLOSUM62", "BLOSUM50", "BLOSUM45", "BLOSUM80"]
        if self.default_matrix not in self.sub_mat_list:
            raise RuntimeWarning(f"Substitutionmatrix default_matrix={default_matrix} unknown!")
        self.substitution_matrices: dict = self.get_substitution_matrices()
        self.n_S = len(self.substitution_matrices.keys())
        self.kernel = self.compute_normalized_k()
        self.w = Parameter(torch.rand(self.n_S)) if not w else w
        self.γ = γ
        self.kernel_parameters: dict = {"σ_E": σ_E, "σ_S": σ_S,
                                "t": t, "w": self.w, "γ": self.γ}
        #self.K_ϕ = self.compute_MKL()

    def get_substitution_matrices(self) -> dict:
        s_dict = {mat: self.scale_substitution_matrix(substitution_matrices.load(mat)) for mat in self.sub_mat_list}
        return s_dict

    @staticmethod
    def scale_substitution_matrix(mat: SArray) -> SArray:
        """scale input substitution matrix between zero and one"""
        # TODO test to apply exp instead, since it is a log Likelihood
        mat_val = np.array(mat.values())
        normalized_vals = (mat_val - np.min(mat_val) + 1)/(np.max(mat_val)-np.min(mat_val) + 1)
        # substitution matrix object requires iterable object for update
        mat.update(zip(mat.keys(), normalized_vals))
        return mat

    def averaged_neighborhood(self, p, q, idx: int) -> float:
        """computes sum over neighborhood (Eq. 7) for both residue chains"""
        res_p, neighbors_p = self.p.adjacency[idx]
        res_q, neighbors_q = self.q.adjacency[idx]
        # check if retrived residues are the same
        assert(p == res_p and q == res_q)
        # Point for Improvement: Eq. 7 assumes neighborhoods are equal
        # implication graph structure stays the same during mutations 
        assert(np.all(neighbors_p == neighbors_q))
        n_sum = 0.
        for n_res in neighbors_p:
            # convert indices of neighbors to Sequence string
            n_res = self.p.sequence[n_res]
            n_sum += self.substitution_matrices.get("BLOSUM62").get(n_res).get(n_res)
        print(f"neighborhood sum: {n_sum}")
        return n_sum

    def k(self, p: str, q: str) -> np.ndarray:
        N = p.shape[0]
        k = 0.
        assert(p.shape[0] == q.shape[0])
        for idx, (res_x, res_y) in enumerate(zip(p, q)):
            # TODO for all matrices in matrix list
            s_val = self.substitution_matrices.get("BLOSUM62").get(res_x).get(res_y)
            k += s_val * self.averaged_neighborhood(p=res_x, q=res_y, idx=idx)
        print(f"k value: {k}")
        return k

    def compute_normalized_k(self) -> np.ndarray:
        norm_factor = np.sqrt(self.k(self.p.sequence, self.p.sequence)*self.k(self.q.sequence, self.q.sequence))
        normalized_k = self.k(self.p.sequence, self.q.sequence) / norm_factor
        return normalized_k

    def compute_MKL(self) -> np.ndarray:
        K_ = 0.
        # weighted kernel by n of internal matrices
        for m in range(self.n_S):
            K_ += self.w[m] * self.kernel[m]**self.γ[m]
            # w is torch trainable parameter
        return K_

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

