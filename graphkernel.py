import numpy as np
import torch
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from Bio.Align import substitution_matrices
from Bio.Align.substitution_matrices import Array as SArray
from contactmapper import ContactMapper

class GraphKernel:
    def __init__(self, cm_protein: ContactMapper, cm_protein_variant: ContactMapper,
                    depth: int=1, σ_E=0.0, σ_S=0.0, t=0.0, w=0.0, γ=0.0):
        self.protein = cm_protein
        self.protein_variant = cm_protein_variant
        self.adjacency_list = self.generate_adjacency()
        self.depth = depth # not used downstream - define neighbors of neighbors
        self.sub_mat_list = ["BLOSUM62", "BLOSUM50", "BLOSUM45", "BLOSUM80"]
        self.substitution_matrices = self.get_substitution_matrices()
        self.n_S = len(self.substitution_matrices.keys())
        self.kernel = self.compute_normalized_k()
        self.w = torch.rand(self.n_S) if not w else w
        self.γ = torch.rand(self.n_S) if not γ else γ
        self.kernel_parameters: dict = {"σ_E": σ_E, "σ_S": σ_S,
                                "t": t, "w": self.w, "γ": self.γ}
        self.K_ϕ = self.compute_MKL()

    def get_substitution_matrices(self) -> dict:
        s_dict = {mat: self.scale_substitution_matrix(substitution_matrices.load(mat)) for mat in self.sub_mat_list}
        return s_dict

    def generate_adjacency(self) -> list:
        """compute adjacency from contact map by retrieving indices"""
        contact_indices = [np.where(row==1) for row in self.protein.contact_maps]
        adj_res_list = self.protein.sequence[contact_indices]
        return adj_res_list

    @staticmethod
    def scale_substitution_matrix(mat):
        """scale input substitution matrix between zero and one"""
        # TODO apply exp instead, since it is a log Likelihood
        mat_val = np.array(mat.values())
        normalized_vals = (mat_val - np.min(mat_val) + 1)/(np.max(mat_val)-np.min(mat_val) + 1)
        # substitution matrix object requires iterable dict for update
        update_dict = {pos: val for pos, val in zip(mat.keys(), normalized_vals)}
        mat.update(update_dict)
        return mat

    def compute_neighborhood(self, res, res_idx) -> float:
        neighborhood_sum = 0.
        neighborhood = []
        for neighbor in self.adjacency_list[res_idx]:
            neighborhood_sum += self.S(res, neighbor)
            neighborhood.append(self.S(res, neighbor))
        neighborhood_mean = np.mean(np.array(neighborhood))
        # compute sum averaged sum over adjacent positions 
        # TODO averaged substitution matrix
        return neighborhood_sum, neighborhood_mean

    def k(self, x, x_) -> np.ndarray:
        N = x.shape[0]
        k = np.zeros((N, N))
        for i, res_x in enumerate(x):
            for j, res_y in enumerate(x_):
                s = self.S(res_x, res_y)
                neighborhood_res_x = self.compute_neighborhood(res_x, i)
                neighborhood_res_y = self.compute_neighborhood(res_y, j)
            k[i:j] = (s * np.mean(neighborhood_res_x, neighborhood_res_y))
        return k

    @staticmethod
    def S(x: str, x_: str, matrix: SArray) -> float:
        # lookup in Blosum Matrix
        return matrix.get(x).get(x_)

    def compute_normalized_k(self) -> np.ndarray:
        norm_factor = np.sqrt(self.k(self.protein, self.protein)*self.k(self.protein_variant, self.protein_variant))
        normalized_k = self.k(self.protein, self.protein_variant) / norm_factor
        return normalized_k

    def compute_MKL(self) -> np.ndarray:
        K_ = 0.
        # weighted kernel by n of internal matrices
        for m in range(self.n_S):
            K_ += self.w[m] * self.kernel[m]**self.γ[m]
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

