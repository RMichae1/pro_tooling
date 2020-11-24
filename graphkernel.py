import numpy as np
import torch
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from Bio.Align import substitution_matrices
from Bio.Align.substitution_matrices import Array as SArray
from contactmapper import ContactMapper

class GraphKernel:
    def __init__(self, cm_protein: ContactMapper, cm_protein_variant: ContactMapper,
                    σ_E=0.0, σ_S=0.0, t=0.0, w=0.0, γ=0.0):
        self.protein = cm_protein
        self.protein_variant = cm_protein_variant
        # TODO import alphabet and replace constant 21 in the code with length of alphabet
        self.blosum62 = self.scale_substitution_matrix(substitution_matrices.load("BLOSUM62").values())
        self.bl osum50 = self.scale_substitution_matrix(substitution_matrices.load("BLOSUM50").values())
        self.blosum45 = self.scale_substitution_matrix(substitution_matrices.load("BLOSUM45").values())
        self.kernel = self.compute_normalized_k()
        self.w = torch.rand(21) if not w else w
        self.γ = torch.rand(21) if not γ else γ
        self.kernel_parameters: dict = {"σ_E": σ_E, "σ_S": σ_S,
                                "t": t, "w": self.w, "γ": self.γ}
        self.K_ϕ = self.compute_MKL()

    @staticmethod
    def scale_substitution_matrix(mat) -> np.ndarray:
        """scale input substitution matrix between zero and one"""
        return (mat - np.min(mat) + 1)/(np.max(mat)-np.min(mat) + 1)

    @staticmethod
    def compute_neighborhood(pos_x: int, pos_y: int) -> float:
        neighborhood_sum = 0.
        # do something
        return neighborhood_sum

    @staticmethod
    def k(x, x_) -> np.ndarray:
        k = np.zeros((len(x), len(x_)))
        for i, res_x in enumerate(x):
            for j, res_x_ in enumerate(x_):
                s = self.S(res_x, res_y)
                for res in neighborhood(x):
                neighborhood = n_sum(res, ?)
            k[idx:?] = (s * neighborhood)
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
        K_ = self.kernel.shape()
        for m in range(21):
            K_= self.w[m] * self.kernel[m]**self.γ[m]
        return K_


    def parameter_inference(self):
        α_S = Normal(0, σ_S)
        β_S = Normal(0, σ_S)
        α_E = Normal(0, σ_E)
        β_E = Normal(0, σ_E)
        σ = np.array([0]) # joint variance parameterized by (σ_S, σ_E, t) - TODO how to put this together?

        marginal_log_likelihood = α - 0.5*y.T*(self.K_ϕ + σ.diagonal()) - 0.5*np.log(K_ϕ + σ.diagonal()) + np.log(Gamma(α_E, β_E).sample()) + np.log(Gamma(α_S, β_S).sample())
        return marginal_log_likelihood

