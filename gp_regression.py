import numpy as np
from numpy.random import multivariate_normal
import torch
from torch import cholesky, cholesky_solve
from torch.distributions import MultivariateNormal, Gamma
import matplotlib.pyplot as plt
from protein_representation import ProteinCollection, AdditiveNoiseRepresentation

class GPRegression:
    def __init__(self, protein_representation: ProteinCollection, noise_factor: AdditiveNoiseRepresentation, n_samples=100):
        # TODO: mutation-lvl CV: randomly select mutations for train, test
        self.X = protein_representation.mutation_ids
        self.X_train, self.x_test = protein_representation.mutation_ids[:-10], protein_representation.mutation_ids[-10:]
        self.σ = noise_factor.σ
        self.N = len(self.X)

        self.y = protein_representation.ΔΔg[:-10, np.newaxis]
        self.y_test = protein_representation.ΔΔg[-10:, np.newaxis]
        self.kernel = protein_representation.mWDK.K_ϕ
        self.p_sample = None`
        self.n_samples = n_samples
        self.K_XX = self.kernel(self.X_train, self.X_train)
        self.K_xX = self.kernel(self.x_test, self.x_test)
        self.K_xx = self.kernel(self.x_test, self.x_test)
        self.μ, self.cov = self.fit()
        # TODO posterior log-likelihood
        l_p = 0 # log-marginal likelihood of model at this point with MWDK

    def fit(self):
        """Alg. 2.1 Rasmussen *GPs in ML* """
        A = self.K_XX + self.σ * torch.eye(self.N)
        L = cholesky(A)
        α = cholesky_solve(self.y, L)

        f_μ = torch.matmul(self.K_xX, α)
        v = cholesky_solve(self.K_xX.T, L)
        cov = self.K_xx - torch.matmul(self.K_xX, v)
        # TODO PyTorch MultivariateNormal behaved numerically unstable hence np
        self.p_sample = multivariate_normal(f_μ.squeeze().numpy(), cov.numpy(), 
                                            size=self.n_samples)
        return f_μ, cov

    def predict(self):
        pass

    def plot(self):
        pass