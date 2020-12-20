import numpy as np
from numpy.random import multivariate_normal
import torch
from torch import cholesky, cholesky_solve
from torch.distributions import MultivariateNormal, Gamma
import matplotlib.pyplot as plt

class GPRegression:
    def __init__(self, X, x_space, y, kernel, σ, mean_function=m, n_samples=100):
        self.X = X
        self.N = X.shape[0]
        self.x_space = x_space
        self.y = y[:, np.newaxis]
        self.kernel = kernel
        self.p_sample = None
        self.n_samples = n_samples
        self.K_XX = self.kernel(X, X)
        self.K_xX = self.kernel(x_space, X)
        self.K_xx = self.kernel(x_space, x_space)
        self.μ = mean_function
        self.log_m_likelihood = None

    def fit(self):
        """Alg. 2.1 Rasmussen *GPs in ML* """
        A = self.K_XX + self.σ * torch.eye(self.N)
        L = cholesky(A)
        α = cholesky_solve(self.y, L)

        f_μ = torch.matmul(k_xX, α)
        v = cholesky_solve(k_xX.T, L)
        cov = self.K_xx - torch.matmul(k_xX, v)

        # TODO posterior log-likelihood
        l_p = 0 # log-marginal likelihood of model at this point with MWDK
        # PyTorch MultivariateNormal behaved numerically unstable hence np
        self.p_sample = multivariate_normal(f_μ.squeeze().numpy(), cov.numpy(), 
                                            size=self.n_samples)
        self.log_m_likelihood = lp + self.Γ_E.log_prob() + self.Γ_S.log_prob()
        
        return f_μ, cov, l_p

    def predict(self):
        pass

    def plot(self):
        pass