import numpy as np
from numpy.random import multivariate_normal
import torch
from torch import cholesky, cholesky_solve
from torch.distributions import MultivariateNormal, Gamma
import matplotlib.pyplot as plt
from protein_representation import ProteinCollection, AdditiveNoiseRepresentation

class GPRegression:
    def __init__(self, protein_representation: ProteinCollection, noise_factor: AdditiveNoiseRepresentation, n_samples=100):
        self.id = protein_representation.pdb_ID
        self.X = protein_representation.mutation_ids
        # TODO: mutation-lvl CV: randomly select mutations for train, test
        self.X_train, self.x_test = protein_representation.mutation_ids[:-10], protein_representation.mutation_ids[-10:]
        self.X_idx = np.arange(len(self.X))
        self.σ = noise_factor.σ
        self.N = len(self.X_train)

        # TODO: mutation-lvl CV: randomly select mutations for train, test
        # TODO  for now cutoff at -10 elem
        self.y = np.array(protein_representation.ΔΔg[:-10])
        self.y_test = np.array(protein_representation.ΔΔg[-10:])
        self.y_tensor = torch.Tensor(self.y[:, np.newaxis]) # -10 index as example
        self.y_test_tensor = torch.Tensor(self.y_test[:, np.newaxis])
        self.kernel = protein_representation.mWDK.K_ϕ
        self.p_sample = None
        self.n_samples = n_samples
        self.K_XX = self.kernel[:-10, :-10]
        self.K_xX = self.kernel[-10:, :-10]
        self.K_xx = self.kernel[-10:, -10:]
        self.μ, self.cov = self.fit()
        # TODO posterior log-likelihood
        l_p = 0 # log-marginal likelihood of model at this point with MWDK

    def fit(self):
        """Alg. 2.1 Rasmussen *GPs in ML* """
        A = self.K_XX + self.σ * torch.eye(self.N)
        L = cholesky(A)
        α = cholesky_solve(self.y_tensor, L)

        f_μ = torch.matmul(self.K_xX, α)
        v = cholesky_solve(self.K_xX.T, L)
        cov = self.K_xx - torch.matmul(self.K_xX, v)
        # TODO PyTorch MultivariateNormal behaved numerically unstable hence np
        self.p_sample = multivariate_normal(f_μ.squeeze().detach().numpy(), cov.detach().numpy(), 
                                            size=self.n_samples)
        return f_μ, cov

    def predict(self):
        pass

    def plot(self):
        _, ax = plt.subplots(1,1, figsize=(15,10))
        for s in self.p_sample:
            ax.scatter(self.y_test, s, color="indianred", alpha=0.3)
        μ = self.μ.squeeze().detach().numpy()
        ax.scatter(self.y_test, μ, color="darkred")
        # annotate mutations per test point
        for i, mut in enumerate(self.x_test):
            ax.annotate(mut, xy=(self.y_test[i], μ[i]), xycoords="data", xytext=(10,10), 
            textcoords='offset points') #, arrowprops=dict(facecolor="black", shrink=0.05))
        ax.set_xlabel("measured ΔΔg")
        ax.set_ylabel("predicted ΔΔg")
        plt.title(f"GP Regression (ex. 10 test mutations) {self.id}")
        plt.savefig(f"./fig/gpr_test_{self.id}.png")
        plt.show()