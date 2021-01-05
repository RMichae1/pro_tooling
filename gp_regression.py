import numpy as np
from numpy.random import multivariate_normal
import torch
from torch import cholesky, cholesky_solve
from torch.distributions import MultivariateNormal, Gamma
import matplotlib.pyplot as plt
from typing import Tuple
from protein_representation import ProteinCollection, AdditiveNoiseRepresentation

# for reproducability:
torch.manual_seed(42)
np.random.seed(42)

class GPRegression:
    def __init__(self, protein_representation: ProteinCollection, noise_factor: AdditiveNoiseRepresentation, n_samples=100):
        self.id = protein_representation.pdb_ID
        self.X = np.array(protein_representation.mutation_ids)
        # TODO: mutation-lvl CV: randomly select mutations for train, test
        self.X_train, self.x_test = None, None
        self.y_train, self.y_test = None, None
        self.K_XX, self.K_xX, self.K_xx = None, None, None
        self.σ = noise_factor.σ
        self.N = 0

        # TODO: mutation-lvl CV: randomly select mutations for train, test
        # TODO  for now cutoff at -10 elem
        self.y = np.array(protein_representation.ΔΔg)
        self.kernel = protein_representation.mWDK.K_ϕ
        self.p_sample = None
        self.n_samples = n_samples
        self.mutation_level_dict = self.mutation_level_GPR()

    def mutation_level_GPR(self) -> dict:
        """
        iteratively sets train and test splits
        has side-effects
        This trains on N-1 data and includes the excluded for test
        TODO not optimal - different approaches needed
        """
        mutation_lvl_dict = dict(μ_list = [], cov_list = [], lml_list = [], samples=[])
        for idx, _ in enumerate(self.X):
            # assign each mutation to testing split 
            self.X_train, self.x_test = self.X[np.arange(self.X.shape[0])!=idx], self.X[idx]
            self.N = len(self.X_train)
            self.y_train, self.y_test = self.y[np.arange(self.y.shape[0])!=idx], self.y[idx]
            self.y_train = torch.Tensor(self.y_train[:, np.newaxis]) 
            #self.y_test = torch.Tensor(self.y_test[:, np.newaxis])
            kernel_size = self.kernel.shape[0]
            # use np slicing index to cut out testing mutation from training kernel
            drop_mutation_row = torch.vstack((self.kernel[:idx, :], self.kernel[idx+1:,]))
            drop_mutation = torch.hstack((drop_mutation_row[:, :idx], drop_mutation_row[:, idx+1:]))
            self.K_XX = drop_mutation
            self.K_xX = drop_mutation_row[:, idx].unsqueeze_(0) # add axis in place
            self.K_xx = torch.Tensor([[self.kernel[idx, idx]]])
            assert(self.K_XX.shape == (self.N, self.N))
            assert(self.K_xX.shape == (1, self.N))
            assert(self.K_xx.shape == (1, 1))

            μ, cov, lml, p_sample = self._fit()
            mutation_lvl_dict['μ_list'].append(μ)
            mutation_lvl_dict['cov_list'].append(cov)
            mutation_lvl_dict['lml_list'].append(lml)
            mutation_lvl_dict['samples'].append(p_sample)
        return mutation_lvl_dict

    def cumulative_mutation_split(self):
        # TODO ? start with 1 mutation in training and increase until all except one mutation (sample)
        pass

    def _fit(self) -> Tuple[torch.Tensor, torch.Tensor, float, np.array]:
        """Alg. 2.1 Rasmussen *GPs in ML* """
        A = self.K_XX + self.σ * torch.eye(self.N)
        L = cholesky(A)
        α = cholesky_solve(self.y_train, L)

        f_μ = torch.matmul(self.K_xX, α)
        v = cholesky_solve(self.K_xX.T, L)
        cov = self.K_xx - torch.matmul(self.K_xX, v)
        # TODO PyTorch MultivariateNormal behaved numerically unstable hence np
        #p_sample = multivariate_normal(f_μ[0].detach().numpy(), cov.detach().numpy(), 
        #                                    size=self.n_samples)
        mN = MultivariateNormal(f_μ, cov)
        p_sample = mN.sample_n(self.n_samples)
        #log_marg_likelihood = mN.log_prob(p_sample)
        log_marg_likelihood = mN.log_prob(self.y_train)
        return f_μ, cov, log_marg_likelihood, p_sample

    def predict(self):
        pass

    def plot(self) -> None:
        _, ax = plt.subplots(1,1, figsize=(15,10))
        for idx, mutation in enumerate(self.X):
            samples = self.mutation_level_dict.get('samples')[idx]
            μ = self.mutation_level_dict.get('μ_list')[idx].squeeze().detach().numpy()
            y_test = self.y[idx]
            for s in samples:
                ax.scatter(y_test, s, color="indianred", alpha=0.3)
            ax.scatter(y_test, μ, color="darkred")
            # annotate mutations at test point
            ax.annotate(mutation, xy=(y_test, μ), xycoords="data", xytext=(10,10), 
                textcoords='offset points') #, arrowprops=dict(facecolor="black", shrink=0.05))
        ax.set_xlabel("measured ΔΔg")
        ax.set_ylabel("predicted ΔΔg")
        plt.title(f"GP Regression (mutation lvl) {self.id}")
        plt.savefig(f"./fig/gpr_{self.id}.png")
        plt.show()
    
    def plot_log_prob(self) -> None:
        _, ax = plt.subplots(1, 1, figsize=(10,10))
        for idx, mut in enumerate(self.X):
            y_train = self.y[np.arange(self.y.shape[0])!=idx]
            lmls = self.mutation_level_dict.get('lml_list')[idx]
            log_prob = [(lml.squeeze().detach().numpy()) for lml in lmls]
            sorted_data = np.array(sorted(zip(y_train, log_prob), key= lambda x: x[0]))
            ax.plot(sorted_data[:, 0], sorted_data[:, 1], "r:", alpha=0.5)
            #ax.plot(sorted_data[:, 0], np.mean(sorted_data[:, 1]), "k.")
            # TODO add mean LML
            # TODO add training data points as dots
        ax.set_xlabel("training ΔΔg")
        ax.set_ylabel("log marginal likelihood")
        plt.title(f"Log Marginal Likelihood over training data {self.id}")
        plt.savefig(f"./fig/gpr_logmarginal_{self.id}.png")
        plt.show()