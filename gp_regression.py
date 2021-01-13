import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import norm

import torch
from torch import cholesky, cholesky_solve
from torch.distributions import MultivariateNormal, Gamma
import matplotlib.pyplot as plt
from typing import Tuple
from protein_representation import ProteinCollection
from utility import Variable

# for reproducability:
torch.manual_seed(42)
np.random.seed(42)


class GPRegression:
    def __init__(self, protein_representation: ProteinCollection, n_samples=100, n_optimization=20):
        # set hyperparameters - see Appendix mGPfusion
        σ_0=1e-6 
        α_E=2.5
        β_E=0.02
        α_S=50.
        β_S=0.007

        self.n_optimization = n_optimization
        self.protein = protein_representation
        self.id = protein_representation.pdb_ID
        self.X = np.array(protein_representation.mutation_ids)
        # TODO: position-lvl CV: randomly select mutations for train, test
        self.X_train, self.x_test = None, None
        self.y_train, self.y_test = None, None
        self.K_XX, self.K_xX, self.K_xx = None, None, None
        self.μ, self.cov, self.lml, self.p_sample = None, None, None, None
        self.σ_E_prior = Gamma(torch.tensor(α_E), torch.tensor(β_E))
        self.σ_S_prior = Gamma(torch.tensor(α_S), torch.tensor(β_S))

        self.t = 1.1
        # init noise terms
        init_σ_E = 0.075 * torch.ones([1, 1], dtype=torch.float64)
        init_σ_S = 0.1 * torch.ones([1, 1], dtype=torch.float64)
        self.σ_E = Variable(init_σ_E, lower=0.001, upper=10)
        self.σ_S = Variable(init_σ_S, lower=0.001, upper=10)
        self.σ_0 = 1e-5 * torch.ones([1, 1], dtype=torch.float64)
        # TODO+ t*self.σ_T init get this from BayesScaler instead !!
        self.σ_T = torch.ones([len(self.protein.mut_ids_is), 1], dtype=torch.float64) * 0.02
        self.σ = self.set_noise_term()

        # TODO: mutation-lvl CV: randomly select mutations for train, test
        # TODO  for now cutoff at -10 elem
        self.y = np.array(protein_representation.ΔΔg)
        self.p_sample = None
        self.n_samples = n_samples
        # self.mutation_level_GPR()
        # self.multiple_kernel_learning()
        # init weights randomly
        init_w = 0.9 * torch.ones([len(self.protein.covariance_matrices), 1], dtype=torch.float64)
        self.weights = Variable(init_w, lower=0, upper=1) 
        # TODO optimize t
        self.mutation_split_GPR()
    
    def set_noise_term(self):
        σ_E = self.σ_E.get_value()
        σ_S = self.σ_S.get_value()
        σ = torch.cat((self.σ_0, 
                    σ_E * torch.ones([len(self.protein.mut_ids_exp), 1], dtype=torch.float64), 
                    (σ_E + σ_S) * torch.ones([len(self.protein.mut_ids_is), 1], dtype=torch.float64) + self.t*self.σ_T))
        return σ

    def mWDK(self):
        """
        compute weighted kernel value from existing covariance matrix
        """
        n = self.X_train.shape[0]
        k = torch.zeros([n, n], dtype=torch.float64)
        for i, mat in enumerate(self.protein.covariance_matrices.values()):
            #k += self.weights[i].clamp(0,1) * mat[:n, :n]
            k += self.weights.get_value()[i] * mat[:n, :n]
        return k

    def neg_ll(self):
        n = self.X_train.shape[0]
        # use unconstrained params
        # TODO for all element in unconstrained apply constrain
        zero_μ = torch.zeros(n, dtype=torch.float64) # TODO compute mean over all training data
        K_XX = self.mWDK()
        noise = self.set_noise_term().squeeze()[:n]
        #noise = self.σ.squeeze()[:n]
        K_XX = K_XX + torch.diag(noise) # TODO built new self sigma
        # set diagonal to add noise
        # zero mean is consistent due to prior assumption
        # print(K_XX)
        # print(f"noise: {noise}")
        nll = - (MultivariateNormal(zero_μ, covariance_matrix=K_XX).log_prob(torch.Tensor(self.y_train)) \
            + self.σ_E_prior.log_prob(self.σ_E.get_value()) + self.σ_S_prior.log_prob(self.σ_S.get_value()))
        nll.requires_grad_(True)
        return nll

    def parameter_optimization(self) -> None:
        optimizer = torch.optim.LBFGS([self.weights.unconstrained, self.σ_E.unconstrained, self.σ_S.unconstrained],
                                        lr=0.99)
        def closure():
            optimizer.zero_grad()
            loss = self.neg_ll()
            loss.backward(retain_graph=True)
            print(f"Loss: {loss}")
            return loss
        for n in range(self.n_optimization):
            print(f"iter: {n}")
            # print(f"weights: {self.weights.get_unconstrained()} <= {self.weights.get_value()}")
            # print(f"sigmas E: {self.σ_E.get_unconstrained()} <= {self.σ_E.get_value()}")
            optimizer.step(closure)
        # set weights after optimization
        print("FINAL:")
        print("weights:")
        print(self.weights.get_value())
        print("sigma:")
        print(self.σ_S.get_value())
        print(self.σ_E.get_value())
        return 
    
    def mutation_split_GPR(self, training=0.75) -> None:
        """
        Split mutations into 75:25 train test split
        """ 
        cutoff = int(training*self.X.shape[0])
        self.X_train, x_test = self.X[:cutoff], self.X[cutoff:]
        self.y_train, y_test = self.y[:cutoff], self.y[cutoff:]
        return

    def mutation_level_GPR(self) -> dict:
        """
        iteratively sets train and test splits
        has side-effects
        This trains on N-1 data and includes the excluded for test
        TODO not optimal - different approaches needed
        """
        for idx, _ in enumerate(self.X):
            # assign each mutation to testing split 
            self.X_train, self.x_test = self.X[np.arange(self.X.shape[0])!=idx], self.X[idx]
            self.N = len(self.X_train)
            self.y_train, self.y_test = self.y[np.arange(self.y.shape[0])!=idx], self.y[idx]
            self.y_train = torch.Tensor(self.y_train[:, np.newaxis]) 
            #self.y_test = torch.Tensor(self.y_test[:, np.newaxis])
            # use np slicing index to cut out testing mutation from training kernel
            drop_mutation_row = torch.vstack((self.kernel[:idx, :], self.kernel[idx+1:,]))
            drop_mutation = torch.hstack((drop_mutation_row[:, :idx], drop_mutation_row[:, idx+1:]))
            self.K_XX = drop_mutation
            self.K_xX = drop_mutation_row[:, idx].unsqueeze_(0) # add axis in place
            self.K_xx = torch.Tensor([[self.kernel[idx, idx]]])
            assert(self.K_XX.shape == (self.N, self.N))
            assert(self.K_xX.shape == (1, self.N))
            assert(self.K_xx.shape == (1, 1))
            self.μ, self.cov, self.lml, self.p_sample = self._fit()
        return

    def cumulative_mutation_split(self):
        # TODO ? start with 1 mutation in training and increase until all except one mutation (sample)
        pass

    def _fit(self, mu, cov, K, ) -> Tuple[torch.Tensor, torch.Tensor, float, np.array]:
        """Alg. 2.1 Rasmussen *GPs in ML* """
        A = self.K_XX + self.σ * torch.eye(self.N) # TODO add sigma_E add sigma_
        ## 0,0 sigma
        ## diag in range exp + sigma_E
        ## 
        L = cholesky(A)
        α = cholesky_solve(self.y_train, L)

        # init fit
        mean = 0
        cov = A
        # compute dist + lml

        f_μ = torch.matmul(self.K_xX, α)
        v = cholesky_solve(self.K_xX.T, L)
        cov = self.K_xx - torch.matmul(self.K_xX, v)
        mN = MultivariateNormal(f_μ, cov)
        # added gamma prior from noise representation as in (Eq. 10) mGPfusion
        log_marg_likelihood = mN.log_prob(self.y_train) + self.noise.σ_E.log_prob(self.noise.σ_E.sample()) + self.noise.σ_S.log_prob(self.noise.σ_S.sample())
        return log_marg_likelihood

    def predict(self):
        return mu, cov, p_sample
        p_sample = mN.sample((self.n_samples,))
        

    def plot(self) -> None:
        _, ax = plt.subplots(1,1, figsize=(15,10))
        for idx, mutation in enumerate(self.X):
            samples = self.mutation_level_dict.get('samples')[idx]
            μ = self.mutation_level_dict.get('μ_list')[idx].squeeze().detach().numpy()
            cov = self.mutation_level_dict.get('cov_list')[idx].squeeze().detach().numpy()
            y_test = self.y[idx]
            for s in samples:
                ax.scatter(y_test, s, color="indianred", alpha=0.3)
            # plot gaussian from computed mean and covariance
            xx = np.arange(-5, 5, 0.1)
            f = norm.pdf(xx, μ, cov)
            ax.plot(y_test+f, xx, "k-")
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