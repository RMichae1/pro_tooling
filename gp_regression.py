import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import norm
from tqdm import tqdm
from typing import List, Tuple

import torch
from torch import cholesky, cholesky_solve
from torch.distributions import MultivariateNormal, Gamma
import matplotlib.pyplot as plt
from typing import Tuple
from protein_representation import ProteinCollection
from graphkernel import KernelLoader
from utility import Variable

# for reproducability:
torch.manual_seed(42)
np.random.seed(42)


class GPRegression:
    def __init__(self, protein_representation: ProteinCollection, X_wt: np.ndarray, 
                X_exp: np.ndarray, X_is: np.ndarray, y_wt: np.ndarray, y_exp: np.ndarray, y_is: np.ndarray,
                y_max: float, adjacencies: np.ndarray, σ_T: float, n_samples=100, n_optimization=20):
        self.X_wt, self.X_exp, self.X_is = X_wt, X_exp, X_is
        self.y_wt, self.y_exp, self.y_is = y_is, y_exp, y_is
        self.y_max = y_max
        self.protein = protein_representation
        self.adjacencies = adjacencies
        # set hyperparameters - see Appendix mGPfusion
        σ_0=1e-6 
        α_E=2.5
        β_E=1/0.02
        α_S=50.
        β_S=1/0.007
        self.t = Variable(1.1 * torch.ones([1, 1], dtype=torch.float64), lower=0.001, upper=10)
        # init prior noise
        self.σ_E_prior = Gamma(torch.tensor(α_E), torch.tensor(β_E))
        self.σ_S_prior = Gamma(torch.tensor(α_S), torch.tensor(β_S))
        # init noise terms
        init_σ_E = 0.075 * torch.ones([1, 1], dtype=torch.float64)
        init_σ_S = 0.1 * torch.ones([1, 1], dtype=torch.float64)
        self.σ_E = Variable(init_σ_E, lower=0.001, upper=10)
        self.σ_S = Variable(init_σ_S, lower=0.001, upper=10)
        self.σ_0 = 1e-5 * torch.ones([1, 1], dtype=torch.float64)
        # TODO+ t*self.σ_T init get this from BayesScaler instead !!
        self.σ_T = σ_T * torch.ones([1, 1], dtype=torch.float64)
        self.σ = self.set_noise_term()
        

        self.n_optimization = n_optimization
        self.id = protein_representation.pdb_ID
        self.X, self.y = self._combine_observations(X_wt, X_exp, X_is, y_wt, y_exp, y_is)
        # initialize required variables for training GP
        # TODO: position-lvl CV: randomly select mutations for train, test
        self.K_XX, self.K_xX, self.K_xx = None, None, None
        self.μ, self.cov, self.lml, self.p_sample = None, None, None, None

        _kernels = KernelLoader()
        self._kernels = _kernels.kernels
        self.p_sample = None
        self.n_samples = n_samples
        # self.mutation_level_GPR()
        # self.multiple_kernel_learning()
        # init weights randomly
        init_w = (0.9/len(self._kernels)) * torch.ones([len(self._kernels), 1], dtype=torch.float64)
        self.weights = Variable(init_w, lower=0, upper=1) 
        self.X_train, self.x_test, self.y_train, self.y_test = self.mutation_split_GPR()

        # TODO WARN: what matrix size to compute matters!
        self.covariance_matrices = self.compute_matrices(X=self.X, 
                                                        adjacencies=self.adjacencies[:len(self.X)])
        # trainable parameters for testing
        self.trainable_parameters: list = [w for w in self.weights.get_value()] + [self.σ_E, self.σ_S, self.t]
    
    @staticmethod
    def check_and_add_axis(x: np.ndarray) -> np.ndarray:
        return x[:, np.newaxis] if len(x.shape) == 1 else x

    def _combine_observations(self, X_wt: np.ndarray, X_exp: np.ndarray, X_is: np.ndarray, 
                            y_wt: np.ndarray, y_exp: np.ndarray, y_is: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        y_wt = self.check_and_add_axis(y_wt)
        X_wt = self.check_and_add_axis(X_wt)
        X_exp, X_is = self.check_and_add_axis(X_exp), self.check_and_add_axis(X_is)
        y_exp, y_is = self.check_and_add_axis(y_exp), self.check_and_add_axis(y_is)
        assert X_exp.shape[1] == X_wt.shape[1]
        assert y_exp.shape[1] == y_wt.shape[1]
        X = torch.Tensor(np.vstack([X_wt, X_exp, X_is]))
        y = torch.Tensor(np.vstack([y_wt, y_exp, y_is]))
        return X, y
    
    def set_noise_term(self):
        σ_E = self.σ_E.get_value()
        σ_S = self.σ_S.get_value()
        t = self.t.get_value()
        σ = torch.cat((self.σ_0, 
                    (σ_E/self.y_max) * torch.ones([len(self.X_exp), 1], dtype=torch.float64), 
                    ((σ_E + σ_S) / self.y_max) * torch.ones([len(self.X_is), 1], dtype=torch.float64) + t*(self.σ_T/self.y_max)))
        return torch.square(σ)

    def compute_matrices(self, X: torch.Tensor, adjacencies: List[tuple]) -> list:
        X = X.detach().numpy().astype(np.int64)
        n = X.shape[0]
        covariance_mats = []
        for i, kernel in tqdm(enumerate(self._kernels)):
            k = torch.zeros([n, n], dtype=torch.float64).float()
            k += kernel.k(X, adjacencies).float()
            covariance_mats.append(k)
        return covariance_mats

    def mWDK(self, X: torch.Tensor) -> torch.Tensor:
        """
        compute weighted kernel value from existing covariance matrix
        """
        n = X.shape[0]
        k = torch.zeros([n, n], dtype=torch.float64)
        assert np.all(n == mat.shape[0] for mat in self.covariance_matrices)
        # TODO query matrix values through X
        for i, mat in enumerate(self.covariance_matrices):
            # WARN: This operation is of type double, but torch doesnt complain
            k += self.weights.get_value()[i].type(torch.float64) * mat[:n, :n]
        return k

    def neg_ll(self):
        n = self.X.shape[0]
        # use unconstrained params
        # TODO for all element in unconstrained apply constrain
        zero_μ = torch.zeros(n, dtype=torch.float64) # TODO compute mean over all training data
        K_XX = self.mWDK(X=self.X)
        noise = self.set_noise_term().squeeze()[:n]
        K_XX = K_XX + torch.diag(noise)
        # zero mean is consistent due to prior assumption
        # print(f"noise: {noise}")
        nll = -(MultivariateNormal(zero_μ, covariance_matrix=K_XX).log_prob(torch.flatten(self.y)).sum() \
            + self.σ_E_prior.log_prob(self.σ_E.get_value()) + self.σ_S_prior.log_prob(self.σ_S.get_value()))
        nll.requires_grad_(True)
        return nll

    def parameter_optimization(self) -> None:
        optimizer = torch.optim.LBFGS([self.weights.unconstrained, self.σ_E.unconstrained, 
                                        self.σ_S.unconstrained, self.t.unconstrained],
                                        lr=0.99)
        def closure():
            optimizer.zero_grad()
            loss = self.neg_ll().sum()
            print(f"Loss: {loss}")
            # TODO double check sum or better mean
            loss.backward()
            return loss
        for n in range(self.n_optimization):
            print(f"iter: {n}")
            # print(f"weights: {self.weights.get_unconstrained()} <= {self.weights.get_value()}")
            # print(f"sigmas E: {self.σ_E.get_unconstrained()} <= {self.σ_E.get_value()}")
            optimizer.step(closure)
        # set weights after optimization
        print("FINAL:")
        print(f"weights: {self.weights.get_value()}")
        print(f"sigmas S={self.σ_S.get_value()} E={self.σ_E.get_value()}:")
        print(f"t = {self.t.get_value()}")
        return 
    
    def mutation_split_GPR(self, training=0.75) -> None:
        """
        Split mutations into 75:25 train test split
        """ 
        cutoff = int(training*self.X.shape[0])
        X_train, x_test = self.X[:cutoff], self.X[cutoff:]
        y_train, y_test = self.y[:cutoff], self.y[cutoff:]
        return X_train, x_test, y_train, y_test

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

    def _fit(self, f_μ=0., cov=None) -> Tuple[torch.Tensor, torch.Tensor, float, np.array]:
        """Alg. 2.1 Rasmussen *GPs in ML* """
        n = self.X_train.shape[0]
        self.K_XX = self.mWDK(self.X_train)
        self.K_xX = self.mWDK(self.X[:, n:])
        self.K_xx = self.mWDK(self.x_test)
        A = self.K_XX + self.σ * torch.eye(n) # TODO add sigma_E add sigma_
        ## 0,0 sigma
        ## diag in range exp + sigma_E
        ## 
        L = cholesky(A)
        α = cholesky_solve(self.y_train, L)

        # init fit
        # mean = 0
        cov = A
        # compute dist + lml

        f_μ = torch.matmul(self.K_xX, α)
        v = cholesky_solve(self.K_xX.T, L)
        cov = self.K_xx - torch.matmul(self.K_xX, v)
        mN = MultivariateNormal(f_μ, cov)
        # added gamma prior from noise representation as in (Eq. 10) mGPfusion
        log_marg_likelihood = mN.log_prob(self.y_train) + self.σ_E_prior.log_prob(self.σ_E.get_value()) + self.σ_S_prior.log_prob(self.σ_S.get_value())
        return f_μ, cov, log_marg_likelihood

    def predict(self, f_μ, cov):
        mN = MultivariateNormal(f_μ, cov)
        p_sample = mN.sample((self.n_samples,))
        return p_sample
        

    def plot(self) -> None:
        μ, cov, lml = self._fit()
        samples = self.predict(μ, cov)
        μ = μ.squeeze().detach().numpy()
        cov = cov.squeeze().detach().numpy()
        y_test = self.y_test
        _, ax = plt.subplots(1,1, figsize=(15,10))
        for idx, (_, mutation) in enumerate(zip(self.x_test, self.protein.mutation_ids[len(self.X_train):])):
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
        for idx, mut in enumerate(self.x_test):
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