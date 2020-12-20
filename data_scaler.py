import numpy as np
import torch
import pyro
from pyro.infer import MCMC, NUTS
from pyro.infer.mcmc.util import initialize_model
import pyro.distributions as dist
import pickle
from os.path import isfile

class BayesScaler:
    """
    Bayesian In Silico Scaling for Rosetta simulated data.
    Using MCMC (w/ NUTS sampler)
    """
    def __init__(self, ΔΔg, 
                    α_a=2., β_a=1.5, α_b=1.3, β_b=2., α_c=2, β_c=5., σ_d=0.15, σ_n=0.5):
        self.samples_N = 10000
        self.warmup_N = 500
        self.chains_N = 1
        self.y = torch.Tensor(ΔΔg)
        self.α_a, self.β_a = α_b, β_b
        self.α_b, self.β_b = α_a, β_a
        self.α_c, self.β_c = α_c, β_c
        self.σ_d, self.σ_n = σ_d, σ_n
        self.θ = []

        ## DEV TEST
        if not isfile('test_mcmc.pickle'):
            self.mcmc, self.y = self.run_mcmc()
            with open('test_mcmc.pickle', 'wb') as outfile:
                pickle.dump(self.mcmc.get_samples(), outfile)
            with open('test_mcmc_y.pickle', 'wb') as outfile:
                pickle.dump(self.y, outfile)
        ## DEV TEST save time on compute mcmc
        else:
            with open('test_mcmc.pickle', 'rb') as infile:
                self.mcmc = pickle.load(infile)
            with open('test_mcmc_y.pickle', 'rb') as infile:
                self.y = pickle.load(infile)
        ## END DEV
        self.σ_T = torch.sum(torch.square(torch.Tensor(self.θ)-self.y)) / self.y.shape[0] # TODO check correctness

    def _model(self, y):
        n_obs = y.shape[0]
        with pyro.plate("data", n_obs):
            a = pyro.sample('a', dist.Gamma(self.α_a, self.β_a))
            b = 0.5 * pyro.sample('b', dist.Beta(self.α_b, self.β_b))
            c = (10/3) * pyro.sample('c', dist.Beta(self.α_c, self.β_c))
            d = pyro.sample('d', dist.Normal(-a, self.σ_d))
            θ = a * torch.exp(c*y) + b*y + d
            self.θ.append(θ)
            y = pyro.sample('y', dist.Normal(θ, self.σ_n), obs=y)
            return y
    
    def run_mcmc(self):
        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(nuts_kernel, num_samples=self.samples_N, warmup_steps=self.warmup_N)
        mcmc.run(self.y)
        obs = mcmc.get_samples()['y'].mean(0)
        return mcmc, obs
       


