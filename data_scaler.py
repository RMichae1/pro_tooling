import numpy as np
import pandas as pd
import torch
import pyro
from pyro.infer import MCMC, NUTS
from pyro.infer.mcmc.util import initialize_model
import pyro.distributions as dist
import pickle
from os.path import isfile
import matplotlib.pyplot as plt
import seaborn as sns

class BayesScaler:
    """
    Bayesian In Silico Scaling for Rosetta simulated data.
    Using MCMC (w/ NUTS sampler)
    """
    def __init__(self, ΔΔg, experimentally_observed_ΔΔg,
                    α_a=2., β_a=1.5, α_b=1.3, β_b=2., α_c=2, β_c=5., σ_d=0.15, σ_n=0.5,
                    samples_N=10000, warmup_N=500):
        pyro.set_rng_seed(42)
        pyro.clear_param_store()
        self.samples_N = samples_N
        self.warmup_N = warmup_N
        self.chains_N = 1
        ΔΔg = self._check_observed(ΔΔg, experimentally_observed_ΔΔg) # TODO does not check if mutations align
        self.ΔΔg = torch.Tensor(ΔΔg)
        self.α_a, self.β_a = α_b, β_b
        self.α_b, self.β_b = α_a, β_a
        self.α_c, self.β_c = α_c, β_c
        self.σ_d, self.σ_n = σ_d, σ_n
        ## DEV TEST
        if not isfile('test_mcmc.pickle'):
            self.mcmc = self.run_mcmc()
            with open('test_mcmc.pickle', 'wb') as outfile:
                pickle.dump(self.mcmc.get_samples(), outfile)
        ## DEV TEST save time on compute mcmc
        else:
            with open('test_mcmc.pickle', 'rb') as infile:
                self.mcmc = pickle.load(infile)
        ## END DEV

        # self.mcmc_samples = {k: v.detach().cpu().numpy() for k, v in self.mcmc.get_samples().items()}
        # a = self.mcmc_samples.get('a').mean(0)
        # b = self.mcmc_samples.get('b').mean(0)
        # c = self.mcmc_samples.get('c').mean(0)
        # d = self.mcmc_samples.get('d').mean(0)
        a = self.mcmc.get('a').mean(0).numpy()
        b = self.mcmc.get('b').mean(0).numpy()
        c = self.mcmc.get('c').mean(0).numpy()
        d = self.mcmc.get('d').mean(0).numpy()
        self.θ = a * np.exp(np.dot(c, ΔΔg)) + np.dot(b, ΔΔg) + d
        
        self.σ_T = np.square(self.θ - ΔΔg)
        self.σ_T_mean = np.sum(np.square(self.θ - ΔΔg)) / len(ΔΔg) # TODO check correctness w.r.t. gp_modeling
    
    def _check_observed(ΔΔg, experimentally_observed_ΔΔg) -> np.array:
        # TODO do the mutations have to align as well?? not mentioned in the paper
        experimental = set(experimentally_observed_ΔΔg)
        observed_ΔΔg = [val for val in ΔΔg if val in experimental]
        return observed_ΔΔg

    def _model(self, ΔΔg):
        a = pyro.sample('a', dist.Gamma(self.α_a, self.β_a))
        b = 0.5 * pyro.sample('b', dist.Beta(self.α_b, self.β_b))
        c = 3.33 * pyro.sample('c', dist.Beta(self.α_c, self.β_c))
        d = pyro.sample('d', dist.Normal(-a, self.σ_d))
        θ = a * torch.exp(c*ΔΔg) + b*ΔΔg + d
        with pyro.plate("data", ΔΔg.shape[0]):
            pyro.sample('obs', dist.Normal(θ, self.σ_n), obs=ΔΔg)
    
    def run_mcmc(self):
        nuts_kernel = NUTS(self._model, jit_compile=True)
        # TODO Hamiltonian Scaling also
        mcmc = MCMC(nuts_kernel, num_samples=self.samples_N, warmup_steps=self.warmup_N)
        mcmc.run(self.ΔΔg)
        return mcmc

    @staticmethod
    def summary(samples):
        """
        Utility function to print latent sites' quantile information.
        """
        site_stats = {}
        for site_name, values in samples.items():
            marginal_site = pd.DataFrame(values)
            describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
            site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
        return site_stats

    def print_summary(self):
        #for site, values in self.summary(self.mcmc_samples).items():
        for site, values in self.summary(self.mcmc).items():
            print("Site: {}".format(site))
            print(values, "\n")

    def plot_scaling(self):
        fig, ax = plt.subplots(1,2 ,figsize=(25,10))
        sns.scatterplot(x=self.ΔΔg.numpy(), y=self.θ, ci=self.σ_T, color=".2", marker=".", ax=ax[0])
        ci_pos = self.θ + self.σ_T * 2
        ci_neg = self.θ - self.σ_T * 2
        ax[0].plot(self.ΔΔg.numpy(), ci_pos, "r.") # TODO order points for smoother plotting
        ax[0].plot(self.ΔΔg.numpy(), ci_neg, "r.")
        # TODO add green interval for sampling posterior
        # barplot over scaled y values
        sns.histplot(self.σ_T, ax=ax[1], label="σ_T distribution", stat="density", log_scale=True)
        #ax[1].bar(x=self.σ_T, height=self.θ, width=0.2, label="θ values")
        ax[0].set_xlabel("ΔΔG original")
        ax[1].set_xlabel("σT")
        ax[0].set_ylabel("ΔΔG yE, yS")
        ax[0].set_ylim((-15, 4))
        ax[0].set_xlim((-20, 4))
        plt.suptitle("Stability Transformation")
        plt.savefig("./fig/bayes_scaling.png")
        plt.legend()
        plt.show()
       