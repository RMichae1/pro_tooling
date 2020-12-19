import torch
import pyro
from pyro.infer import MCMC, NUTS
import pyro.poutine as poutine
import pyro.distributions as d

class IS_Scaler:
    """
    Bayesian In Silico Scaling for Rosetta simulated data
    """
    def __init__(self, protein, a, b, c, d, 
                α_a=2., β_a=1.5, α_b=1.3, β_b=2., α_c=2, β_c=5., σ_d=0.15, σ_n=0.5):
        self.protein = protein
        self.samples_N = 10000
        self.warmup_N = 500
        self.chains_N = 1
        self.y = None # TODO read out from IS mat data
        self.a, self.b, self.c, self.d = a, b, c, d
        self.α_a, self.β_a = α_b, β_b
        self.α_b, self.β_b = α_a, β_a
        self.α_c, self.β_c = α_c, β_c
        self.σ_d, self.σ_n = σ_d, σ_n
        self.θ = (self.a, self.b, self.c, self.d)

        self.mcmc = self.run_mcmc()
        self.y = self.mcmc.get_samples()['obs'].mean(0)
        self.σ_T = self.y - 0 # TODO difference between posterior mean and individual value ???

    def _model(self):
        a = pyro.sample('a', d.Gamma(self.α_a, self.β_a))
        b = 0.5 * pyro.sample('b', d.Beta(self.α_b, self.β_b))
        c = (10/3) * pyro.sample('c', d.Beta(self.α_c, self.β_c))
        d = pyro.sample('d', d.Normal(-a, self.σ_d))

        θ = a * torch.exp(c*self.y) + b*self.y + d

        return pyro.sample("obs", d.Normal(θ, self.σ_n))

    def _conditioned_model(self):
        """
        treat observations as ocnditionally independent
        """
        poutine.condition(self._model, data={'obs': self.y})(self.σ_n)
    
    def run_mcmc(self):
        nuts_kernel = NUTS(self._conditioned_model, jit_compile=True)
        mcmc = MCMC(nuts_kernel, num_samples=self.samples_N, warmup_steps=self.warmup_N, num_chains=self.chains_N)
        mcmc.run(self._model)
        return mcmc
       


