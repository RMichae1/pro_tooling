from os import path
import numpy as np
from numpy.random import multivariate_normal
import torch
from torch.distributions import Gamma
from scipy import io


def parse_mutations(mat_file, query: str=None) -> dict:
    if isinstance(mat_file, str) and mat_file.endswith(".mat"):
        mat_file = io.loadmat(mat_file)
    if not query:
        query = list(mat_file.keys())[-1]
    if isinstance(mat_file, dict) and query in mat_file.keys():
        mutations_dict: dict = {}
        for pdb, mutations in mat_file.get(query):
            # flatten nested data structure in the process
            m_ddg_tuples = [(m[0], a[0][0]) for m, a in mutations]
            mutations_dict[pdb[0]] = m_ddg_tuples
        return mutations_dict
    else:
        raise RuntimeError(f"Requested {query} data not in provided mat-file {mat_file}!")

def aa2index(aa):
    aa_array = np.array(["A", "R", "N", "D", "C", "Q", "E", "G", 
                        "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"])
    return np.where(aa_array == aa)[0][0]


class Variable:
    def __init__(self, v, lower, upper):
        self.unconstrained = self.inverse(v, lower, upper)
        # TODO: make sure unconstrained requires grad
        self.lower = lower
        self.upper = upper
        
    def get_unconstrained(self):
        return self.unconstrained
        
    def get_value(self):
        return self.constrain(self.unconstrained, self.lower, self.upper)

    @staticmethod
    def inverse(val, lower, upper):
        inverse = -torch.log( (upper-lower) / (val-lower) -1)
        inverse.requires_grad_(True)
        return inverse

    @staticmethod
    def constrain(val, lower, upper):
        """
        constrain through σ function
        """
        constrained = lower + (upper-lower) * (1 / (1 + torch.exp(-val)))
        constrained.requires_grad_(True)
        return constrained