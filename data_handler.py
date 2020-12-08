from os import path
import numpy as np
from numpy.random import multivariate_normal
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
    

class DataModel:
    def __init__(pdb_id, σ_0=10e-6):
        self.pdb_id = pdb_id
        X_E, y_E = self.load_experimental_data()
        X_S, y_S = self.load_simulated_data()
        x_0, y_0 = self.load_wildtype_data()
        scaled_y_S = self.apply_bayesian_scaling(y_S)
        self.N = 1 + len(X_E) + len(X_S)
        self.X = (x_0, X_E, X_S)
        self.y = (y_0, y_E, scaled_y_S)
        self.σ_E = Gamma(α_E, β_E) # TODO how to assign priors in Gamma distribution?
        self.σ_S = Gamma(α_S, β_S)
        self.ε_0 = multivariate_normal(0, σ_0)
        self.ε_E = multivariate_normal(0, self.σ_E)
        self.ε_S = multivariate_normal(0, self.σ_S)

    def load_experimental_data(self):
        pass

    def load_simulated_data(self):
        pass

    def load_wildtype_data(self):
        pass

    def apply_bayesian_scaling(self, y):
        # TODO implement Bayesian Scaling W 51
        return y


if __name__ == "__main__":
    # some test code
    #ddg_web_mat = load_mat_file("./data/ddg_web.mat")
    #ddg_rosetta_multi_mat = load_mat_file("./data/ddg_rosetta_multi.mat")
    protherm_parsed_mutations = parse_mutations("./data/ddg_protherm.mat", query="ddg_protherm")
    rosetta_parsed_mutations = parse_mutations("./data/ddg_rosetta_single.mat")
    print(list(protherm_parsed_mutations.keys())[0])
    print(list(protherm_parsed_mutations.values())[0])
    print(rosetta_parsed_mutations.keys())