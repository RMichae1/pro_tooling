import re
from random import sample
import numpy as np
import pandas as pd
from os.path import isfile
import pickle
from copy import deepcopy
from contact_mapper import ContactMapper
from graphkernel import MatrixKernel, KernelLoader
from data_scaler import BayesScaler
from utility import aa2index, parse_mutations, convert_aa_sequence
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.distributions import Normal, Gamma
from torch.nn import Parameter
from data_scaler import BayesScaler


class ProteinCollection:
    """
    Class that captures protein properties Sequence, adjacecny, mutations and
    computed covariance matrices
    """
    def __init__(self, contactmap: ContactMapper, pdb_ID: str, 
            mutations_exp: dict={}, mutations_sim: dict={}, scaling=False, 
            TESTING=False):
        self.pdb_ID: str = pdb_ID
        self.contactmap = contactmap
        self.adjacency = contactmap.adjacency
        self.sequence = contactmap.sequence
        assert self.pdb_ID == self.contactmap.pdb_ID
        self._kernels = KernelLoader()
        self.mut_S_exp, self.mut_adj_exp, self.ΔΔg_exp, self.mut_ids_exp = parse_mutations(mutation_dict=mutations_exp.get(pdb_ID), 
                                                    sequence=self.sequence, adjacency=self.adjacency)
        self.mut_S_is, self.mut_adj_is, self.ΔΔg_is, self.mut_ids_is = parse_mutations(mutation_dict=mutations_sim.get(pdb_ID), 
                                                    sequence=self.sequence, adjacency=self.adjacency)
        self.mutated_sequences = self.mut_S_exp + self.mut_S_is
        self.mutated_adjacencies = self.mut_adj_exp + self.mut_adj_is

        self.scaler = None
        if scaling:
            self.scaler = BayesScaler(is_mutations=self.mut_ids_is, exp_mutations=self.mut_ids_exp, 
            ΔΔg=self.ΔΔg_is, experimentally_observed_ΔΔg=self.ΔΔg_exp)
            # overwrite in-silico data with scaled
            self.ΔΔg_is = self.scaler.transform(self.ΔΔg_is)
        self.mutation_ids = ["WT"] + self.mut_ids_exp + self.mut_ids_is
        self.ΔΔg = np.concatenate((np.array([0]), self.ΔΔg_exp, self.ΔΔg_is))

    def compute_matrices(self) -> dict:
        """
        compute substitution over all mutations with one-another
        """
        print("Computing kernel matrices ...")
        ks_dict = {kernel: {} for kernel in self._kernels.sub_matrices_ids}
        sequences = [self.sequence] + self.mutated_sequences
        sequences = convert_aa_sequence(sequences)
        for k_name, kernel in zip(self._kernels.sub_matrices_ids, self._kernels.kernels):
            # N = wt+mutations
            ks_dict[k_name] = torch.Tensor(kernel.k(sequences, self.adjacency))
        return ks_dict

    def generate_df_representation(self) -> pd.DataFrame:
        df_list = []
        covariance_matrices = self.compute_matrices()
        for mat_type, matrix_kernel in covariance_matrices.items():
            mk_df = pd.DataFrame(matrix_kernel.detach().numpy(), columns=self.mutation_ids)
            df_list.append(mk_df)
        total_df = pd.DataFrame({'idx': covariance_matrices.keys(), 'mat': df_list})
        return total_df

    def plot_sub_matrices(self, savefig: str="./fig/", per_row=5, plot_range=None):
        # TODO plot only range of matrices (e.g. 10 mutations)
        filename = f"{savefig}/sub_matrices_{self.pdb_ID}.png"
        fig, ax = plt.subplots(int(np.ceil(len(self._kernels.sub_matrices_ids)/per_row)), per_row, figsize=(8.27, 11.69))
        matrices_df = self.generate_df_representation()
        for idx, mat in enumerate(matrices_df['mat']):
            matplot = ax[idx].imshow(mat.to_numpy())
            ax[idx].set_yticks(np.arange(len(mat.columns)))
            ax[idx].set_yticklabels(mat.columns, size=5)
            ax[idx].set_xticks(np.arange(len(mat.columns)))
            ax[idx].set_xticklabels(mat.columns, rotation=90, size=5)
            if idx > 0:
                ax[idx].set_yticks([])
            ax[idx].set_title("{}".format(self._kernels.sub_matrices_ids[idx]))
        fig.colorbar(matplot, ax=ax[idx], fraction=0.046, pad=0.04)
        plt.savefig(filename)
        plt.legend()
        #plt.show()

