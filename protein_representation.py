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
from utility import aa2index
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
        self.mutations_exp = mutations_exp.get(pdb_ID)
        self.mutations_sim = mutations_sim.get(pdb_ID)
        if TESTING:
            # reduce complexity by sampling simulations
            print(f"TESTING: Selecting {int(0.5*len(self.mutations_sim))} in silico mutations")
            self.mutations_sim = sample(self.mutations_sim, int(0.5*len(self.mutations_sim)))
        self.mutation_ids = ["WT"]
        self.ΔΔg = [0]
        self._kernels = KernelLoader()
        self.mut_S_exp, self.mut_adj_exp, self.mut_S_is, self.mut_adj_is = self.derive_mutations()
        self.mutated_sequences = self.mut_S_exp + self.mut_S_is
        # TODO mutated adjacencies not used downstream
        self.mutated_adjacencies = self.mut_adj_exp + self.mut_adj_is
        self.mut_ids_exp = self.mutation_ids[:len(self.mut_S_exp)]
        self.mut_ids_is = self.mutation_ids[len(self.mut_S_exp):]

        self.ΔΔg_exp = self.ΔΔg[:len(self.mut_S_exp)]
        self.ΔΔg_is = self.ΔΔg[len(self.mut_S_exp):]
        self.covariance_matrices: dict = self.compute_matrices()
        self.matrices_df: pd.DataFrame = self.generate_df_representation()
        
        self.scaler = None
        if scaling:
            self.scaler = BayesScaler(is_mutations=self.mut_ids_is, exp_mutations=self.mut_ids_exp, 
            ΔΔg=self.ΔΔg_is, experimentally_observed_ΔΔg=self.ΔΔg_exp)
            # overwrite in-silico data with scaled
            self.ΔΔg[len(self.mut_S_exp):] = self.scaler.transform(self.ΔΔg_is)

    def derive_mutations(self) -> Tuple[list, list, list, list]:
        """
        parses mutations from mutation_dictionaries
        mutates ΔΔg and mutation id as class properties
        """
        print("Parsing experimental mutations ...")
        mut_S_exp, mut_adj_exp = self._parse_mutations(self.mutations_exp)
        print("Parsing in silico mutations ...")
        mut_S_is, mut_adj_is = self._parse_mutations(self.mutations_sim)
        return mut_S_exp, mut_adj_exp, mut_S_is, mut_adj_is

    def _parse_mutations(self, mutation_dict) -> Tuple[list, list]:
        mutated_sequences = []
        mutated_adjacencies = []
        if not mutation_dict:
            print("WARNING: No mutations provided.")
            return mutated_sequences, mutated_adjacencies
        for (mutation, ddg) in tqdm(mutation_dict):
            # TODO rewrite this, return list instead, in-place side-effects are ugly
            self.ΔΔg.append(ddg)
            self.mutation_ids.append(mutation)
            # deepcopy to ensure that the underlying wildtype is not overwritten
            sequence = deepcopy(self.sequence)
            adjacency = deepcopy(self.adjacency)
            mutation_tuples = self._parse_and_assert_mutations(mutation)
            for _, idx, mut in mutation_tuples:
                sequence[idx] = mut
                # change imutable reference tuple by creating new tuple
                adjacency[idx] = (mut, adjacency[idx][1])
            mutated_sequences.append(sequence)
            mutated_adjacencies.append(adjacency)
        return mutated_sequences, mutated_adjacencies

    def _parse_and_assert_mutations(self, mutation) -> Tuple[str, int, str]:
        """
        decompose and handle mutliple mutations (not single point)
        """
        mutation_tuples = []
        s_mutations = re.split(r'(\d+)([A-Z])', mutation)[:-1]
        for i in range(0, len(s_mutations), 3):
            seq_res = s_mutations[i]
            seq_idx = int(s_mutations[i+1])-1 # offset - PDB-format counts from 1
            seq_mut = s_mutations[i+2]
            #assert self.sequence[seq_idx] == seq_res
            #assert self.contactmap.adjacency[seq_idx][0] == seq_res 
            mutation_tuples.append((seq_res, seq_idx, seq_mut))
        return mutation_tuples

    def compute_matrices(self) -> dict:
        """
        compute substitution over all mutations with one-another
        """
        print("Computing kernel matrices ...")
        ks_dict = {kernel: {} for kernel in self._kernels.sub_matrices_names}
        sequences = [self.sequence] + self.mutated_sequences
        sequences = np.array([np.array([aa2index(aa) for aa in seq], dtype=np.int64) for seq in sequences], dtype=np.int64)
        # TODO Changes in adjacencies are not accounted for !
        # adjacencies = [self.adjacency] + self.mutated_adjacencies
        adjacencies = self.adjacency
        for k_name, kernel in zip(self._kernels.sub_matrices_names, self._kernels.kernels):
            # N = wt+mutations
            ks_dict[k_name] = torch.Tensor(kernel.k(sequences, adjacencies))
        return ks_dict

    def generate_df_representation(self) -> pd.DataFrame:
        df_list = []
        for mat_type, matrix_kernel in self.covariance_matrices.items():
            mk_df = pd.DataFrame(matrix_kernel.detach().numpy(), columns=self.mutation_ids)
            df_list.append(mk_df)
        total_df = pd.DataFrame({'idx': self.covariance_matrices.keys(), 'mat': df_list})
        return total_df

    def plot_sub_matrices(self, savefig="./fig/"):
        # TODO plot only range of matrices (e.g. 10 mutations)
        filename = f"{savefig}/sub_matrices_{self.pdb_ID}.png"
        fig, ax = plt.subplots(1, len(self.covariance_matrices.items()), figsize=(30,20))
        for idx, mat in enumerate(self.matrices_df['mat']):
            matplot = ax[idx].imshow(mat.to_numpy())
            ax[idx].set_yticks(np.arange(len(mat.columns)))
            ax[idx].set_yticklabels(mat.columns, size=5)
            ax[idx].set_xticks(np.arange(len(mat.columns)))
            ax[idx].set_xticklabels(mat.columns, rotation=90, size=5)
            if idx > 0:
                ax[idx].set_yticks([])
            ax[idx].set_title("{}".format(self._kernels.sub_matrices_names[idx]))
        fig.colorbar(matplot, ax=ax[idx], fraction=0.046, pad=0.04)
        plt.savefig(filename)
        plt.legend()
        plt.show()

