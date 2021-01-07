import re
from random import sample
import numpy as np
import pandas as pd
from os.path import isfile
import pickle
from copy import deepcopy
from contact_mapper import ContactMapper
from graphkernel import MatrixKernel, WeightedDecompositionKernel
from graphkernel import KernelLoader
from data_scaler import BayesScaler
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.distributions import Normal, Gamma
from torch.nn import Parameter
from sklearn.preprocessing import StandardScaler ## TODO take out this quickfix

scaler = StandardScaler()

class ProteinCollection:
    """
    Class that captures protein properties and observed values,
    computed covariance matrices
    """
    def __init__(self, contactmap: ContactMapper, pdb_ID: str, 
            mutations_exp: dict={}, mutations_sim: dict={}, TESTING=False):
        self.pdb_ID = pdb_ID
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
        # TODO testing with one sub-matrix
        self.kernels = KernelLoader(sub_matrices=["BLOSUM62", "BLOSUM45"])
        self.mut_S_exp, self.mut_adj_exp, self.mut_S_is, self.mut_adj_is = self.derive_mutations()
        self.mutated_sequences = self.mut_S_exp + self.mut_S_is
        self.mutated_adjacencies = self.mut_adj_exp + self.mut_adj_is
        self.ΔΔg_exp = self.ΔΔg[:len(self.mut_S_exp)]
        self.ΔΔg_is = self.ΔΔg[len(self.mut_S_exp):]
        # TODO For testing compute once and save result as pickle
        if not isfile(f'test_wdk_{self.pdb_ID}.pickle'):
            self.matrix_kernels: dict = self.compute_matrices()
            with open(f'test_wdk_{self.pdb_ID}.pickle', 'wb') as file_handle:
                pickle.dump(self.matrix_kernels, file_handle)
        else:
            with open(f'test_wdk_{self.pdb_ID}.pickle', 'rb') as file_handle:
                self.matrix_kernels = pickle.load(file_handle)
        #self.matrix_kernels = self.compute_matrices()
        self.matrices_df: pd.DataFrame = self.generate_df_representation()
        self.mWDK = WeightedDecompositionKernel(kernels=self.matrices_df)
        mwdk_values = self.mWDK.K_ϕ.detach().numpy()
        # TODO scaling quickfix
        scaler.fit(mwdk_values)
        scaled_mwdk_values = scaler.transform(mwdk_values)
        ##
        self.mwdk_df: pd.DataFrame = pd.DataFrame(scaled_mwdk_values, index=self.mutation_ids, columns=self.mutation_ids)

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
            assert self.sequence[seq_idx] == seq_res
            assert self.contactmap.adjacency[seq_idx][0] == seq_res 
            mutation_tuples.append((seq_res, seq_idx, seq_mut))
        return mutation_tuples

    def compute_matrices(self) -> dict:
        """
        compute substitution over all mutations with one-another
        """
        print("Computing kernel matrices ...")
        wdks = {kernel: {} for kernel in self.kernels.sub_matrices_names}
        sequences = [self.sequence] + self.mutated_sequences
        adjacencies = [self.adjacency] + self.mutated_adjacencies
        for k_name, kernel in zip(self.kernels.sub_matrices_names, self.kernels.kernels):
            # N = wt+mutations
            wdks[k_name] = {self.mutation_ids[idx]: 
                [kernel.k(p_sequence=p_seq, q_sequence=q_seq, p_adjacency=p_adj, q_adjacency=q_adj)
                for q_seq, q_adj in zip(sequences[idx:], adjacencies[idx:])]
                    for idx, (p_seq, p_adj) in tqdm(enumerate(zip(sequences, adjacencies)))}
        return wdks

    def build_df_from_mk(self, matrix_kernel) -> pd.DataFrame:
        """
        Build diagonal matrix from the provided matrix_kernel.
        Account for diagonal through zero-padding as a difference from max.
        """
        df = pd.DataFrame(0, index=self.mutation_ids, columns=self.mutation_ids)
        for mutation, val in matrix_kernel.items():
            ref_len = max(map(len, matrix_kernel.values()))
            zero_padded = [0 for _ in range(ref_len-len(val))]
            data_row = zero_padded + val
            df.loc[mutation] = data_row
        val_mat = df.to_numpy()
        # complete matrix from upper triangle
        complete_mat = val_mat + val_mat.T
        # overwrite diagonal values, since they are doubled
        idx = np.arange(val_mat.shape[0])
        complete_mat[idx, idx] = val_mat[idx, idx]
        df.iloc[:, :] = complete_mat
        return df

    def generate_df_representation(self) -> pd.DataFrame:
        df_list = []
        for mat_type, matrix_kernel in self.matrix_kernels.items():
            mk_df = self.build_df_from_mk(matrix_kernel)
            df_list.append(mk_df)
        total_df = pd.DataFrame({'idx': self.matrix_kernels.keys(), 'mat': df_list})
        return total_df

    def plot_sub_matrices(self):
        _, ax = plt.subplots(1, len(self.matrix_kernels.items()), figsize=(30,20))
        for idx, wdk in enumerate(self.matrices_df['mat']):
            ax[idx].imshow(wdk.to_numpy())
            ax[idx].set_yticks(np.arange(len(wdk.columns)))
            ax[idx].set_yticklabels(wdk.columns, size=5)
            ax[idx].set_xticks(np.arange(len(wdk.columns)))
            ax[idx].set_xticklabels(wdk.columns, rotation=90, size=5)
            if idx > 0:
                ax[idx].set_yticks([])
            ax[idx].set_title("{}".format(self.kernels.sub_matrices_names[idx]))
        plt.savefig("./fig/mat_viz.png")
        plt.legend()
        plt.show()

    def plot_mwdk(self):
        _, ax = plt.subplots(1, 1, figsize=(20, 10))
        im = ax.imshow(self.mwdk_df.to_numpy())
        ax.set_yticks(np.arange(len(self.mwdk_df.columns)))
        ax.set_xticks(np.arange(len(self.mwdk_df.columns)))
        ax.set_yticklabels(self.mwdk_df.columns, size=5)
        ax.set_xticklabels(self.mwdk_df.columns, rotation=90, size=5)
        ax.set_title("mWDK values")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("", rotation=-90, va="bottom")
        plt.savefig("./fig/mwdk.png")
        plt.show()


class ProteinCollectionSimulated(ProteinCollection):
    """
    Subclass of ProteinCollection for (scaled) Rosetta simulated input
    """
    def __init__(self, contactmap: ContactMapper, pdb_ID: str, pdb_mutations: dict):
        super().__init__(contactmap, pdb_ID, pdb_mutations)
        self.scaler = BayesScaler(self.ΔΔg)
        self.ΔΔg = self.scaler.y
        

class AdditiveNoiseRepresentation:
    def __init__(self, protein_representation: ProteinCollection, σ_0=1e-6, 
                α_E=2.5, β_E=0.02, α_S=50., β_S=0.007):
        # TODO find out how .sample needs to be called...
        self.ε_0 = Normal(0, torch.tensor(σ_0)).sample()
        self.σ_experimental = Gamma(torch.tensor(α_E), torch.tensor(β_E)).sample()
        self.σ_simulated = Gamma(torch.tensor(α_S), torch.tensor(β_S)).sample()
        if protein_representation.__class__.__name__ == "ProteinCollection":
            self.σ = self.σ_experimental
        elif protein_representation.__class__.__name__ == "ProteinCollectionSimulated":
            σ_T = protein_representation.scaler.σ_T
            t = Parameter(1.1) # init t-value
            self.σ = self.σ_experimental + self.σ_simulated + t*σ_T
        else:
            raise RuntimeError("Protein Collection needs to be of type : ProteinCollection or ProteinCollectionSimulated !")
        self.ε = Normal(0, self.σ)
        self.y_WT = np.array(protein_representation.ΔΔg[0]) + self.ε_0.numpy()
        self.y = np.array(protein_representation.ΔΔg[1:])
        ε_exp = np.array([self.ε.sample() for _ in range(len(self.y))])
        self.y += ε_exp
        self.y = np.append(self.y_WT, self.y)


