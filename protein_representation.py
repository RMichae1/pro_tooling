import re
import numpy as np
import pandas as pd
from os.path import isfile
import pickle
from copy import deepcopy
from contact_mapper import ContactMapper
from graphkernel import MatrixKernel
from graphkernel import KernelFactory
from typing import Tuple
import matplotlib.pyplot as plt


class ProteinCollection:
    def __init__(self, contactmap: ContactMapper, pdb_ID: str, pdb_mutations: dict):
        self.pdb_ID = pdb_ID
        self.contactmap = contactmap
        self.adjacency = contactmap.adjacency
        self.sequence = contactmap.sequence
        assert self.pdb_ID == self.contactmap.pdb_ID
        self.mutation_dict = pdb_mutations.get(pdb_ID)
        self.mutation_ids = ["WT"]
        self.ΔΔg = [0]
        self.kernel_factory = KernelFactory()
        self.mutated_sequences, self.mutated_adjacencies = self.derive_mutations()
        # TODO For testing compute once and save result as pickle
        if not isfile('test_wdk.pickle'):
            self.wdk_kernels: dict = self.compute_wdk()
            with open('test_wdk.pickle', 'wb') as file_handle:
                pickle.dump(self.wdk_kernels, file_handle)
        else:
            with open('test_wdk.pickle', 'rb') as file_handle:
                self.wdk_kernels = pickle.load(file_handle) 
        # TODO remove after testing
        self.wdk_df: pd.DataFrame = self.generate_df_representation()
        # TODO write kernel output in df representation
        # TODO plot result

    def parse_and_assert_mutations(self, mutation) -> Tuple[str, int, str]:
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

    def derive_mutations(self) -> list:
        """
        parses mutations from mutation_dictionary
        mutates ΔΔg and mutation id as class properties
        """
        mutated_sequences = []
        mutated_adjacencies = []
        for (mutation, ddg) in self.mutation_dict:
            self.ΔΔg.append(ddg)
            self.mutation_ids.append(mutation)
            # deepcopy to ensure that the underlying wildtype is not overwritten
            sequence = deepcopy(self.sequence)
            adjacency = deepcopy(self.adjacency)
            mutation_tuples = self.parse_and_assert_mutations(mutation)
            for _, idx, mut in mutation_tuples:
                sequence[idx] = mut
                # change imutable reference tuple by creating new tuple
                adjacency[idx] = (mut, adjacency[idx][1])
            mutated_sequences.append(sequence)
            mutated_adjacencies.append(adjacency)
        return mutated_sequences, mutated_adjacencies

    def compute_wdk(self) -> dict:
        """
        evaluate all mutations with one-another
        """
        wdks = {kernel: {} for kernel in self.kernel_factory.sub_matrices}
        for k_name, kernel in zip(self.kernel_factory.sub_matrices, self.kernel_factory.kernels):
            sequences = [self.sequence] + self.mutated_sequences
            adjacencies = [self.adjacency] + self.mutated_adjacencies
            for idx, (p_seq, p_adj) in enumerate(zip(sequences, adjacencies)):
                print(f"MUTATION {self.mutation_ids[idx]}") # N = wt+mutations
                mat_vals = []
                for q_seq, q_adj in zip(sequences[idx:], adjacencies[idx:]):
                    # set object properties before computation
                    kernel.p_sequence = p_seq
                    kernel.q_sequence = q_seq
                    kernel.p_adjacency = p_adj
                    kernel.q_adjacency = q_adj
                    mat_vals.append(kernel.k())
                    # TODO invoke WDK instead of regular kernels
                wdks[k_name][self.mutation_ids[idx]] = mat_vals
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
        return df

    def generate_df_representation(self) -> pd.DataFrame:
        df_list = []
        for mat_type, matrix_kernel in self.wdk_kernels.items():
            mk_df = self.build_df_from_mk(matrix_kernel)
            df_list.append(mk_df)
        total_df = pd.DataFrame({'idx': self.wdk_kernels.keys(), 'mat': df_list})
        return total_df

    def plot_wdks(self):
        _, ax = plt.subplots(1, len(self.wdk_kernels.items()), figsize=(20,10))
        for idx, wdk in enumerate(self.wdk_df['mat']):
            ax[idx].imshow(wdk.values)
            ax[idx].set_yticks(np.arange(len(wdk.columns)))
            ax[idx].set_yticklabels(wdk.columns)
            ax[idx].set_xticks(np.arange(len(wdk.columns)))
            ax[idx].set_xticklabels(wdk.columns)
            plt.xticks(rotation=90)
            if idx > 0:
                ax[idx].set_yticks([])
            ax[idx].set_title("{}".format(self.kernel_factory.sub_matrices[idx]))
        plt.savefig("./fig/mat_viz.png")
        plt.show()
