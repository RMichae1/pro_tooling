import re
import numpy as np
import pandas as pd
from copy import deepcopy
from contact_mapper import ContactMapper
from graphkernel import MatrixKernel
from typing import Tuple
import matplotlib.pyplot as plt


class ProteinCollection:
    def __init__(self, contactmap: ContactMapper, pdb_ID: str, pdb_mutations: dict):
        self.pdb_ID = pdb_ID
        self.wt_contactmap = contactmap
        self.wt_adjacency = self.wt_contactmap.adjacency
        self.wt_sequence = self.wt_contactmap.sequence
        assert self.pdb_ID == self.wt_contactmap.pdb_ID
        self.mutation_dict = pdb_mutations.get(pdb_ID)
        self.mutation_ids = ["WT"]
        self.ΔΔg = [0]
        self.mutated_sequences, self.mutated_adjacencies = self.derive_mutations()
        self.wdk_kernels: dict = self.compute_wdk()
        self.wdk_df: pd.DataFrame = self.generate_df_representation() 

    def parse_and_assert_mutations(self, mutation) -> Tuple[str, int, str]:
        mutation_tuples = []
        s_mutations = re.split(r'(\d+)([A-Z])', mutation)[:-1]
        for i in range(0, len(s_mutations), 3):
            seq_res = s_mutations[i]
            seq_idx = int(s_mutations[i+1])-1 # PDB-format counts from 1
            seq_mut = s_mutations[i+2]
            assert self.wt_sequence[seq_idx] == seq_res
            assert self.wt_contactmap.adjacency[seq_idx][0] == seq_res 
            mutation_tuples.append((seq_res, seq_idx, seq_mut))
        return mutation_tuples

    def derive_mutations(self) -> list:
        mutated_sequences = []
        mutated_adjacencies = []
        for (mutation, ddg) in self.mutation_dict:
            self.ΔΔg.append(ddg)
            self.mutation_ids.append(mutation)
            # ensure that the underlying wildtype is not overwritten
            sequence = deepcopy(self.wt_sequence)
            adjacency = deepcopy(self.wt_contactmap.adjacency)
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
        wdks = {}
        eval_sequences = [self.wt_sequence] + self.mutated_sequences
        eval_adjacencies = [self.wt_adjacency] + self.mutated_adjacencies
        for idx, (p_seq, p_adj) in enumerate(zip(eval_sequences, eval_adjacencies)):
            print(f"MUTATION {self.mutation_ids[idx]}") # N = wt+mutations
            sub_wdks = []
            for q_seq, q_adj in zip(eval_sequences[idx:], eval_adjacencies[idx:]):
                wdk = MatrixKernel(p_sequence=p_seq, p_adjacency=p_adj, 
                                        q_sequence=q_seq, q_adjacency=q_adj)
            wdks[self.mutation_ids[idx]] = wdk
        return wdks

    def extract_sub_matrix(self):
        """
        For df representation convert MUT: MAT: k_val to MAT: MUT: k_val
        """
        matrix_extract = {mat: [] for mat in self.wdk_kernels.keys()}
        for mut, wdk in self.wdk_kernels.items():
            for mat, values in wdk.items():
                print(values)
                m_vals = [val.numpy() for val in values]
            flat_vals = [val for sub in m_vals for val in sub]
            matrix_extract[mat][mut] = flat_vals
        print(matrix_extract)
        return matrix_extract

    def build_df_from_mk(seld, matrix_kernel) -> pd.DataFrame:
        df = pd.DataFrame(0, index=self.mutation_ids, columns=self.mutation_ids)
        for wdk, val in matrix_kernel.kernel.items():
            ref_len = max(map(len, matrix_kernel.values()))
            zero_padded = [0 for _ in range(ref_len-len(val))]
            data_row = zero_padded + val
            df.loc[wdk] = data_row
        return df


    def generate_df_representation(self) -> pd.DataFrame:
        df_list = []
        matrix_kernels = self.extract_sub_matrix()
        for matrix_kernel in matrix_kernels:
            mk_df = self.build_df_from_mk(matrix_kernel)
            df_list.append(mk_df)
        total_df = pd.DataFrame({'idx': np.arange(len(matrix_kernels)), 'mat': df_list})
        return total_df

    def plot_wdks(self):
        _, ax = plt.subplots(1,1, figsize=(15,10))
        ax.imshow(self.wdk_df.values)
        ax.set_yticks(np.arange(len(self.wdk_df.columns)))
        ax.set_yticklabels(self.wdk_df.columns)
        ax.set_xticks(np.arange(len(self.wdk_df.columns)))
        ax.set_xticklabels(self.wdk_df.columns)
        plt.title("BLOSUM62 Covariance Matrix")
        plt.show()
