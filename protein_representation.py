import re
import numpy as np
from copy import deepcopy
from contact_mapper import ContactMapper
from graphkernel import WeightedDecompositionKernel
from typing import Tuple


class ProteinCollection:
    def __init__(self, contactmap: ContactMapper, pdb_ID: str, pdb_mutations: dict):
        self.pdb_ID = pdb_ID
        self.wt_contactmap = contactmap
        self.wt_adjacency = self.wt_contactmap.adjacency
        self.wt_sequence = self.wt_contactmap.sequence
        self.wt_ΔΔg = 0
        assert self.pdb_ID == self.wt_contactmap.pdb_ID
        self.mutation_dict = pdb_mutations.get(pdb_ID)
        self.mutation_ids = []
        self.ΔΔg = []
        self.mutated_sequences, self.mutated_adjacencies = self.derive_mutations()
        self.wdk_kernels: dict = self.compute_wdk()

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
            print(f"MUTATION {self.mutation_ids[idx]}")
            sub_wdks = []
            for q_seq, q_adj in zip(eval_sequences[idx:], eval_adjacencies[idx:]):
                wdk = WeightedDecompositionKernel(p_sequence=p_seq, p_adjacency=p_adj, 
                                        q_sequence=q_seq, q_adjacency=q_adj)
                sub_wdks.append(wdk)
            print(f"SUB WDKS {sub_wdks}")
            wdks[self.mutation_ids[idx]] = sub_wdks
        return wdks
