import re
from copy import deepcopy
from contact_mapper import ContactMapper
from graphkernel import WeightedDecompositionKernel
from typing import Tuple


class ProteinCollection:
    def __init__(self, contactmap: ContactMapper, pdb_ID: str, pdb_mutations: dict):
        self.pdb_ID = pdb_ID
        self.wt_contactmap = contactmap
        print(self.wt_contactmap)
        self.wt_sequence = self.wt_contactmap.sequence
        self.wt_ΔΔg = 0
        assert self.pdb_ID == self.wt_contactmap.pdb_ID
        self.mutation_dict = pdb_mutations.get(pdb_ID)
        self.X = []
        self.ΔΔg = []
        self.mutated_sequence, self.mutated_adjacency = self.derive_mutations()
        self.wdk_kernels = self.compute_wdk()

    def derive_mutations(self) -> list:
        mutated_sequences = []
        mutated_adjacencies = []
        for (mutation, ddg) in self.mutation_dict:
            self.ΔΔg.append(ddg)
            # ensure that the underlying wildtype is not overwritten
            sequence = deepcopy(self.wt_sequence)
            adjacency = deepcopy(self.wt_contactmap.adjacency)
            split_mutation_str = re.split(r'(\d+)([A-Z])', mutation)[:-1]
            #print(split_mutation_str)
            for i in range(0, len(split_mutation_str), 3):
                seq_res = split_mutation_str[i]
                seq_idx = int(split_mutation_str[i+1]) - 1 # PDB-format counts from 1
                seq_mut = split_mutation_str[i+2]
                assert self.wt_sequence[seq_idx] == seq_res
                assert self.wt_contactmap.adjacency[seq_idx][0] == seq_res 
                print(f"mutating pos {seq_idx} to {seq_mut}")
                print(f"mutating adj ref {seq_idx} from {self.wt_contactmap.adjacency[seq_idx][0]} to {seq_mut}")
                sequence[seq_idx] = seq_mut
                # change reference tuple
                adjacency[seq_idx] = (seq_mut, adjacency[seq_idx][1])
            mutated_sequences.append(sequence)
            mutated_adjacencies.append(adjacency)
        return mutated_sequences, mutated_adjacencies

    def compute_wdk(self):
        wdks = []
        for mutant_seq, mutant_adj in zip(self.mutated_sequence, self.mutated_adjacency):
            wt_seq = self.wt_contactmap.sequence
            wt_adj = self.wt_contactmap.adjacency
            wdk = WeightedDecompositionKernel(p_sequence=wt_seq, p_adjacency=wt_adj, 
                                        q_sequence=mutant_seq, q_adjacency=mutant_adj)
            wdks.append(wdk)
        return wdks
