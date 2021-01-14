import re
from os import path
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
from torch.distributions import Gamma
from scipy import io
from typing import List, Tuple


def parse_matlab_mutation_file(mat_file, query: str=None) -> dict:
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


def parse_and_assert_mutations(mutation: str) -> Tuple[str, int, str]:
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


def parse_mutations(sequence: str, adjacency: List[tuple], mutation_dict: dict) -> Tuple[list, list]:
    mutated_sequences = []
    mutated_adjacencies = []
    ΔΔg = []
    mutation_ids = []
    if not mutation_dict:
        print("WARNING: No mutations provided.")
        return mutated_sequences, mutated_adjacencies
    for (mutation, ddg) in tqdm(mutation_dict):
        ΔΔg.append(ddg)
        mutation_ids.append(mutation)
        # deepcopy to ensure that the underlying wildtype is not overwritten
        sequence = deepcopy(sequence)
        adjacency = deepcopy(adjacency)
        mutation_tuples = parse_and_assert_mutations(mutation)
        for _, idx, mut in mutation_tuples:
            sequence[idx] = mut
            # change imutable reference tuple by creating new tuple
            adjacency[idx] = (mut, adjacency[idx][1])
        mutated_sequences.append(sequence)
        mutated_adjacencies.append(adjacency)
    return mutated_sequences, mutated_adjacencies, np.array(ΔΔg), mutation_ids


def preprocess_observations(y_wild_type, y_wetlab, y_scaled):
    y = np.vstack([y_wild_type, y_wetlab, y_scaled])
    mean_y = np.mean(y)
    y -= mean_y
    max_y = np.max(np.abs(y))
    y /= max_y
    return mean_y, max_y, y[[0], :], y[1:y_wetlab.shape[0]+1, :], y[1+y_wetlab.shape[0]:, :]


def aa2index(aa):
    aa_array = np.array(["A", "R", "N", "D", "C", "Q", "E", "G", 
                        "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"])
    return np.where(aa_array == aa)[0][0]


def convert_aa_sequence(sequences: list):
    return np.array([np.array([aa2index(aa) for aa in seq], dtype=np.int64) for seq in sequences], dtype=np.int64)


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
        inverse.type(torch.float64)
        inverse.requires_grad_(True)
        return inverse

    @staticmethod
    def constrain(val, lower, upper):
        """
        constrain through σ function
        """
        constrained = lower + (upper-lower) * (1 / (1 + torch.exp(-val)))
        constrained.type(torch.float64)
        constrained.requires_grad_(True)
        return constrained