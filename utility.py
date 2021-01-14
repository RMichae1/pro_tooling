import re
import os
import warnings
from os import path
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
from torch.distributions import Gamma
import scipy
from scipy import io
from typing import List, Tuple
from Bio.Seq import Seq


def get_split_training_and_test_data(pdb_id: str, cutoff_distance: float, p=None):
    x_wild_type, y_wild_type, X_wetlab, y_wetlab, X_insilico, y_insilico, matching_mutations, contact_graph =\
        load_pdb_id_data(pdb_id, cutoff_distance=cutoff_distance)

    if p is None:
        p = np.random.permutation(X_wetlab.shape[0])
    assert(p.shape[0] == X_wetlab.shape[0])
    X_test = X_wetlab[p[:20], :]  # 20 data points from the wetlab experiments are withheld for testing
    y_test = y_wetlab[p[:20], :]
    #matching_mutations = np.hstack([matching_mutations[p[20:], [0]], matching_mutations[20:, [1]]])  # TODO: does this work? seems so
    matching_mutations[:, 0] = p[matching_mutations[:, 0]]
    matching_mutations = matching_mutations[20:, :]
    y_train_wetlab_matching = y_wetlab[matching_mutations[:, 0], :]  # observations stem only from the training set
    y_insilico_matching = y_insilico[matching_mutations[:, 1], :]
    X_wetlab = X_wetlab[p[20:], :]
    y_wetlab = y_wetlab[p[20:], :]
    return contact_graph, x_wild_type, y_wild_type, X_wetlab, y_wetlab, X_insilico, y_insilico, y_train_wetlab_matching, \
           y_insilico_matching, X_test, y_test


def load_pdb_id_data(pdb_id: str, cutoff_distance=5.):
    wild_type, contact_graph = get_sequence_and_contact_graph(pdb_id=pdb_id, cutoff_distance=cutoff_distance, chain_id=None)
    x_wild_type, X_wetlab, y_wetlab, X_insilico, y_insilico, matching_mutations = load_mutations(pdb_id, wild_type)
    return x_wild_type, 0., X_wetlab, y_wetlab, X_insilico, y_insilico, matching_mutations, contact_graph


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
        seq = deepcopy(sequence)
        adj = deepcopy(adjacency)
        mutation_tuples = parse_and_assert_mutations(mutation)
        for _, idx, mut in mutation_tuples:
            seq[idx] = mut
            # change imutable reference tuple by creating new tuple
            adj[idx] = (mut, adjacency[idx][1])
        mutated_sequences.append(seq)
        mutated_adjacencies.append(adj)
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
                        "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"])
    return np.where(aa_array == aa)[0][0]


def convert_aa_sequence(sequences: list):
    return np.array([np.array([aa2index(aa) for aa in seq], dtype=np.int64) for seq in sequences], dtype=np.int64)


def list_of_pairs_2_seq(list):
    seq_str = []
    last_i = -1
    for i, s in list:
        if i - last_i > 1:
            raise RuntimeError("Parsed Sequence has gaps!")
        last_i = i
        seq_str.append(s)
    return Seq(''.join(seq_str))


def get_sequence_and_contact_graph_from_ref_matlab_file(pdb_id: str,  cutoff_distance=5., chain_id=None) -> (Seq, list):
    if not cutoff_distance == 5.:
        raise RuntimeError("The matlab reference files have a fixed cutoff distance of 5 angstrom!")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filename = pdb_id + '.mat'
    mat = scipy.io.loadmat(os.path.join(data_dir, filename))
    ref_sequence = np.squeeze(mat['sequence']['letters'])
    seq = []
    for i, aa in enumerate(ref_sequence):
        seq.append((i, aa[0]))
    return list_of_pairs_2_seq(seq), convert_graph_from_matlab_file(mat['contact_map'])


def convert_graph_from_matlab_file(al):
    contact_map = []
    for ns in al:
        contact_map.append(np.array(ns[0][0]) - 1)
    return contact_map


def get_sequence_and_contact_graph(*args, **kwargs):
    warnings.warn("Using local MATLAB files to load data.")
    return get_sequence_and_contact_graph_from_ref_matlab_file(*args, **kwargs)


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