import re
import os
import warnings
from os import path
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy
from scipy import io
import pandas as pd
from typing import List, Tuple
from Bio.Seq import Seq
import pickle


#######
### EXTERNAL UTILS

def get_split_training_and_test_data(pdb_id: str, cutoff_distance: float, p=None):
    x_wild_type, y_wild_type, X_wetlab, y_wetlab, X_insilico, y_insilico, matching_mutations, contact_graph = \
        load_pdb_id_data(pdb_id, cutoff_distance=cutoff_distance)

    if p is None:
        p = np.random.permutation(X_wetlab.shape[0])
    assert (p.shape[0] == X_wetlab.shape[0])
    X_test = X_wetlab[p[:20], :]  # 20 data points from the wetlab experiments are withheld for testing
    y_test = y_wetlab[p[:20], :]
    # matching_mutations = np.hstack([matching_mutations[p[20:], [0]], matching_mutations[20:, [1]]])  # TODO: does this work? seems so
    matching_mutations[:, 0] = p[matching_mutations[:, 0]]
    matching_mutations = matching_mutations[20:, :]
    y_train_wetlab_matching = y_wetlab[matching_mutations[:, 0], :]  # observations stem only from the training set
    y_insilico_matching = y_insilico[matching_mutations[:, 1], :]
    X_wetlab = X_wetlab[p[20:], :]
    y_wetlab = y_wetlab[p[20:], :]
    return contact_graph, x_wild_type, y_wild_type, X_wetlab, y_wetlab, X_insilico, y_insilico, y_train_wetlab_matching, \
           y_insilico_matching, X_test, y_test


def load_pdb_id_data(pdb_id: str, cutoff_distance=5.):
    wild_type, contact_graph = get_sequence_and_contact_graph(pdb_id=pdb_id, cutoff_distance=cutoff_distance,
                                                              chain_id=None)
    x_wild_type, X_wetlab, y_wetlab, X_insilico, y_insilico, matching_mutations = load_mutations(pdb_id, wild_type)
    return x_wild_type, 0., X_wetlab, y_wetlab, X_insilico, y_insilico, matching_mutations, contact_graph


def load_mutations(pdb_id: str, wild_type: Seq):
    data_dir = os.path.dirname(__file__)

    x_wild_type = seq2int(str(wild_type)) # CORRECT aa2idx label-encoding

    wetlab_results_file = os.path.join("data/mgp", "ddg_protherm.mat")
    wetlab_mat = scipy.io.loadmat(os.path.join(data_dir, wetlab_results_file))['ddg_protherm']
    id = -1
    for i in range(wetlab_mat.shape[0]):
        if wetlab_mat[i, 0] == pdb_id:
            id = i
            break
    y_wetlab = np.concatenate(wetlab_mat[id, 1][:, 1]).ravel()[:, np.newaxis]
    X_wetlab, single_wl_mutations, single_mutations_idx = apply_wetlab_mutations(wild_type, x_wild_type,
                                                                                 wetlab_mat[id, 1][:, 0])

    insilico_results_file = os.path.join("data/mgp", "ddg_rosetta_single.mat")
    mat = scipy.io.loadmat(os.path.join(data_dir, insilico_results_file))["ddg_rosetta_single"]
    y_insilico = np.concatenate(mat[id, 1][:, 1]).ravel()[:, np.newaxis]
    X_insilico, matching_mutations = apply_insilico_mutations(wild_type, x_wild_type, mat[id, 1][:, 0],
                                                              single_wetlab_mutations=single_wl_mutations,
                                                              single_mutations_idx=single_mutations_idx)

    # it appears that the matlab source code does not use the simulation results from multiple mutations

    return x_wild_type, X_wetlab, y_wetlab, X_insilico, y_insilico, matching_mutations


def apply_wetlab_mutations(wild_type, x_wildtype, mutations):
    """

    :param wild_type:
    :param x_wildtype:
    :param mutations:
    :return:
        X
        single_mutations: for each sequence position, a list of the mutations applied in that position (single mutation)
        single_mutations_idx: the corresponding index in X
    """
    X = np.tile(x_wildtype, [len(mutations), 1])
    single_mutations = {}
    single_mutations_idx = {}
    for i, m in enumerate(mutations):
        mutations, idx_, mutation = apply_mutation(X, i, m[0], wild_type)
        if mutations == 1:
            if single_mutations.get(idx_) is None:
                single_mutations[idx_] = []
                single_mutations_idx[idx_] = []
            single_mutations[idx_].append(mutation)
            single_mutations_idx[idx_].append(i)
    return X, single_mutations, single_mutations_idx


def apply_insilico_mutations(wild_type, x_wildtype, mutations, single_wetlab_mutations, single_mutations_idx):
    """

    :param wild_type:
    :param x_wildtype:
    :param mutations:
    :param single_wetlab_mutations:
    :param single_mutations_idx:
    :return:
        X
        a two-dimensional array of indices in X_wetlab and X that correspond to the same mutation
    """
    X = np.tile(x_wildtype, [len(mutations), 1])
    matching_mutations = []
    for i, m in enumerate(mutations):
        mutations, idx_, mutation = apply_mutation(X, i, m[0], wild_type)
        if mutations == 1:
            if single_wetlab_mutations.get(idx_) is not None:
                for j, m_ in enumerate(single_wetlab_mutations.get(idx_)):
                    if mutation == m_:
                        matching_mutations.append((single_mutations_idx.get(idx_)[j], i))
                        break
    return X, np.array(matching_mutations)


def apply_mutation(X, i, m, wild_type):
    # m is a string of the format: wild-type amino acid, position, mutated amino acid
    j = 0
    mutations = 0
    while j < len(m):
        ref_aa = m[j]
        idx = []
        j += 1
        while m[j].isnumeric():
            idx.append(m[j])
            j += 1
        mutation = m[j]
        j += 1

        idx_ = int(''.join(idx)) - 1
        assert (wild_type[idx_] == ref_aa)
        X[i, idx_] = aa2int(mutation)
        mutations += 1
    return mutations, idx_, mutation


alphabet = "ARNDCQEGHILKMFPSTWYV-"
# map amino acids to integers (A->0, R->1, etc)
a2n = dict((a, n) for n, a in enumerate(alphabet))


def aa2int(x: str):
    return a2n.get(x, a2n['-'])


def seq2int(seq: str):
    int_seq = np.zeros(len(seq), dtype=int)
    for i, s in enumerate(seq):
        int_seq[i] = aa2int(s)
    return int_seq


def preprocess_observations(y_wild_type, y_wetlab, y_scaled):
    y = np.vstack([y_wild_type, y_wetlab, y_scaled])
    mean_y = np.mean(y)
    y -= mean_y
    max_y = np.max(np.abs(y))
    y /= max_y
    return mean_y, max_y, y[[0], :], y[1:y_wetlab.shape[0] + 1, :], y[1 + y_wetlab.shape[0]:, :]


def list_of_pairs_2_seq(list):
    seq_str = []
    last_i = -1
    for i, s in list:
        if i - last_i > 1:
            raise RuntimeError("Parsed Sequence has gaps!")
        last_i = i
        seq_str.append(s)
    return Seq(''.join(seq_str))


def get_sequence_and_contact_graph_from_ref_matlab_file(pdb_id: str, cutoff_distance=5., chain_id=None) -> (Seq, list):
    if not cutoff_distance == 5.:
        raise RuntimeError("The matlab reference files have a fixed cutoff distance of 5 angstrom!")
    data_dir = os.path.join(os.path.dirname(__file__), "data/mgp")
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


#########
### UTILS FROM THIS PROJECT:

def get_mutation_idx(mutation_ids: list) -> list:
    idx_lst = [re.findall(r'\d+', mut) for mut in mutation_ids]
    int_idx_lst = [[int(idx) for idx in sublist] for sublist in idx_lst]
    return int_idx_lst


def parse_matlab_mutation_file(mat_file, query: str = None) -> dict:
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
        seq_idx = int(s_mutations[i + 1]) - 1  # offset - matlab/PDB-format counts from 1
        seq_mut = s_mutations[i + 2]
        # assert self.sequence[seq_idx] == seq_res
        # assert self.contactmap.adjacency[seq_idx][0] == seq_res
        mutation_tuples.append((seq_res, seq_idx, seq_mut))
    return mutation_tuples


def parse_mutations(sequence: str, adjacency: List[tuple], mutation_dict: dict) -> Tuple[list, list, np.ndarray, list]:
    mutated_sequences = []
    mutated_adjacencies = []
    ΔΔg = []
    mutation_ids = []
    if not mutation_dict:
        print("WARNING: No mutations provided.")
        return mutated_sequences, mutated_adjacencies, np.array(ΔΔg), mutation_ids
    for (mutation, ddg) in tqdm(mutation_dict):
        ΔΔg.append(float(ddg))
        # deepcopy to ensure that the underlying wildtype is not overwritten
        seq = deepcopy(sequence)
        adj = deepcopy(adjacency)
        mutation_tuples = parse_and_assert_mutations(mutation)
        mutation_string = "".join([str(t) for t in mutation_tuples])
        mutation_ids.append(mutation_string)
        for _, idx, mut in mutation_tuples:
            seq[idx] = mut
            # change imutable reference tuple by creating new tuple
            adj[idx] = (mut, adjacency[idx][1])
        mutated_sequences.append(seq)
        mutated_adjacencies.append(adj)
    return mutated_sequences, mutated_adjacencies, np.array(ΔΔg), mutation_ids


def index2aa(idx):
    aa_lookup = {k: idx for idx, k in enumerate(["A", "R", "N", "D", "C", "Q", "E", "G",
                                                 "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"])}
    return list(aa_lookup)[idx]

def aa2index(aa):
    """
    ORDER OF THE ALPHABET MATTERS !!
    """
    aa_lookup = {k: idx for idx, k in enumerate(["A", "R", "N", "D", "C", "Q", "E", "G",
                                                 "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"])}
    return aa_lookup.get(aa, 21)  # default value is unknown


def convert_aa_sequence(sequences: list):
    return np.array([np.array([aa2index(aa) for aa in seq], dtype=np.int64) for seq in sequences], dtype=np.int64)


def compute_ρ(y_vec: np.ndarray, y_pred_μ: np.ndarray) -> float:
    """
    ρ computation as defined in (S7)
    """
    pred_μ = np.mean(y_pred_μ)
    exp_μ = np.mean(y_vec)
    ρ = np.sum((y_vec - exp_μ) * (y_pred_μ - pred_μ))
    norm = np.sqrt(np.sum((y_vec - exp_μ) ** 2) * np.sum((y_pred_μ - pred_μ) ** 2))
    ρ /= norm
    return ρ


def compute_rmse(y: np.ndarray, y_pred_μ) -> float:
    """
    RMSE computation as defined in (S8)
    """
    n_obs = y.shape[0]
    rmse = np.sqrt(np.sum((y - y_pred_μ) ** 2) / n_obs)
    return rmse


def one_hot_encoding(arr: np.ndarray) -> np.ndarray:
    oh_arr = np.zeros((arr.size, arr.max() + 1))
    oh_arr[np.arange(arr.size), arr] = 1
    return oh_arr


def oh_to_le_encoding(arr):
    return np.argmax(arr[0])


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
        inverse = -torch.log((upper - lower) / (val - lower) - 1)
        inverse.type(torch.float64)
        inverse.requires_grad_(True)
        return inverse

    @staticmethod
    def constrain(val, lower, upper):
        """
        constrain through σ function
        """
        constrained = lower + (upper - lower) * (1 / (1 + torch.exp(-val)))
        constrained.type(torch.float64)
        constrained.requires_grad_(True)
        return constrained


def seq_collate(tensor):
    one_hot_sequence, weights, neff = zip(*tensor)
    return torch.stack(one_hot_sequence), torch.stack(weights), neff[0]


class WeightedMSADataset(Dataset):
    """Weighted MSA"""

    def __init__(self, encoded_sequence, num_classes, weight_batch_size=1000, dtype=torch.float):
        self.encoded_sequence = torch.Tensor(encoded_sequence).to(torch.int64)
        self.one_hot_sequence = F.one_hot(self.encoded_sequence,
                                          num_classes=num_classes).to(dtype)
        # Calculate weights
        weights = []
        flat_one_hot = self.one_hot_sequence.flatten(1)
        gap_code = aa2index("-")  # CORRECT AA LOOKUP HERE
        for i in range(self.one_hot_sequence.size(0) // weight_batch_size + 1):
            x = flat_one_hot[i * weight_batch_size: (i + 1) * weight_batch_size]
            similarities = torch.mm(x, flat_one_hot.T)
            lengths = (self.encoded_sequence[i * weight_batch_size: (i + 1) * weight_batch_size] != gap_code).sum(
                1).unsqueeze(-1)
            w = 1.0 / (similarities / lengths).gt(0.8).sum(1).float()
            weights.append(w)
        self.weights = torch.cat(weights)
        self.neff = self.weights.sum()

    def __len__(self):
        return self.encoded_sequence.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.one_hot_sequence[index], self.weights[index], self.neff
