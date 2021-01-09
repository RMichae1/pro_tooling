import os
import sys
from pathlib import Path
import numpy as np

from contact_mapper import ContactMapper
from graphkernel import MatrixKernel

# append gp modeling path for testing
sys.path.append("/home/rcml/gp_modeling/")
# from kernel.weighted_decomposition import WeightedDecomposition, ElementWiseKernel
# from kernel.MGPFusionKernel import MGPFusionKernel
# from load_data.load_pdb_id_data import get_preprocessed_training_and_test_data

# # TODO refactor global variables into build-up tear-down testing structure
# # init object instances
# cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/1pga.pdb", tri_dist=True)

# # test reference BLOSUM 62
# x_wild_type, X_wetlab, X_insilico, y_wild_type, y_wetlab, y_scaled, scaling_std, mean_y, max_y, contact_graph, \
#             X_test, y_test = get_preprocessed_training_and_test_data("1PGA")
# ref_mat = MGPFusionKernel(adjacency_graph=contact_graph)
# for m, n, _ in ref_mat.matrices:
#     if n == 'HENS920102': # ID for BLOSUM62
#         break
# gpm_wdk = WeightedDecomposition(substitution_matrix=m, contact_map=contact_graph)

# def test_reference_kernel_exists():
#     assert gpm_wdk is not None

def naive_K(seq, adj, S):
    N = seq.shape[0]
    K = np.zeros([N, N])
    for p in range(N):
        for q in range(N):
            for idx in range(seq.shape[1]):
                nbps = adj[idx]
                for l in nbps:
                    K[p, q] += S[seq[p, l], seq[q, l]]
                K[p, q] *= S[seq[p, idx], seq[q, idx]]
    # normalize
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            K[p, q] /= (np.sqrt(K[p, q]) * np.sqrt(K[p, q]))
    for n in range(0, N):
        K[n, n] = 1
    return K

def test_elementwise_kernel():
    mk = MatrixKernel(matrix=["BLOSUM62"])
    N = 50 # mutations
    L = 20 # sequence length
    AA = 20 # amino acids
    S = mk.matrix
    seqs = np.random.randint(0, AA, size=[N, L])
    adj = [np.random.randint(0, L, [np.random.randint(0, L)]) for _ in range(0, L)]
    augm_adj = [("X", val) for val in adj]
    k = mk.k(sequences=seqs, adjacencies=augm_adj)
    k_ref = naive_K(seq=seqs, adj=adj, S=S)
    np.testing.assert_almost_equal(k, k_ref)

# TODO test sub matrices against gp_modeling reference

# TODO test weighted mWDK against gp_modeling reference
