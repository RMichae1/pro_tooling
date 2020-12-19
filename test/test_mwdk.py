import os
import sys
from pathlib import Path
import numpy as np

from contact_mapper import ContactMapper
from graphkernel import MatrixKernel

# append gp modeling path for testing
sys.path.append("/home/rcml/gp_modeling/")
from kernel.weighted_decomposition import WeightedDecomposition, ElementWiseKernel
from kernel.MGPFusionKernel import MGPFusionKernel
from load_data.load_pdb_id_data import get_preprocessed_training_and_test_data

# TODO refactor global variables into build-up tear-down testing structure
# init object instances
cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/1pga.pdb", tri_dist=True)
wdk = MatrixKernel(p_sequence=cm_tri.sequence, p_adjacency=cm_tri.adjacency,
                                    q_sequence=cm_tri.sequence, q_adjacency=cm_tri.adjacency, sub_matrix="BLOSUM62")

# test reference BLOSUM 62
x_wild_type, X_wetlab, X_insilico, y_wild_type, y_wetlab, y_scaled, scaling_std, mean_y, max_y, contact_graph, \
            X_test, y_test = get_preprocessed_training_and_test_data("1PGA")
ref_mat = MGPFusionKernel(adjacency_graph=contact_graph)
for m, n, _ in ref_mat.matrices:
    if n == 'HENS920102': # ID for BLOSUM62
        break
gpm_wdk = WeightedDecomposition(substitution_matrix=m, contact_map=contact_graph)


def test_mat_kernel_to_contactmap():
    # are the strings the same?
    assert "".join(cm_tri.sequence) == "".join(wdk.p_sequence)

def test_reference_kernel_exists():
    assert gpm_wdk is not None

def test_reference_BLOSUM62_kernel():
    # TODO compare parsed S_matrix with gp_modeling reference handling
    assert gpm_wdk == wdk