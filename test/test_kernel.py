import os
import pytest
import numpy as np
from utility import convert_graph_from_matlab_file, get_sequence_and_contact_graph
from utility import parse_mutations, parse_matlab_mutation_file, convert_aa_sequence
from graphkernel import MatrixKernel
from scipy.io import loadmat
from protein_representation import ProteinCollection
from contact_mapper import ContactMapper

cm = ContactMapper(pdb_file="./pdb/1pga.pdb", tri_dist=True)
mut_exp = parse_matlab_mutation_file("./data/ddg_protherm.mat", query="ddg_protherm")
mut_is = parse_matlab_mutation_file("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")


def test_normalized_kernel():
        ref_file = loadmat(os.path.join(os.path.dirname(__file__), os.path.join("data", "1PGAkernel_matrices.mat")))
        ref_K_list = ref_file["kernel_matrices"]
        matrices = ref_file["subMats"]
        ref_contact_graph = convert_graph_from_matlab_file(ref_file["al"])
        num_wet_lab_obs = ref_K_list[0][0].shape[0] - 1

        sequence_WT = get_sequence_and_contact_graph(pdb_id="1PGA", cutoff_distance=5., chain_id=None)[0]
        sequence_WT = list(sequence_WT)

        prot = ProteinCollection(cm, pdb_ID="1PGA", mutations_exp=mut_exp, mutations_sim=mut_is)
        # TEST is reference sequence equal to own parsed sequence
        assert np.all([x == y for x,y in zip(sequence_WT, prot.sequence)])

        # TEST adjacencies
        contacts = np.array([contacts for res, contacts in cm.adjacency])
        assert len(ref_contact_graph) == len(contacts)
        #assert np.all([elem_ref == elem for elem_ref, elem in zip(ref_contact_graph, contacts)])
        
        mut_S_exp, _, _, _ = parse_mutations(mutation_dict=mut_exp.get(prot.pdb_ID),
                                                    sequence=sequence_WT, adjacency=ref_contact_graph)
        mut_S = np.vstack([sequence_WT, mut_S_exp])


        for i, m in enumerate(matrices):
            kernel = MatrixKernel(matrix=m[0], matrix_id=None)
            k = kernel.k(convert_aa_sequence(mut_S), adjacencies=ref_contact_graph)
            # k = NormalizedKernel(WeightedDecomposition(substitution_matrix=m[0], contact_map=ref_contact_graph), w=1.0, gamma=1.0)
            np.testing.assert_almost_equal(k.detach().numpy(), ref_K_list[i][0])