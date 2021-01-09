import numpy as np

from contact_mapper import ContactMapper
from graphkernel import MatrixKernel
from data_utility import parse_mutations
from protein_representation import ProteinCollection

mutational_dict_exp = parse_mutations("./data/ddg_protherm.mat", query="ddg_protherm")
mutational_dict_is = parse_mutations("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")
cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/1pga.pdb", tri_dist=True)
rosetta_collection = ProteinCollection(cm_tri, pdb_ID="1PGA", mutations_sim=mutational_dict_is)
pcol = ProteinCollection(cm_tri, pdb_ID="1PGA", mutations_exp=mutational_dict_exp)

def test_mutation_parsing_exists():
    assert mutational_dict_exp is not None and mutational_dict_is is not None

def test_mutation_parsing_dict():
    assert isinstance(mutational_dict_exp, dict)

# TODO test contents of mutation

def test_rosetta_mutations_exist():
    assert rosetta_collection is not None

def test_rosetta_mutations_ddg():
    ddg = [0] + mutational_dict_is # TODO query dict content ddg
    assert ddg == rosetta_collection.ΔΔg



