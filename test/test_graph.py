from contact_mapper import ContactMapper

cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/1pga.pdb", tri_dist=True)

def test_cm_tri_exists():
    assert cm_tri is not None

def test_cm_tri_distance_matrix_dim():
    len_N = len(cm_tri.sequence)
    assert len(cm_tri.distance_matrices[0]) == len_N

# TODO test if contact map is binary

# TODO test if distance map is continuous