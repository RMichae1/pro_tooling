from contact_mapper import ContactMapper

cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/1pga.pdb", tri_dist=True)

def test_cm_tri_exists():
    assert cm_tri is not None

def test_cm_tri_distance_matrix_dim():
    len_N = len(cm_tri.sequence)
    assert len(cm_tri.distance_matrices[0]) == len_N

def test_cm_binary():
    for r in cm_tri.contact_maps:
        for val in r:
            if not val != 0 or val != 1:
                return False
    return True

# TODO test if distance map is continuous
def test_dist_mat_continuous():
    print(cm.distance_matrices)
    pass 

# TODO test against reference