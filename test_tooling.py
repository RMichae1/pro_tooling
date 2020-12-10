from contact_mapper import ContactMapper
from protein_representation import ProteinCollection
from graphkernel import MatrixKernel
from data_handler import parse_mutations

if __name__ == "__main__":

    # Create and test Contact Mapper
    # # example case 1PGA - CA-distance
    # cm = ContactMapper(pdb_file="/home/rcml/pdb/1pga.pdb")
    # print(cm.contact_maps)
    # print(cm.distance_matrices)
    # cm.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    # cm.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")
    # # example case 1PGA - residue distance
    cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/1pga.pdb", tri_dist=True)
    # cm_tri.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    # cm_tri.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")

    # # example case 1PGA - residue distance - with non AAs
    # cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/1pga.pdb", tri_dist=True, check_AA=False)
    # cm_tri.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    # cm_tri.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")

    # # example case 1LZI - CA-distance
    # cm = ContactMapper(pdb_file="/home/rcml/pdb/1lzi.pdb")
    # cm.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    # cm.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")
    # # example case 1LZI - residue distance
    # cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/1lzi.pdb", tri_dist=True)
    # cm_tri.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    # cm_tri.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")

    # # example case 2LZM - CA-distance
    # cm = ContactMapper(pdb_file="/home/rcml/pdb/2lzm.pdb")
    # print(cm.contact_maps)
    # print(cm.distance_matrices)
    # print(len(cm.distance_matrices))
    # for d in cm.distance_matrices:
    #     print(len(d))
    # cm.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    # cm.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")
    # # example case 2LZM - residue distance
    # cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/2lzm.pdb", tri_dist=True)
    # cm_tri.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    # cm_tri.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")

    # Create and test Graph Kernel

    wdk = MatrixKernel(p_sequence=cm_tri.sequence, p_adjacency=cm_tri.adjacency,
                                        q_sequence=cm_tri.sequence, q_adjacency=cm_tri.adjacency)
    print(wdk.kernel)
    #print(wdk.K_ϕ)

    mutational_dict = parse_mutations("./data/ddg_protherm.mat", query="ddg_protherm")
    pcol = ProteinCollection(cm_tri, pdb_ID="1PGA", pdb_mutations=mutational_dict)
    print(pcol.wdk_df)
    pcol.plot_wdks()