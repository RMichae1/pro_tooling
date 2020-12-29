from contact_mapper import ContactMapper
from protein_representation import ProteinCollection, ProteinCollectionSimulated
from protein_representation import AdditiveNoiseRepresentation
from graphkernel import MatrixKernel
from data_handler import parse_mutations
from data_scaler import BayesScaler
from gp_regression import GPRegression

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
                                        q_sequence=cm_tri.sequence, q_adjacency=cm_tri.adjacency, sub_matrix="BLOSUM62")
    print("Computed Kernel Value - WT w/ BLOSUM 62 {}".format(wdk.kernel_value))
    #print(wdk.K_ϕ)

    mutational_dict = parse_mutations("./data/ddg_protherm.mat", query="ddg_protherm")
    pcol = ProteinCollection(cm_tri, pdb_ID="1PGA", pdb_mutations=mutational_dict)
    print(pcol.matrices_df)
    # pcol.plot_sub_matrices()
    print(pcol.mwdk_df)
    # pcol.plot_mwdk()

    # ros_mut_dict = parse_mutations("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")

    # # instantiate and parse IS data
    # rosetta_collection = ProteinCollection(cm_tri, pdb_ID="1PGA", pdb_mutations=ros_mut_dict)
    # # scale using Bayesian Scaling
    # bs_rosetta = BayesScaler(ΔΔg=rosetta_collection.ΔΔg)
    # print(bs_rosetta.θ)
    # print(bs_rosetta.ΔΔg)
    # #print(bs_rosetta.mcmc_samples)
    # print(bs_rosetta.print_summary())
    # print(bs_rosetta.σ_T)
    # print(len(bs_rosetta.σ_T))
    # #print(bs_rosetta.plot_scaling())
    # #rosetta_collection = ProteinCollectionSimulated(cm_tri, pdb_ID="1PGA", pdb_mutations=ros_mut_dict)
    # #print(rosetta_collection.ΔΔg)

    noisy_protein = AdditiveNoiseRepresentation(protein_representation=pcol)

    gpr = GPRegression(protein_representation=pcol, noise_factor=noisy_protein)
    print(gpr.p_sample)