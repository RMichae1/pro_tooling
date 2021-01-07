from contact_mapper import ContactMapper
from protein_representation import ProteinCollection, ProteinCollectionSimulated
from protein_representation import AdditiveNoiseRepresentation
from graphkernel import MatrixKernel
from data_utility import parse_mutations
from data_scaler import BayesScaler
from gp_regression import GPRegression

if __name__ == "__main__":

    # Create and test Contact Mapper
    # # example case 1PGA - CA-distance

    # example case 1PGA - residue distance
    cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/1pga.pdb", tri_dist=True)
    
    mutational_dict_exp = parse_mutations("./data/ddg_protherm.mat", query="ddg_protherm")
    mutational_dict_is = parse_mutations("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")

    pcol = ProteinCollection(cm_tri, pdb_ID="1PGA", 
                    mutations_exp=mutational_dict_exp, mutations_sim=mutational_dict_is,
                    TESTING=True)
    print(pcol.matrices_df)
    pcol.plot_sub_matrices()
    print(pcol.mwdk_df)
    pcol.plot_mwdk()
    print(pcol.mWDK.K_ϕ) # TODO normalize this appropriately

    # instantiate and parse IS data

    # scale using Bayesian Scaling
    bs_rosetta = BayesScaler(ΔΔg=pcol.ΔΔg_is, experimentally_observed_ΔΔg=pcol.ΔΔg_exp)
    print(bs_rosetta.θ)
    # print(bs_rosetta.ΔΔg)
    # print(bs_rosetta.mcmc_samples)
    print(bs_rosetta.print_summary())
    # print(bs_rosetta.σ_T)
    print(bs_rosetta.plot_scaling())
    # #rosetta_collection = ProteinCollectionSimulated(cm_tri, pdb_ID="1PGA", pdb_mutations=ros_mut_dict)
    # #print(rosetta_collection.ΔΔg)

    noisy_protein = AdditiveNoiseRepresentation(protein_representation=pcol)

    # gpr = GPRegression(protein_representation=pcol, noise_factor=noisy_protein)
    # print(gpr.mutation_level_dict.get('μ_list')[0:3])
    # print(gpr.mutation_level_dict['cov_list'][0:3])
    # print(gpr.mutation_level_dict['lml_list'][0])
    # gpr.plot_log_prob()
    # gpr.plot()