import numpy as np
from contact_mapper import ContactMapper
from protein_representation import ProteinCollection
from graphkernel import MatrixKernel
from utility import parse_matlab_mutation_file, parse_mutations, convert_aa_sequence
from utility import preprocess_observations
from data_scaler import BayesScaler
from gp_regression import GPRegression

if __name__ == "__main__":

    # Create and test Contact Mapper
    # # example case 1PGA - CA-distance

    # example case 1PGA - residue distance
    cm_tri = ContactMapper(pdb_file="./pdb/1pga.pdb", tri_dist=True)
    
    mutations_dict_exp = parse_matlab_mutation_file("./data/ddg_protherm.mat", query="ddg_protherm")
    mutations_dict_is = parse_matlab_mutation_file("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")

    pcol = ProteinCollection(cm_tri, pdb_ID="1PGA", mutations_exp=mutations_dict_exp, mutations_sim=mutations_dict_is,
                    TESTING=True)
    # print(pcol.plot_sub_matrices())

    mut_S_exp, mut_adj_exp, ΔΔg_exp, mut_ids_exp = parse_mutations(mutation_dict=mutations_dict_exp.get(pcol.pdb_ID), 
                                                    sequence=pcol.sequence, adjacency=pcol.adjacency)
    mut_S_is, mut_adj_is, ΔΔg_is, mut_ids_is = parse_mutations(mutation_dict=mutations_dict_is.get(pcol.pdb_ID), 
                                                    sequence=pcol.sequence, adjacency=pcol.adjacency)
    X_exp, X_is = convert_aa_sequence(mut_S_exp), convert_aa_sequence(mut_S_is)
    y_wt = np.array([0])
    X_wt = convert_aa_sequence([pcol.sequence])

    print(X_exp)
    print(X_is)


    # # scale using Bayesian Scaling
    # bs_rosetta = BayesScaler(is_mutations=mut_ids_is, ΔΔg=pcol.ΔΔg_is, exp_mutations=mut_ids_exp, 
    #                     experimentally_observed_ΔΔg=pcol.ΔΔg_exp, TESTING=True, pdb_ID="1PGA")

    # print("theta")
    # print(bs_rosetta.θ)
    # print("sigma T")
    # print(bs_rosetta.σ_T)
    # print(bs_rosetta.σ_T.shape)
    # print(bs_rosetta.print_summary())
    # print(bs_rosetta.plot_scaling())
    σ_T = 1.3756

    # # TODO preprocess y

    # gpr = GPRegression(protein_representation=pcol, X_wt=X_wt, X_exp=X_exp, X_is=X_is, 
    #                      y_wt=y_wt, y_exp=ΔΔg_exp, y_is=ΔΔg_is, σ_T=σ_T)
    # print(gpr.X)
    # print(gpr.neg_ll())
    # gpr.parameter_optimization()
    # print(gpr.neg_ll())
    # # gpr.plot_log_prob()
    # # gpr.plot()