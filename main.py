import numpy as np
import re
import os
from scipy.io import loadmat
from contact_mapper import ContactMapper
from protein_representation import ProteinCollection
from graphkernel import MatrixKernel
from utility import parse_matlab_mutation_file, parse_mutations, convert_aa_sequence
from utility import preprocess_observations
from utility import convert_graph_from_matlab_file
from utility import get_mutation_idx
from data_scaler import BayesScaler
from gp_regression import GPRegression

if __name__ == "__main__":

    # Create and test Contact Mapper
    # # example case 1PGA - CA-distance

    # example case 1PGA - residue distance
    pga_file = loadmat(os.path.join(os.path.dirname(__file__), os.path.join("data", "1PGA.mat")))
    ref_adj = convert_graph_from_matlab_file(pga_file["contact_map"])

    cm_tri = ContactMapper(pdb_file="./pdb/1pga.pdb", tri_dist=True)
    cm_tri.adjacency = ref_adj # just to make sure adjacencies are propagated correctly
    
    mutations_dict_exp = parse_matlab_mutation_file("./data/ddg_protherm.mat", query="ddg_protherm")
    mutations_dict_is = parse_matlab_mutation_file("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")

    pcol = ProteinCollection(cm_tri, pdb_ID="1PGA", mutations_exp=mutations_dict_exp, mutations_sim=mutations_dict_is,
                    TESTING=True)

    mut_S_exp, mut_adj_exp, ΔΔg_exp, mut_ids_exp = parse_mutations(mutation_dict=mutations_dict_exp.get(pcol.pdb_ID), 
                                                    sequence=pcol.sequence, adjacency=ref_adj)
    mut_S_is, mut_adj_is, ΔΔg_is, mut_ids_is = parse_mutations(mutation_dict=mutations_dict_is.get(pcol.pdb_ID), 
                                                    sequence=pcol.sequence, adjacency=ref_adj)
    X_exp, X_is = convert_aa_sequence(mut_S_exp), convert_aa_sequence(mut_S_is)
    y_wt = np.array([0])[:, np.newaxis]
    X_wt = convert_aa_sequence([pcol.sequence])

    # scale using Bayesian Scaling
    bs_rosetta = BayesScaler(is_mutations=mut_ids_is, ΔΔg=pcol.ΔΔg_is, exp_mutations=mut_ids_exp, 
                        experimentally_observed_ΔΔg=pcol.ΔΔg_exp, TESTING=False, pdb_ID="1PGA")

    # print("theta")
    # print(bs_rosetta.θ)
    # print("sigma T")
    # print(bs_rosetta.σ_T)
    # print(bs_rosetta.σ_T.shape)
    # print(bs_rosetta.plot_scaling())
    ΔΔg_exp = ΔΔg_exp[:, np.newaxis]
    ΔΔg_is_scaled = bs_rosetta.transform(ΔΔg_is)[:, np.newaxis]
    
    # mean sigma from scaler run
    σ_T = 1.41618

    # Scale y-values as done in the implementation by normalizing with mean and max
    mean_y, max_y, y_wt, ΔΔg_exp, ΔΔg_is_scaled = preprocess_observations(y_wt, ΔΔg_exp, ΔΔg_is_scaled)

    gpr = GPRegression(protein_representation=pcol, X_wt=X_wt, X_exp=X_exp, X_is=X_is, 
                         y_wt=y_wt, y_exp=ΔΔg_exp, y_is=ΔΔg_is_scaled, adjacencies=ref_adj, 
                         σ_T=σ_T, y_max=max_y, y_mean=mean_y)
    # print(gpr.neg_ll())
    # gpr.parameter_optimization()
    print(gpr.neg_ll())
    opt_results, gpr_results, n_mutations = gpr.position_level_CV()
    #print(opt_results)
    print(gpr_results)
    print(n_mutations)
    mu = [res.get("mu") for res in gpr_results]
    cov = [res.get("cov") for res in gpr_results]
    ys = [res.get("y_exp") for res in gpr_results]
    gpr.plot(f_μ=mu, cov=cov, y_test=ys, mutations=n_mutations)
    # # gpr.plot_log_prob()
