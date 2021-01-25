import numpy as np
import pickle
import re
import os
from scipy.io import loadmat
from contact_mapper import ContactMapper
from protein_representation import ProteinCollection
from utility import parse_matlab_mutation_file, parse_mutations, convert_aa_sequence
from utility import preprocess_observations
from utility import convert_graph_from_matlab_file
from utility import get_mutation_idx
from data_scaler import BayesScaler
from gp_regression import GPRegression
from visualization import plot_hyperparameters, plot_mean_over_weights, plot_pos_lvl_gpr_individual, plot_sigmas
from visualization import generate_results_table, generate_total_results_table 
from visualization import plot_mut_lvl_gpr_total, plot_mut_lvl_gpr_individual
from visualization import plot_pos_lvl_gpr_total, plot_pos_lvl_gpr_individual
from visualization import plot_covariance_matrices, plot_mWDK

def write_results(gpr_results: dict, gpr, suffix=""):
    with open(f"./results/1PGA_gpr_results_{gpr.cv_flag}_{suffix}.pickle", "wb") as outfile:
        pickle.dump(gpr_results, outfile)

pdbs = ["1PGA", "1CSP", "1BPI", "1RGG", "1RTB", "2RN2", "4LYZ","2LZM", "1BVC"]

if __name__ == "__main__":
    pdb="1PGA"
    print(pdb)
    # Contact Mapper
    # example case 1PGA - residue distance
    pga_file = loadmat(os.path.join(os.path.dirname(__file__), os.path.join("data", f"{pdb}.mat")))
    ref_adj = convert_graph_from_matlab_file(pga_file["contact_map"])

    cm_tri = ContactMapper(pdb_file=f"./pdb/{pdb.lower()}.pdb", tri_dist=True)
    cm_tri.adjacency = ref_adj # just to make sure adjacencies are propagated correctly
    
    mutations_dict_exp = parse_matlab_mutation_file("./data/ddg_protherm.mat", query="ddg_protherm")
    mutations_dict_is = parse_matlab_mutation_file("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")

    pcol = ProteinCollection(cm_tri, pdb_ID=pdb, mutations_exp=mutations_dict_exp, mutations_sim=mutations_dict_is,
                    TESTING=False)

    mut_S_exp, mut_adj_exp, ΔΔg_exp, mut_ids_exp = parse_mutations(mutation_dict=mutations_dict_exp.get(pcol.pdb_ID), 
                                                    sequence=pcol.sequence, adjacency=ref_adj)
    mut_S_is, mut_adj_is, ΔΔg_is, mut_ids_is = parse_mutations(mutation_dict=mutations_dict_is.get(pcol.pdb_ID), 
                                                    sequence=pcol.sequence, adjacency=ref_adj)
    X_exp, X_is = convert_aa_sequence(mut_S_exp), convert_aa_sequence(mut_S_is)
    y_wt = np.array([0])[:, np.newaxis]
    X_wt = convert_aa_sequence([pcol.sequence])

    # # scale using Bayesian Scaling
    # bs_rosetta = BayesScaler(is_mutations=mut_ids_is, ΔΔg=pcol.ΔΔg_is, exp_mutations=mut_ids_exp, 
    #                     experimentally_observed_ΔΔg=pcol.ΔΔg_exp, TESTING=False, pdb_ID=pdb)
    # bs_rosetta.plot_scaling()
    # ΔΔg_exp = ΔΔg_exp[:, np.newaxis]
    # ΔΔg_is_scaled = bs_rosetta.transform(ΔΔg_is)[:, np.newaxis]


    # # Scale y-values as done in the implementation by normalizing with mean and max
    # mean_y, max_y, y_wt, ΔΔg_exp, ΔΔg_is_scaled = preprocess_observations(y_wt, ΔΔg_exp, ΔΔg_is_scaled)

    # gpr = GPRegression(protein_representation=pcol, X_wt=X_wt, X_exp=X_exp, X_is=X_is, 
    #                      y_wt=y_wt, y_exp=ΔΔg_exp, y_is=ΔΔg_is_scaled, adjacencies=ref_adj, 
    #                      σ_T=bs_rosetta.σ_T, y_max=max_y, y_mean=mean_y)

    # # get optimization values
    # init_neg_ll = gpr.neg_ll()
    # gpr.parameter_optimization()
    # end_neg_ll = gpr.neg_ll()
    # # plot MKL after optimization
    # mats = gpr.covariance_matrices
    # mWDK = gpr.mWDK(gpr.X, gpr.covariance_matrices)
    # plot_mWDK(pcol, mWDK.detach().numpy())

    # with open("./results/hyper/hyper_X_all_1PGA.pickle", "wb") as outfile:
    #     pickle.dump({"nll": (init_neg_ll, end_neg_ll), "w": gpr.weights.get_value()}, outfile)
    # gpr.reset_trainable_parameters()
    # experimental pos-lvl CV
    # gpr_results_pos_lvl = gpr.position_level_CV()
    # write_results(gpr_results_pos_lvl, gpr)
    # # reference ALL pos-lvl CV
    # gpr_results_pos_lvl_ref = gpr.position_level_CV_reference()
    # write_results(gpr_results_pos_lvl_ref, gpr)
    # mutation lvl CV (LOO)
    # gpr_results_mutation_lvl = gpr.mutation_level_CV()
    # write_results(gpr_results_mutation_lvl, gpr)


    # TODO run below for plotting
    # plot results
    pdbs = ["1PGA", "1CSP", "1BPI", "1RGG", "2RN2", "4LYZ","2LZM", "1RTB", "1BVC"]
    # plot_pos_lvl_gpr_total(proteins=pdbs, results_dir="./results/mGPfusion/pos_cv/")
    plot_pos_lvl_gpr_individual(proteins=pdbs, results_dir="./results/mGPfusion/pos_cv/")
    # plot position level reference
    plot_pos_lvl_gpr_total(proteins=pdbs, results_dir="./results/mGPfusion/pos_cv/", 
                save_fig="./fig/gpr/", suffix="pos-lvl")
    plot_pos_lvl_gpr_total(proteins=pdbs, results_dir="./results/mGPfusion/pos_cv_ref_error/", 
            save_fig="./fig/gpr/", suffix="2σ-pos-lvl")
    plot_hyperparameters(proteins=pdbs)
    plot_sigmas(proteins=pdbs)
    plot_mean_over_weights(proteins=pdbs, dir="./results/mGPfusion/pos_cv/", suffix="pos-lvl")
    plot_mean_over_weights(proteins=pdbs, dir="./results/mGPfusion/mut_cv/", suffix="mut-lvl")
    # plot_mean_over_weights(proteins=pdbs, dir="./results/mGPfusion/mut_cv_ref_error/", suffix="2σ")
    result_df = generate_results_table(proteins=pdbs, method=["mGPfusion", "2σ mGPfusion"], 
        dir="./results/")
    print(result_df)
    print(result_df.to_latex(index=True, bold_rows=True))
    total_results_df = generate_total_results_table(proteins=pdbs, dir="./results/")
    print(total_results_df)
    print(total_results_df.to_latex(index=True, bold_rows=True))
    plot_mut_lvl_gpr_total(proteins=pdbs, results_dir="./results/mGPfusion/mut_cv/")
    plot_mut_lvl_gpr_individual(proteins=pdbs, results_dir="./results/mGPfusion/mut_cv/")
    # plot_mut_lvl_gpr_individual(proteins=pdbs, results_dir="./results/mGPfusion/mut_cv/", suffix="uncertainties",
    #         uncertainties=True)