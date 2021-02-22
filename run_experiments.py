from scipy.io import loadmat
import os
from contact_mapper import ContactMapper
from protein_representation import ProteinCollection
from data_scaler import BayesScaler
from gp_regression import GPRegression
import torch

from utility import parse_matlab_mutation_file, parse_mutations, convert_graph_from_matlab_file
from utility import convert_aa_sequence, preprocess_observations

from visualization import plot_hyperparameters, plot_mean_over_weights, plot_pos_lvl_gpr_individual, plot_sigmas
from visualization import generate_results_table, generate_total_results_table 
from visualization import plot_mut_lvl_gpr_total, plot_mut_lvl_gpr_individual
from visualization import plot_pos_lvl_gpr_total, plot_pos_lvl_gpr_individual
from visualization import plot_covariance_matrices, plot_mWDK
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import pickle

todos = []
done = []
pdbs = ["1PGA", "1CSP", "1BPI", "1RGG", "1RTB", "2RN2", "4LYZ","2LZM", "1BVC"]
buggy = ["1BNI", "1VQB", "1LZI", "2CI2","1RN1", "1PIN"]

# get mutations for pdb
exp_mutations = parse_matlab_mutation_file(f"{os.path.dirname(__file__)}/data/mgp/ddg_protherm.mat", 
                query="ddg_protherm")
sim_mutations = parse_matlab_mutation_file(f"{os.path.dirname(__file__)}/data/mgp/ddg_rosetta_single.mat", 
                query="ddg_rosetta_single")


def write_results(gpr_results: dict, gpr, dir, suffix=""):
    ref_dir = os.path.dirname(__file__)
    with open(f"{ref_dir}/results/{dir}/{gpr.cv_flag}_{gpr.protein.pdb_ID}_{suffix}.pickle", "wb") as outfile:
        pickle.dump(gpr_results, outfile)


def run_plotting(pdbs=pdbs):
    plot_pos_lvl_gpr_individual(proteins=pdbs, results_dir="./results/mGPfusion/pos_cv/")
    # plot position level reference
    plot_pos_lvl_gpr_total(proteins=pdbs, results_dir="./results/mGPfusion/pos_cv/", 
                save_fig="./fig/gpr/", suffix="pos-lvl")
    plot_hyperparameters(proteins=pdbs)
    plot_sigmas(proteins=pdbs)
    plot_mean_over_weights(proteins=pdbs, dir="./results/mGPfusion/pos_cv/", suffix="pos-lvl")
    plot_mean_over_weights(proteins=pdbs, dir="./results/mGPfusion/mut_cv/", suffix="mut-lvl")
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

def init_mgp_regression(pdb):
    pga_file = loadmat(os.path.join(os.path.dirname(__file__), os.path.join("data/mgp/", f"{pdb}.mat")))
    ref_adj = convert_graph_from_matlab_file(pga_file["contact_map"])

    cm_tri = ContactMapper(pdb_file=f"./pdb/{pdb.lower()}.pdb", tri_dist=True)
    cm_tri.adjacency = ref_adj # just to make sure adjacencies are propagated correctly
    
    mutations_dict_exp = parse_matlab_mutation_file("./data/mgp/ddg_protherm.mat", query="ddg_protherm")
    mutations_dict_is = parse_matlab_mutation_file("./data/mgp/ddg_rosetta_single.mat", query="ddg_rosetta_single")

    pcol = ProteinCollection(cm_tri, pdb_ID=pdb, mutations_exp=mutations_dict_exp, mutations_sim=mutations_dict_is,
                    TESTING=False)

    mut_S_exp, mut_adj_exp, ΔΔg_exp, mut_ids_exp = parse_mutations(mutation_dict=mutations_dict_exp.get(pcol.pdb_ID), 
                                                    sequence=pcol.sequence, adjacency=ref_adj)
    mut_S_is, mut_adj_is, ΔΔg_is, mut_ids_is = parse_mutations(mutation_dict=mutations_dict_is.get(pcol.pdb_ID), 
                                                    sequence=pcol.sequence, adjacency=ref_adj)
    X_exp, X_is = convert_aa_sequence(mut_S_exp), convert_aa_sequence(mut_S_is)
    y_wt = np.array([0])[:, np.newaxis]
    X_wt = convert_aa_sequence([pcol.sequence])

    # scale using Bayesian Scaling
    bs_rosetta = BayesScaler(is_mutations=mut_ids_is, ΔΔg=pcol.ΔΔg_is, exp_mutations=mut_ids_exp, 
                        experimentally_observed_ΔΔg=pcol.ΔΔg_exp, TESTING=False, pdb_ID=pdb)
    bs_rosetta.plot_scaling()
    ΔΔg_exp = ΔΔg_exp[:, np.newaxis]
    ΔΔg_is_scaled = bs_rosetta.transform(ΔΔg_is)[:, np.newaxis]

    # Scale y-values as done in the implementation by normalizing with mean and max
    mean_y, max_y, y_wt, ΔΔg_exp, ΔΔg_is_scaled = preprocess_observations(y_wt, ΔΔg_exp, ΔΔg_is_scaled)

    gpr = GPRegression(protein_representation=pcol, X_wt=X_wt, X_exp=X_exp, X_is=X_is, 
                         y_wt=y_wt, y_exp=ΔΔg_exp, y_is=ΔΔg_is_scaled, adjacencies=ref_adj, 
                         σ_T=bs_rosetta.σ_T, y_max=max_y, y_mean=mean_y)
    return gpr


def run_mgpfusion_experiment(pdb):
    print(pdb)
    gpr = init_mgp_regression(pdb)
    ### POSITION LEVEL CV
    # pos-lvl CV
    gpr_results_pos_lvl = gpr.position_level_CV(ref=False)
    write_results(gpr_results_pos_lvl, gpr, dir="mGPfusion/pos_cv", suffix="")
    # pos-lvl CV w/ mGP reference error
    gpr_results_pos_lvl = gpr.position_level_CV(ref=True)
    write_results(gpr_results_pos_lvl, gpr, dir="mGPfusion/pos_cv", sub_dirsuffix="_2sigma")
    # pos-lvl CV no optimization
    gpr_results_pos_lvl = gpr.position_level_CV(ref=False, optim=False)
    write_results(gpr_results_pos_lvl, gpr, dir="mGPfusion/pos_cv", suffix="_no_optim")
    # pos-lvl CV no optimization w/ reference error
    gpr_results_pos_lvl = gpr.position_level_CV(ref=True, optim=False)
    write_results(gpr_results_pos_lvl, gpr, dir="mGPfusion/pos_cv", suffix="_no_optim_2sigma")
    ### MUTATION LEVEL CV
    # mutation lvl CV (LOO)
    gpr_results_mutation_lvl = gpr.mutation_level_CV(ref=False)
    write_results(gpr_results_mutation_lvl, gpr, dir="mGPfusion/mut_cv", suffix="")
    # mutation lvl CV w/ mGP reference error
    gpr_results_mutation_lvl = gpr.mutation_level_CV(ref=True)
    write_results(gpr_results_mutation_lvl, gpr, dir="mGPfusion/mut_cv", suffix="_2sigma")
    # mutation lvl CV no optimization
    gpr_results_mutation_lvl = gpr.mutation_level_CV(ref=False, optim=False)
    write_results(gpr_results_mutation_lvl, gpr, dir="mGPfusion/mut_cv", suffix="_no_optim")
    # mutation lvl CV no optimization w/ reference error
    gpr_results_mutation_lvl = gpr.mutation_level_CV(ref=True, optim=False)
    write_results(gpr_results_mutation_lvl, gpr, dir="mGPfusion/mut_cv", suffix="_no_optim_2sigma")


def run_BLAT_experiment():
    blat_file = os.path.join(os.path.dirname(__file__), os.path.join("data/blat/BLAT_ECOLX_Palzkill2012.csv"))
    blat_df = pd.read_csv(blat_file, sep=";", index_col=0)
    blat_df.ddG_stat = blat_df.ddG_stat.str.replace(",", ".").astype(float)
    clipped_mutations = [(mut, ddg) for (mut, ddg) in zip(blat_df.mutant, blat_df.ddG_stat) if int(mut[1:-1])<=263]
    # WARNING: we clip mutations at position 263 - mutations go until 286, however pdb is only 263 (main chain) long
    mutation_dict = {"1BTL" : clipped_mutations}
    
    pdb_file = "./pdb/1btl.pdb"
    contact_map = ContactMapper(pdb_file=pdb_file, tri_dist=True)
    # contact_map.plot_distance_matrix()
    # contact_map.plot_contact_map()

    pcol = ProteinCollection(contact_map, pdb_ID="1BTL", mutations_exp=mutation_dict, mutations_sim={})
    adjacencies = contact_map.adjacency
    mut_S_exp, mut_adj_exp, ΔΔg_exp, mut_ids_exp = parse_mutations(mutation_dict=mutation_dict.get(pcol.pdb_ID), 
                                                    sequence=pcol.sequence, adjacency=adjacencies)
    X_exp = convert_aa_sequence(mut_S_exp)
    y_wt = np.array([0.])[:, np.newaxis]
    X_wt = convert_aa_sequence([pcol.sequence])

    ΔΔg_exp = np.array(ΔΔg_exp)[:, np.newaxis]
    ΔΔg_is = np.array([])[:, np.newaxis]

     # Scale y-values as done in the implementation by normalizing with mean and max
    mean_y, max_y, y_wt, ΔΔg_exp, ΔΔg_is_scaled = preprocess_observations(y_wt, ΔΔg_exp,  ΔΔg_is)

    gpr = GPRegression(protein_representation=pcol, X_wt=X_wt, X_exp=X_exp, X_is=np.array([]), 
                        y_wt=y_wt, y_exp=ΔΔg_exp, y_is=ΔΔg_is_scaled, adjacencies=adjacencies, 
                        σ_T=torch.Tensor([0.]), y_max=max_y, y_mean=mean_y, fusion=False)
    
    gpr_results_mutation_lvl = gpr.mutation_level_CV(ref=False, optim=False)
    write_results(gpr_results_mutation_lvl, gpr, dir="mGPfusion/blat/mut_cv", suffix="_no_optim")
    gpr_results_mutation_lvl = gpr.mutation_level_CV(ref=False, optim=True)
    write_results(gpr_results_mutation_lvl, gpr, dir="mGPfusion/blat/mut_cv", suffix="_")
    gpr_results_pos_lvl = gpr.position_level_CV(ref=False, optim=False)
    write_results(gpr_results_pos_lvl, gpr, dir="mGPfusion/blat/pos_cv", suffix="_no_optim")
    gpr_results_pos_lvl = gpr.position_level_CV(ref=False, optim=True)
    write_results(gpr_results_pos_lvl, gpr, dir="mGPfusion/blat/pos_cv", suffix="_")


if __name__ == "__main__":
    # for pdb in pdbs:
    #     run_mgpfusion_experiment(pdb=pdb)
    # run_plotting()
    run_BLAT_experiment()
    # run_plotting(pdbs=["1PGA", "1CSP", "2RN2"])
