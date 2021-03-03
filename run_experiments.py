from scipy.io import loadmat
import os
import argparse
from contact_mapper import ContactMapper
from protein_representation import ProteinCollection
from data_scaler import BayesScaler
from gp_regression import GPRegression
import torch
from tqdm import tqdm
from utility import parse_matlab_mutation_file, parse_mutations, convert_graph_from_matlab_file
from utility import convert_aa_sequence, preprocess_observations
from utility import compute_rmse, compute_ρ
from utility import get_mutation_idx
from visualization import plot_hyperparameters, plot_mean_over_weights, plot_pos_lvl_gpr_individual, plot_sigmas
from visualization import generate_results_table, generate_total_results_table 
from visualization import plot_mut_lvl_gpr_total, plot_mut_lvl_gpr_individual
from visualization import plot_pos_lvl_gpr_total, plot_pos_lvl_gpr_individual
from visualization import plot_covariance_matrices, plot_mWDK
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_squared_error
import tracemalloc

import numpy as np
import pandas as pd
import pickle
import mlflow


def run_plotting(pdbs):
    plot_pos_lvl_gpr_individual(proteins=pdbs, results_dir="./results/mGPfusion/pos_cv/")
    # plot position level reference
    plot_pos_lvl_gpr_total(proteins=pdbs, results_dir="./results/mGPfusion/pos_cv/", 
                save_fig="./fig/gpr/", suffix="pos-lvl")
    plot_hyperparameters(proteins=pdbs)
    plot_sigmas(proteins=pdbs)
    plot_mean_over_weights(proteins=pdbs, dir="./results/mGPfusion/pos_cv/", suffix="pos-lvl")
    plot_mean_over_weights(proteins=pdbs, dir="./results/mGPfusion/mut_cv/", suffix="mut-lvl")
    result_df = generate_results_table(proteins=pdbs, method=["mGPfusion", "2σ mGPfusion"], dir="./results/")
    print(result_df)
    print(result_df.to_latex(index=True, bold_rows=True))
    total_results_df = generate_total_results_table(proteins=pdbs, dir="./results/")
    print(total_results_df)
    print(total_results_df.to_latex(index=True, bold_rows=True))
    plot_mut_lvl_gpr_total(proteins=pdbs, results_dir="./results/mGPfusion/mut_cv/")
    plot_mut_lvl_gpr_individual(proteins=pdbs, results_dir="./results/mGPfusion/mut_cv/")
    # plot_mut_lvl_gpr_individual(proteins=pdbs, results_dir="./results/mGPfusion/mut_cv/", suffix="uncertainties",
    #         uncertainties=True)


def init_experiment_run(pdb: str) -> tuple:
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
                        experimentally_observed_ΔΔg=pcol.ΔΔg_exp, TESTING=False, pdb_ID=pdb, cached=True)
    σ_T = bs_rosetta.σ_T

    ΔΔg_exp = ΔΔg_exp[:, np.newaxis]
    ΔΔg_is_scaled = bs_rosetta.transform(ΔΔg_is)[:, np.newaxis]
    # Scale y-values as done in the implementation by normalizing with mean and max
    mean_y, max_y, y_wt, ΔΔg_exp, ΔΔg_is_scaled = preprocess_observations(y_wt, ΔΔg_exp, ΔΔg_is_scaled)
    return pcol, X_wt, X_exp, X_is, y_wt, ΔΔg_exp, ΔΔg_is_scaled, ref_adj, bs_rosetta.σ_T, max_y, mean_y


def init_mgp_regression(pcol, X_wt, X_exp, X_is, y_wt, ΔΔg_exp, ΔΔg_is_scaled, ref_adj, 
                    σ_T, max_y, mean_y, fusion=True):
    gpr = GPRegression(protein_representation=pcol, X_wt=X_wt, X_exp=X_exp, X_is=X_is, 
                         y_wt=y_wt, y_exp=ΔΔg_exp, y_is=ΔΔg_is_scaled, adjacencies=ref_adj, 
                         σ_T=σ_T, y_max=max_y, y_mean=mean_y, cached=True, fusion=fusion)
    return gpr


def cached_mut_lvl_CV(idx, pdb, reference: bool=False, optim: bool=True) -> dict:
    optimization_parameters, mutations, fit_parameters = {}, {}, {}
    pcol, X_wt, X_exp, X_is, y_wt, ΔΔg_exp, ΔΔg_is_scaled, ref_adj, bs_rosetta, max_y, mean_y = init_experiment_run(pdb)
    # get all experimental mutations incl WT
    if idx == 0: # exclude WT from CV
        return optimization_parameters, mutations, fit_parameters
    gpr = init_mgp_regression(pcol, X_wt, X_exp, X_is, y_wt, ΔΔg_exp, ΔΔg_is_scaled, ref_adj, bs_rosetta, max_y, mean_y)
    # set train and testing indices
    gpr.set_train_index(np.delete(np.arange(0, gpr.X.shape[0]), idx))
    gpr.set_test_index(np.array([idx]))
    n_mutations = gpr.protein.mutation_ids[idx].count(")") # get mutations by closing brackets on tuple
    nll_init = gpr.neg_ll()
    if optim:
        try:
            gpr.parameter_optimization()
        except RuntimeError as _:
            print("Optimization broke.")
            gpr.reset_trainable_parameters()
    nll_end = gpr.neg_ll()
    optimization_parameters = {"nll_init": nll_init, 
                                "nll_end": nll_end,
                                "w": gpr.weights.get_value().detach().numpy(),
                                "sigma_S": gpr.σ_S.get_value().detach().numpy(),
                                "sigma_E": gpr.σ_E.get_value().detach().numpy(),
                                "t": gpr.t.get_value().detach().numpy()}
    f_μ, cov = gpr._fit(ref=reference)
    fit_parameters = {"mu": f_μ.squeeze().detach().numpy(),
                    "cov": cov.squeeze().detach().numpy(),
                    "y_exp": (gpr.y_test.detach().numpy() * gpr.y_max) + gpr.y_mean
                    }
    return optimization_parameters, n_mutations, fit_parameters


def run_mgpfusion_experiment_pos_lvl(pdb: str, idx: int, optim: bool, ref: bool, verbose=False, write=True) -> bool:
    if verbose:
        print(f"{pdb} - pos: {idx},  optim: {optim}, reference: {ref}")
    pcol, X_wt, X_exp, X_is, y_wt, ΔΔg_exp, ΔΔg_is_scaled, ref_adj, bs_rosetta, max_y, mean_y = init_experiment_run(pdb)
    experimental_mutation_index = get_mutation_idx(pcol.mut_ids_exp)
    # gather all mutations at that position and assign train and test indices
    mutation_bool_mask = np.array([bool(idx in mut) for mut in experimental_mutation_index])
    test_mutation_idx = np.where(mutation_bool_mask)[0]
    not_test_mutation_idx = np.where(~mutation_bool_mask)[0]
    if len(test_mutation_idx) == 0:
        print(f"No Mutation at pos:{idx} - skipping...")
        return 
    n_mutations = np.array([len(mut) for mut in experimental_mutation_index if bool(idx in mut)])
    gpr = init_mgp_regression(pcol, X_wt, X_exp, X_is, y_wt, ΔΔg_exp, ΔΔg_is_scaled, ref_adj, bs_rosetta, max_y, mean_y)
    # split into train and test
    gpr.set_test_index(1+test_mutation_idx) # offset with WT 
    # combine WT + not selected + in silico for training data
    train_index = np.concatenate([np.array([0]), 1+not_test_mutation_idx, 
                    np.arange(start=len(gpr.X_exp)+1, stop=gpr.X.shape[0])]) # all simulated data are training data
    gpr.set_train_index(train_index)
    # optimize
    nll_init = gpr.neg_ll()
    if optim:
        try:
            gpr.parameter_optimization()
        except RuntimeError as _:
            print("Optimization broke.")
            gpr.reset_trainable_parameters()
    nll_end = gpr.neg_ll()
    f_μ, cov = gpr._fit(ref=ref)
    # write optimization results
    opt_params = {"w": gpr.weights.get_value().detach().numpy(),
                                "sigma_S": gpr.σ_S.get_value().detach().numpy(),
                                "sigma_E": gpr.σ_E.get_value().detach().numpy(),
                                "t": gpr.t.get_value().detach().numpy(),
                                "nll_init": nll_init.detach().numpy(), 
                                "nll_end": nll_end.detach().numpy()}
    fit_params = {'mu': f_μ.squeeze().detach().numpy(),
                            'cov': cov.squeeze().detach().numpy(),
                            'y_exp': (gpr.y_test.detach().numpy() * gpr.y_max) + gpr.y_mean
                            }
    spearman_r, spearman_p = spearmanr(fit_params.get('mu'), fit_params.get("y_exp"))
    mse = mean_squared_error(fit_params.get('mu'), fit_params.get("y_exp"))
    mlflow.log_metric(key="spearman r", value=spearman_r, step=idx)
    mlflow.log_metric(key="spearman p", value=spearman_p, step=idx)
    mlflow.log_metric(key="mse", value=mse, step=idx)
    filename = f"/home/rimichael/pro_tooling/output/{pdb}_pos_lvl_opt_{optim}_ref_{ref}_{idx}.pkl"
    if write and bool(opt_params):
        data_dict = {**opt_params, **fit_params, "idx": idx, "n_mut": n_mutations}
        with open(filename, "wb") as outfile:
            pickle.dump(data_dict, outfile)
        mlflow.log_artifact(filename)
    return 

    # ### MUTATION LEVEL CV
    # # mutation lvl CV (LOO)
    # gpr_results_mutation_lvl = cached_mut_lvl_CV(reference=False, pdb=pdb)
    # write_results(gpr_results_mutation_lvl, dir="mGPfusion/mut_cv", cv="mut_lvl")
    # # mutation lvl CV w/ mGP reference error
    # gpr_results_mutation_lvl = cached_mut_lvl_CV(reference=True, pdb=pdb)
    # write_results(gpr_results_mutation_lvl, dir="mGPfusion/mut_cv", cv="mut_lvl_REF")
    # # mutation lvl CV no optimization
    # gpr_results_mutation_lvl = cached_mut_lvl_CV(reference=False, optim=False, pdb=pdb)
    # write_results(gpr_results_mutation_lvl, dir="mGPfusion/mut_cv", cv="mut_lvl_no_optim")
    # # mutation lvl CV no optimization w/ reference error
    # gpr_results_mutation_lvl = cached_mut_lvl_CV(reference=True, optim=False, pdb=pdb)
    # write_results(gpr_results_mutation_lvl, dir="mGPfusion/mut_cv", suffix="mut_lvl_no_optim_REF")

def run_mgpfusion_experiment_mut_lvl(pdb: str, idx: int, optim: bool, ref: bool, verbose=False) -> dict:
    """
    Runs Loo CV routine on experiment
    """
    if verbose:
        print(f"{pdb} - pos: {idx},  optim: {optim}, reference: {ref}")
    gpr_results_mutation_lvl = cached_mut_lvl_CV(idx=idx, pdb=pdb, reference=ref, optim=optim)
    return gpr_results_mutation_lvl


def cached_BLAT_pos_lvl_CV(reference: bool=False, optim: bool=True) -> dict:
    blat_file = os.path.join(os.path.dirname(__file__), os.path.join("data/blat/BLAT_ECOLX_Ranganathan2015.csv"))
    blat_df = pd.read_csv(blat_file)
    blat_df["growth"] = blat_df["2500"]
    clipped_mutations = [(mut, growth) for (mut, growth) in zip(blat_df.mutant, blat_df.growth) if int(mut[1:-1])<=263]
    # WARNING: we clip mutations at position 263 - mutations go until 286, however pdb is only 263 (main chain) long
    mutation_dict = {"1FQG" : clipped_mutations}
    pdb_file = "./pdb/1fqg.pdb"
    contact_map = ContactMapper(pdb_file=pdb_file, tri_dist=True)
    # contact_map.plot_distance_matrix()
    # contact_map.plot_contact_map()
    pcol = ProteinCollection(contact_map, pdb_ID="1FQG", mutations_exp=mutation_dict, mutations_sim={})
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

    mutations = []
    optimization_parameters = []
    fit_parameters = []
    for pos in tqdm(range(len(pcol.sequence))):
        print("reinitialize ...")
        gpr = init_mgp_regression(pcol, X_wt, X_exp, np.array([]), y_wt, ΔΔg_exp, ΔΔg_is_scaled, adjacencies, 
                                σ_T=torch.Tensor([0.]), max_y=max_y, mean_y=mean_y, fusion=False)
        experimental_mutation_index = get_mutation_idx(gpr.protein.mut_ids_exp)
        # gather all mutations at that position and assign train and test indices
        mutation_bool_mask = np.array([bool(pos in mut) for mut in experimental_mutation_index])
        test_mutation_idx = np.where(mutation_bool_mask)[0]
        not_test_mutation_idx = np.where(~mutation_bool_mask)[0]
        n_mutations = np.array([len(mut) for mut in experimental_mutation_index if bool(pos in mut)])
        # split into train and test
        gpr.set_test_index(1+test_mutation_idx) # offset with WT 
        # combine WT + not selected + in silico for training data
        train_index = np.concatenate([np.array([0]), 1+not_test_mutation_idx, 
                        np.arange(start=len(gpr.X_exp)+1, stop=gpr.X.shape[0])]) # all simulated data are training data
        gpr.set_train_index(train_index)
        if gpr.x_test.shape[0] == 0:
            print(f"No Mutation at pos:{pos} - skipping...")
            continue
        # optimize
        nll_init = gpr.neg_ll()
        if optim:
            try:
                gpr.parameter_optimization()
            except RuntimeError as _:
                print("Optimization broke.")
                gpr.reset_trainable_parameters()
        nll_end = gpr.neg_ll()
        f_μ, cov = gpr._fit(ref=reference)
        # write optimization results
        optimization_parameters.append({"w": gpr.weights.get_value(),
                                    "sigma_S": gpr.σ_S.get_value(),
                                    "sigma_E": gpr.σ_E.get_value(),
                                    "t": gpr.t.get_value(),
                                    "nll": (nll_init, nll_end)})
        mutations.append(n_mutations)
        fit_parameters.append({'mu': f_μ.squeeze().detach().numpy(),
                                'cov': cov.squeeze().detach().numpy(),
                                'y_exp': (gpr.y_test.detach().numpy() * gpr.y_max) + gpr.y_mean
                                })
    predictions = np.concatenate([np.atleast_1d(x) for x in [elem.get('mu') for elem in fit_parameters]])
    experimental = np.concatenate([x for sub in [elem.get('y_exp') for elem in fit_parameters] for x in sub])
    rho = compute_ρ(y_vec=experimental, y_pred_μ=predictions)
    rmse = compute_rmse(y=experimental, y_pred_μ=predictions)
    results = {"optimization": optimization_parameters, 
                "regression": fit_parameters, 
                "mutations": mutations,
                "rho": rho,
                "rmse": rmse}
    return results


def run_BLAT_experiment():    
    gpr_results_pos_lvl = cached_BLAT_pos_lvl_CV(reference=False, optim=False)
    write_results(gpr_results_pos_lvl, dir="mGPfusion/blat/pos_cv", cv="pos_lvl_no_optim", pdb="1FQG")
    gpr_results_pos_lvl = cached_BLAT_pos_lvl_CV(reference=False, optim=True)
    write_results(gpr_results_pos_lvl, dir="mGPfusion/blat/pos_cv", suffix="pos_lvl", pdb="1FQG")


def run_BLAT_experiment_PALZKILL_1BTL():
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
    cv_options = ["pos_lvl", "mut_lvl"]
    parser = argparse.ArgumentParser(description="Experiment Module - run specific Regression calls.")
    parser.add_argument("-p", "--pdb", type=str, help="str identifier of pdb file")
    parser.add_argument("-i", "--idx", type=int, help="index of CV run")
    parser.add_argument("-r", "--run", type=str, choices=cv_options, help="type of CV routine to run")
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbosity boolean.")
    parser.add_argument("-o", "--optim", action='store_true', help="Run optimization boolean flag.")
    parser.add_argument("-m", "--mode", action='store_true', help="Run reference modus (2 sigma) boolean flag.")
    parser.add_argument("--seed", type=int, default=42, help="Randomness seed for replicability.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # get mutations for pdb
    exp_mutations = parse_matlab_mutation_file(f"{os.path.dirname(__file__)}/data/mgp/ddg_protherm.mat", 
                    query="ddg_protherm")
    sim_mutations = parse_matlab_mutation_file(f"{os.path.dirname(__file__)}/data/mgp/ddg_rosetta_single.mat", 
                    query="ddg_rosetta_single")

    with mlflow.start_run():
        if args.run == "pos_lvl":
            run_mgpfusion_experiment_pos_lvl(pdb=args.pdb, idx=args.idx, 
                                                optim=args.optim, ref=args.mode, verbose=args.verbose)
        elif args.run == "mut_lvl":
            run_mgpfusion_experiment_mut_lvl(pdb=args.pdb, idx=args.idx, optim=args.optim, 
                                                ref=args.mode, verbose=args.verbose)
        else:
            parser.print_help()
            raise RuntimeError("Wrong CV option provided. See help.")
    # run_plotting()
    # run_BLAT_experiment()
    # run_BLAT_experiment_2()
