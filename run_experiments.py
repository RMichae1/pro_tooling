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
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import tracemalloc

import numpy as np
import pandas as pd
import pickle
import mlflow
from mlflow.tracking import MlflowClient


def init_experiment_run(pdb: str, mutations_dict_exp: dict=None, mutations_dict_is: dict=None) -> tuple:
    cm_tri = ContactMapper(pdb_file=f"./pdb/{pdb.lower()}.pdb", tri_dist=True)
    ref_adj = cm_tri.adjacency
    ref_mat_file = os.path.join(os.path.dirname(__file__), os.path.join("data/mgp/", f"{pdb.upper()}.mat"))
    if os.path.isfile(ref_mat_file):
        pga_file = loadmat(ref_mat_file)
        ref_adj = convert_graph_from_matlab_file(pga_file["contact_map"]) # in case precalculated contacts exist
        cm_tri.adjacency = ref_adj # propagate contactmap to all dependencies
    if not mutations_dict_exp:
        mutations_dict_exp = parse_matlab_mutation_file("./data/mgp/ddg_protherm.mat", query="ddg_protherm")
    if not mutations_dict_is:
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

    ΔΔg_exp = ΔΔg_exp[:, np.newaxis]
    ΔΔg_is_scaled = bs_rosetta.transform(ΔΔg_is)[:, np.newaxis]
    # Scale y-values as done in the implementation by normalizing with mean and max
    mean_y, max_y, y_wt, ΔΔg_exp, ΔΔg_is_scaled = preprocess_observations(y_wt, ΔΔg_exp, ΔΔg_is_scaled)
    # TODO replace bs_rosetta.σ_T with σ_T_samples
    return pcol, X_wt, X_exp, X_is, y_wt, ΔΔg_exp, ΔΔg_is_scaled, ref_adj, bs_rosetta.σ_T, max_y, mean_y


def init_mgp_regression(pcol, X_wt, X_exp, X_is, y_wt, ΔΔg_exp, ΔΔg_is_scaled, ref_adj, 
                    σ_T, max_y, mean_y, fusion=True):
    gpr = GPRegression(protein_representation=pcol, X_wt=X_wt, X_exp=X_exp, X_is=X_is, 
                         y_wt=y_wt, y_exp=ΔΔg_exp, y_is=ΔΔg_is_scaled, adjacencies=ref_adj, 
                         σ_T=σ_T, y_max=max_y, y_mean=mean_y, cached=True, fusion=fusion)
    return gpr


def run_mgpfusion_experiment_pos_lvl(pdb: str, idx: int, optim: bool, ref: bool, run_id: str, verbose=False,
                                     write=True, exp_mutation_dict: dict=None, is_mutation_dict: dict=None) -> None:
    if verbose:
        print(f"{pdb} - pos: {idx},  optim: {optim}, reference: {ref}")
    pcol, X_wt, X_exp, X_is, y_wt, ΔΔg_exp, ΔΔg_is_scaled, ref_adj, bs_rosetta, max_y, mean_y = init_experiment_run(pdb, 
                                                            mutations_dict_exp=exp_mutation_dict, mutations_dict_is=is_mutation_dict)
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
    client = MlflowClient()
    run = client.get_run(run_id)
    print(run.info.run_id)
    spearman_r, spearman_p = spearmanr(fit_params.get('mu'), fit_params.get("y_exp"))
    mse = mean_squared_error(np.atleast_1d(fit_params.get('mu')), np.atleast_1d(fit_params.get("y_exp")))
    client.log_metric(run_id=run.info.run_id, key="spearman r", value=spearman_r, step=idx)
    client.log_metric(run_id=run.info.run_id, key="spearman p", value=spearman_p, step=idx)
    client.log_metric(run_id=run.info.run_id, key="mse", value=mse, step=idx)
    filename = f"/home/rimichael/pro_tooling/output/{pdb}_pos_lvl_opt_{optim}_ref_{ref}_{idx}.pkl"
    if write and bool(opt_params):
        data_dict = {**opt_params, **fit_params, "idx": idx, "n_mut": n_mutations}
        with open(filename, "wb") as outfile:
            pickle.dump(data_dict, outfile)
        client.log_artifact(run.info.run_id, filename)
    #mlflow.end_run()
    return


def run_mgpfusion_experiment_mut_lvl(pdb: str, idx: int, optim: bool, ref: bool, run_id: str,
                                    verbose=False, write=True, exp_mutation_dict: dict=None,
                                    is_mutation_dict: dict=None) -> None:
    """
    Runs Loo CV routine on experiment
    """
    if verbose:
        print(f"{pdb} - pos: {idx},  optim: {optim}, reference: {ref}")
    pcol, X_wt, X_exp, X_is, y_wt, ΔΔg_exp, ΔΔg_is_scaled, ref_adj, bs_rosetta, max_y, mean_y = init_experiment_run(pdb, 
                                                                    mutations_dict_exp=exp_mutation_dict, mutations_dict_is=is_mutation_dict)
    # get all experimental mutations incl WT
    if idx == 0:  # exclude WT from CV
        print("WT excluded from LOO")
        return
    gpr = init_mgp_regression(pcol, X_wt, X_exp, X_is, y_wt, ΔΔg_exp, ΔΔg_is_scaled, ref_adj, bs_rosetta, max_y, mean_y)
    # set train and testing indices
    gpr.set_train_index(np.delete(np.arange(0, gpr.X.shape[0]), idx))
    gpr.set_test_index(np.array([idx]))
    n_mutations = gpr.protein.mutation_ids[idx].count(")")  # get mutations by closing brackets on tuple
    nll_init = gpr.neg_ll()
    if optim:
        try:
            gpr.parameter_optimization()
        except RuntimeError as _:
            print("Optimization broke.")
            gpr.reset_trainable_parameters()
    nll_end = gpr.neg_ll()
    f_μ, cov = gpr._fit(ref=ref)
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
    client = MlflowClient()
    run = client.get_run(run_id)
    print(run.info.run_id)
    spearman_r, spearman_p = spearmanr(fit_params.get('mu'), fit_params.get("y_exp"))
    mse = mean_squared_error(np.atleast_1d(fit_params.get('mu')), np.atleast_1d(fit_params.get("y_exp")))
    client.log_metric(run_id=run.info.run_id, key="spearman r", value=spearman_r, step=idx)
    client.log_metric(run_id=run.info.run_id, key="spearman p", value=spearman_p, step=idx)
    client.log_metric(run_id=run.info.run_id, key="mse", value=mse, step=idx)
    filename = f"./output/{pdb}_mut_lvl_opt_{optim}_ref_{ref}_{idx}.pkl"
    if write and bool(opt_params):
        data_dict = {**opt_params, **fit_params, "idx": idx, "n_mut": n_mutations}
        with open(filename, "wb") as outfile:
            pickle.dump(data_dict, outfile)
        client.log_artifact(run.info.run_id, filename)
    # mlflow.end_run()
    return


def prepare_blat(in_file: str="./data/blat/BLAT_ECOLX_Ranganathan2015.csv", in_silico: str=None):
    blat_df = pd.read_csv(in_file)
    blat_df["growth"] = blat_df["2500"]
    is_mutation_dict = None
    if in_silico:
        pass # TODO implement is_mutation from BLAT simulations
    clipped_mutations = [(mut, growth) for (mut, growth) in zip(blat_df.mutant, blat_df.growth) if int(mut[1:-1])<=263]
    # WARNING: we clip mutations at position 263 - mutations go until 286, however pdb is only 263 (main chain) long
    mutation_dict = {"1FQG": clipped_mutations}
    return blat_df, mutation_dict, is_mutation_dict


def prepare_tll(in_file: str="./data/tll/lipase_variants_tll_tm_tapo_20nov2020.xlsx", in_silico: str=None):
    tll_df = pd.read_excel(in_file)
    # filter out more than 10 mutations
    tll_df = tll_df[tll_df.mut2wt_1ein_join.str.count(" ")<=9].dropna()
    is_mutations = None
    if in_silico:
        if not os.path.exists(in_silico):
            raise FileNotFoundError("Provided in Silico data does not exist!")
        is_df = pd.read_excel(in_silico)
        is_df = is_df.merge(tll_df, how="left", left_on="var_name", right_on="TSA.sample")[["mut2wt_1ein_join", "ddG"]].dropna()
        is_df["mutations"] = is_df.mut2wt_1ein_join.str.replace(" ", "")
        is_df["ddG"] = is_df.ddG.astype(float)
        is_mutations = [(mut, y) for (mut, y) in zip(is_df.mutations, is_df.ddG)]
    tll_df = tll_df[["mut2wt_1ein_join", "TSA.Tm"]]
    # mean over experimental measurements
    tll_df = tll_df.groupby("mut2wt_1ein_join").mean().reset_index()
    tll_df["mutations"] = tll_df.mut2wt_1ein_join.str.replace(" ", "")
    tll_df["TSA"] = tll_df["TSA.Tm"].astype(float)
    exp_mutations = [(mut, y) for (mut, y) in zip(tll_df.mutations, tll_df.TSA)]

    exp_mutation_dict = {"1TIB": exp_mutations}
    is_mutation_dict = {"1TIB": is_mutations}
    return tll_df, exp_mutation_dict, is_mutation_dict


def run_pos_lvl_CV_no_fusion(pdb:str, idx: int, mutation_dict: dict,  run_id: int,
                             ref: bool=False, optim: bool=True, write: bool=True, use_reference_map: bool=False) -> dict:
    pdb_file = f"./pdb/{pdb.lower()}.pdb"
    contact_map = ContactMapper(pdb_file=pdb_file, tri_dist=True)
    pcol = ProteinCollection(contact_map, pdb_ID=pdb, mutations_exp=mutation_dict, mutations_sim={})
    adjacencies = contact_map.adjacency

    # use reference matlab file if required
    ref_mat_file = os.path.join(os.path.dirname(__file__), os.path.join("data/mgp/", f"{pdb.upper()}.mat"))
    if os.path.isfile(ref_mat_file) and use_reference_map:
        pga_file = loadmat(ref_mat_file)
        adjacencies = convert_graph_from_matlab_file(pga_file["contact_map"])  # in case precalculated contacts exist
        contact_map.adjacency = adjacencies  # propagate contactmap to all dependencies

    mut_S_exp, mut_adj_exp, ΔΔg_exp, mut_ids_exp = parse_mutations(mutation_dict=mutation_dict.get(pcol.pdb_ID),
                                                                   sequence=pcol.sequence, adjacency=adjacencies)
    X_exp = convert_aa_sequence(mut_S_exp)
    y_wt = np.array([0.])[:, np.newaxis]
    X_wt = convert_aa_sequence([pcol.sequence])
    ΔΔg_exp = np.array(ΔΔg_exp)[:, np.newaxis]
    ΔΔg_is = np.array([])[:, np.newaxis]
     # Scale y-values as done in the implementation by normalizing with mean and max
    mean_y, max_y, y_wt, ΔΔg_exp, ΔΔg_is_scaled = preprocess_observations(y_wt, ΔΔg_exp,  ΔΔg_is)

    experimental_mutation_index = get_mutation_idx(pcol.mut_ids_exp)
    # gather all mutations at that position and assign train and test indices
    mutation_bool_mask = np.array([bool(idx in mut) for mut in experimental_mutation_index])
    test_mutation_idx = np.where(mutation_bool_mask)[0]
    if len(test_mutation_idx) == 0:
        print(f"No Mutation at pos:{idx} - skipping...")
        return 
    not_test_mutation_idx = np.where(~mutation_bool_mask)[0]
    gpr = init_mgp_regression(pcol, X_wt, X_exp, np.array([]), y_wt, ΔΔg_exp, ΔΔg_is_scaled, adjacencies, 
                            σ_T=torch.Tensor([0.]), max_y=max_y, mean_y=mean_y, fusion=False)
    n_mutations = np.array([len(mut) for mut in experimental_mutation_index if bool(idx in mut)])
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
    optimization_params = {"w": gpr.weights.get_value(),
                                "sigma_S": gpr.σ_S.get_value(),
                                "sigma_E": gpr.σ_E.get_value(),
                                "t": gpr.t.get_value(),
                                "nll": (nll_init, nll_end)}
    mutations = n_mutations
    fit_params = {'mu': f_μ.squeeze().detach().numpy(),
                            'cov': cov.squeeze().detach().numpy(),
                            'y_exp': (gpr.y_test.detach().numpy() * gpr.y_max) + gpr.y_mean
                            }
    client = MlflowClient()
    run = client.get_run(run_id)
    print(run.info.run_id)
    spearman_r, spearman_p = spearmanr(fit_params.get('mu'), fit_params.get("y_exp"))
    mse = mean_squared_error(np.atleast_1d(fit_params.get('mu')), np.atleast_1d(fit_params.get("y_exp")))
    client.log_metric(run_id=run.info.run_id, key="spearman r", value=spearman_r, step=idx)
    client.log_metric(run_id=run.info.run_id, key="spearman p", value=spearman_p, step=idx)
    client.log_metric(run_id=run.info.run_id, key="mse", value=mse, step=idx)
    results = {"optimization": optimization_params,
                "regression": fit_params,
                "mutations": mutations,
                "spearman corr": (spearman_r, spearman_p),
                "mse": mse}
    filename = f"/home/rimichael/pro_tooling/output/{pdb}_NO_FUSION_pos_lvl_opt_{optim}_ref_{ref}_{idx}.pkl"
    if write:
        with open(filename, "wb") as outfile:
            pickle.dump(results, outfile)
        client.log_artifact(run.info.run_id, filename)
    return


def run_BLAT_experiment_PALZKILL_1BTL():
    raise NotImplementedError("1BTL not yet implemented!")
    # TODO
    # blat_file = os.path.join(os.path.dirname(__file__), os.path.join("data/blat/BLAT_ECOLX_Palzkill2012.csv"))
    # blat_df = pd.read_csv(blat_file, sep=";", index_col=0)
    # blat_df.ddG_stat = blat_df.ddG_stat.str.replace(",", ".").astype(float)
    # clipped_mutations = [(mut, ddg) for (mut, ddg) in zip(blat_df.mutant, blat_df.ddG_stat) if int(mut[1:-1])<=263]
    # # WARNING: we clip mutations at position 263 - mutations go until 286, however pdb is only 263 (main chain) long
    # mutation_dict = {"1BTL" : clipped_mutations}
    #
    # pdb_file = "./pdb/1btl.pdb"
    # contact_map = ContactMapper(pdb_file=pdb_file, tri_dist=True)
    # # contact_map.plot_distance_matrix()
    # # contact_map.plot_contact_map()
    #
    # pcol = ProteinCollection(contact_map, pdb_ID="1BTL", mutations_exp=mutation_dict, mutations_sim={})
    # adjacencies = contact_map.adjacency
    # mut_S_exp, mut_adj_exp, ΔΔg_exp, mut_ids_exp = parse_mutations(mutation_dict=mutation_dict.get(pcol.pdb_ID),
    #                                                 sequence=pcol.sequence, adjacency=adjacencies)
    # X_exp = convert_aa_sequence(mut_S_exp)
    # y_wt = np.array([0.])[:, np.newaxis]
    # X_wt = convert_aa_sequence([pcol.sequence])
    #
    # ΔΔg_exp = np.array(ΔΔg_exp)[:, np.newaxis]
    # ΔΔg_is = np.array([])[:, np.newaxis]
    #
    #  # Scale y-values as done in the implementation by normalizing with mean and max
    # mean_y, max_y, y_wt, ΔΔg_exp, ΔΔg_is_scaled = preprocess_observations(y_wt, ΔΔg_exp,  ΔΔg_is)
    #
    # gpr = GPRegression(protein_representation=pcol, X_wt=X_wt, X_exp=X_exp, X_is=np.array([]),
    #                     y_wt=y_wt, y_exp=ΔΔg_exp, y_is=ΔΔg_is_scaled, adjacencies=adjacencies,
    #                     σ_T=torch.Tensor([0.]), y_max=max_y, y_mean=mean_y, fusion=False)


if __name__ == "__main__":
    cv_options = ["pos_lvl", "mut_lvl"]
    data_options = ["tll", "blat"]
    parser = argparse.ArgumentParser(description="Experiment Module - run specific Regression calls.")
    parser.add_argument("-p", "--pdb", type=str, help="Identifier string of pdb file")
    parser.add_argument("-i", "--idx", type=int, help="Index of CV run")
    parser.add_argument("-r", "--run", type=str, choices=cv_options, help="type of CV routine to run")
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbosity boolean.")
    parser.add_argument("-o", "--optim", action='store_true', help="Run optimization boolean flag.")
    parser.add_argument("-m", "--mode", action='store_true', help="Run reference modus (2 σ) boolean flag.")
    parser.add_argument("--seed", type=int, default=42, help="Randomness seed for replicability.")
    parser.add_argument("-e", "--experiment", type=str, help="Experiment ID for mlflow")
    parser.add_argument("--run_id", type=str, help="Run ID of mlflow run.")
    parser.add_argument("--no_fusion", action="store_true", help="Run mGP instead of mGPfusion, disregard Rosetta simulations.")
    parser.add_argument("--data", type=str, choices=data_options, help="Type of run")
    parser.add_argument("--ref_contact", action="store_true", help="Use reference contactmap from matlab.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.run == "pos_lvl" and not args.no_fusion:
        mutation_dict, is_mutation_dict = None, None
        if args.data=="tll":
            _, mutation_dict, is_mutation_dict = prepare_tll(in_silico="./data/tll/TLL_IS_closed_results.xlsx")
        if args.data=="blat":
            _, mutation_dict, is_mutation_dict = prepare_blat()
        run_mgpfusion_experiment_pos_lvl(pdb=args.pdb, idx=args.idx, optim=args.optim, ref=args.mode,
                                         verbose=args.verbose, run_id=args.run_id, exp_mutation_dict=mutation_dict, 
                                         is_mutation_dict=is_mutation_dict)
    elif args.run == "mut_lvl" and not args.no_fusion:
        mutation_dict, is_mutation_dict = None, None
        if args.data == "tll":
            _, mutation_dict, is_mutation_dict = prepare_tll(in_silico="./data/tll/TLL_IS_closed_results.xlsx")
        run_mgpfusion_experiment_mut_lvl(pdb=args.pdb, idx=args.idx, optim=args.optim, ref=args.mode,
                                         verbose=args.verbose, run_id=args.run_id, exp_mutation_dict=mutation_dict,
                                         is_mutation_dict=is_mutation_dict)
    elif args.run == "pos_lvl" and args.no_fusion:
        if args.data=="blat":
            _, mutations_dict_exp, _ = prepare_blat()
        if args.data=="tll":
            _, mutations_dict_exp, _ = prepare_tll()
        else:
            mutations_dict_exp = parse_matlab_mutation_file("./data/mgp/ddg_protherm.mat", query="ddg_protherm")
        run_pos_lvl_CV_no_fusion(pdb=args.pdb, idx=args.idx, mutation_dict=mutations_dict_exp, optim=args.optim,
                                 use_reference_map=args.ref_contact, run_id=args.run_id)
    else:
        parser.print_help()
        raise RuntimeError("Wrong CV option provided. See help.")
    # run_plotting()
    # run_BLAT_experiment()
    # run_BLAT_experiment_2()
