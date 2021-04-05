from scipy.io import loadmat
import os
import sys
import argparse
import torch
from utility import compute_rmse, compute_ρ
from utility import get_mutation_idx
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import tracemalloc
from experiment import Experiment
import numpy as np
import pandas as pd
import pickle
import mlflow
from mlflow.tracking import MlflowClient


def run_mgpfusion_experiment_pos_lvl(experiment: Experiment, verbose=True, write=True) -> None:
    if verbose:
        print(f"{experiment.pdb} - pos: {experiment.idx},  optim: {experiment.optimization}, reference: {experiment.ref}")
    # TODO make mutation index an experiment property
    experimental_mutation_index = get_mutation_idx(experiment.protein.mut_ids_exp)
    # gather all mutations at that position and assign train and test indices
    mutation_bool_mask = np.array([bool(experiment.idx in mut) for mut in experimental_mutation_index])
    test_mutation_idx = np.where(mutation_bool_mask)[0]
    not_test_mutation_idx = np.where(~mutation_bool_mask)[0]
    if len(test_mutation_idx) == 0:
        print(f"No Mutation at pos:{experiment.idx} - skipping...")
        return
    n_mutations = np.array([len(mut) for mut in experimental_mutation_index if bool(experiment.idx in mut)])
    # split into train and test
    experiment.gpr.set_test_index(1 + test_mutation_idx)  # offset with WT
    # combine WT + not selected + in silico for training data
    train_index = np.concatenate([np.array([0]), 1 + not_test_mutation_idx,
                                  np.arange(start=len(experiment.gpr.X_exp) + 1,
                                            stop=experiment.gpr.X.shape[0])])  # all simulated data are training data
    experiment.gpr.set_train_index(train_index)
    # optimize
    nll_init = experiment.gpr.neg_ll()
    if experiment.optimization:
        try:
            experiment.gpr.parameter_optimization()
        except RuntimeError as _:
            print("Optimization broke.")
            experiment.gpr.reset_trainable_parameters()
    nll_end = experiment.gpr.neg_ll()
    f_μ, cov = experiment.gpr._fit(ref=experiment.two_sigma)
    # write optimization results
    opt_params = {"w": experiment.gpr.weights.get_value().detach().numpy(),
                  "sigma_S": experiment.gpr.σ_S.get_value().detach().numpy(),
                  "sigma_E": experiment.gpr.σ_E.get_value().detach().numpy(),
                  "t": experiment.gpr.t.get_value().detach().numpy(),
                  "nll_init": nll_init.detach().numpy(),
                  "nll_end": nll_end.detach().numpy()}
    fit_params = {'mu': f_μ.squeeze().detach().numpy(),
                  'cov': cov.squeeze().detach().numpy(),
                  'y_exp': (experiment.gpr.y_test.detach().numpy() * experiment.gpr.y_max) + experiment.gpr.y_mean
                  } # TODO make this part of the experiment wrapper
    client = MlflowClient()
    run = client.get_run(experiment.run_id)
    print(run.info.run_id)
    spearman_r, spearman_p = spearmanr(fit_params.get('mu'), fit_params.get("y_exp"))
    mse = mean_squared_error(np.atleast_1d(fit_params.get('mu')), np.atleast_1d(fit_params.get("y_exp")))
    client.log_metric(run_id=run.info.run_id, key="spearman r", 
                    value=spearman_r, step=experiment.idx)
    client.log_metric(run_id=run.info.run_id, key="spearman p", 
                    value=spearman_p, step=experiment.idx)
    client.log_metric(run_id=run.info.run_id, key="mse", value=mse, 
                    step=experiment.idx)
    filename = f"./output/{experiment.pdb}_pos_lvl_opt_{experiment.optimization}_ref_{experiment.two_sigma}_{experiment.idx}.pkl"
    if write and bool(opt_params):
        data_dict = {**opt_params, **fit_params, "idx": experiment.idx, "n_mut": n_mutations}
        with open(filename, "wb") as outfile:
            pickle.dump(data_dict, outfile)
        client.log_artifact(run.info.run_id, filename)
    # mlflow.end_run()
    return


def run_mgpfusion_experiment_mut_lvl(experiment: Experiment, verbose=False, write=True) -> None:
    """
    Runs Loo CV routine on experiment
    """
    if verbose:
        print(f"{experiment.pdb} - pos: {experiment.idx},  optim: {experiment.optimization}, reference: {experiment.two_sigma}")
    # get all experimental mutations incl WT
    if experiment.idx == 0:  # exclude WT from CV
        print("WT excluded from LOO")
        return
    # set train and testing indices
    experiment.gpr.set_train_index(np.delete(np.arange(0, experiment.gpr.X.shape[0]), experiment.idx))
    experiment.gpr.set_test_index(np.array([experiment.idx]))
    n_mutations = experiment.gpr.protein.mutation_ids[experiment.idx].count(")")  # get mutations by closing brackets on tuple
    nll_init = experiment.gpr.neg_ll()
    if experiment.optimization:
        try:
            experiment.gpr.parameter_optimization()
        except RuntimeError as _:
            print("Optimization broke.")
            experiment.gpr.reset_trainable_parameters()
    nll_end = experiment.gpr.neg_ll()
    f_μ, cov = experiment.gpr._fit(ref=experiment.two_sigma)
    opt_params = {"w": experiment.gpr.weights.get_value().detach().numpy(),
                  "sigma_S": experiment.gpr.σ_S.get_value().detach().numpy(),
                  "sigma_E": experiment.gpr.σ_E.get_value().detach().numpy(),
                  "t": experiment.gpr.t.get_value().detach().numpy(),
                  "nll_init": nll_init.detach().numpy(),
                  "nll_end": nll_end.detach().numpy()}
    fit_params = {'mu': f_μ.squeeze().detach().numpy(),
                  'cov': cov.squeeze().detach().numpy(),
                  'y_exp': (experiment.gpr.y_test.detach().numpy() * experiment.gpr.y_max) + experiment.gpr.y_mean
                  }
    client = MlflowClient()
    run = client.get_run(experiment.run_id)
    print(run.info.run_id)
    spearman_r, spearman_p = spearmanr(fit_params.get('mu'), fit_params.get("y_exp"))
    mse = mean_squared_error(np.atleast_1d(fit_params.get('mu')), np.atleast_1d(fit_params.get("y_exp")))
    client.log_metric(run_id=run.info.run_id, key="spearman r", value=spearman_r, step=experiment.idx)
    client.log_metric(run_id=run.info.run_id, key="spearman p", value=spearman_p, step=experiment.idx)
    client.log_metric(run_id=run.info.run_id, key="mse", value=mse, step=experiment.idx)
    filename = f"./output/{experiment.pdb}_mut_lvl_opt_{experiment.optimization}_ref_{experiment.two_sigma}_{experiment.idx}.pkl"
    if write and bool(opt_params):
        data_dict = {**opt_params, **fit_params, "idx": experiment.idx, "n_mut": n_mutations}
        with open(filename, "wb") as outfile:
            pickle.dump(data_dict, outfile)
        client.log_artifact(run.info.run_id, filename)
    # mlflow.end_run()
    return


def run_pos_lvl_CV_no_fusion(experiment: Experiment, write: bool = True) -> dict:
    experimental_mutation_index = get_mutation_idx(experiment.protein.mut_ids_exp)
    # gather all mutations at that position and assign train and test indices
    mutation_bool_mask = np.array([bool(experiment.idx in mut) for mut in experimental_mutation_index])
    test_mutation_idx = np.where(mutation_bool_mask)[0]
    if len(test_mutation_idx) == 0:
        print(f"No Mutation at pos:{experiment.idx} - skipping...")
        return
    not_test_mutation_idx = np.where(~mutation_bool_mask)[0]
    n_mutations = np.array([len(mut) for mut in experimental_mutation_index if bool(experiment.idx in mut)])
    # split into train and test
    experiment.gpr.set_test_index(1 + test_mutation_idx)  # offset with WT
    # combine WT + not selected + in silico for training data
    train_index = np.concatenate([np.array([0]), 1 + not_test_mutation_idx,
                                  np.arange(start=len(experiment.gpr.X_exp) + 1,
                                            stop=experiment.gpr.X.shape[0])])  # all simulated data are training data
    experiment.gpr.set_train_index(train_index)
    # optimize
    nll_init = experiment.gpr.neg_ll()
    if experiment.optimization:
        try:
            experiment.gpr.parameter_optimization()
        except RuntimeError as _:
            print("Optimization broke.")
            experiment.gpr.reset_trainable_parameters()
    nll_end = experiment.gpr.neg_ll()
    f_μ, cov = experiment.gpr._fit(ref=experiment.two_sigma)
    # write optimization results
    optimization_params = {"w": experiment.gpr.weights.get_value(),
                           "sigma_S": experiment.gpr.σ_S.get_value(),
                           "sigma_E": experiment.gpr.σ_E.get_value(),
                           "t": experiment.gpr.t.get_value(),
                           "nll": (nll_init, nll_end)}
    mutations = n_mutations
    fit_params = {'mu': f_μ.squeeze().detach().numpy(),
                  'cov': cov.squeeze().detach().numpy(),
                  'y_exp': (experiment.gpr.y_test.detach().numpy() * experiment.gpr.y_max) + experiment.gpr.y_mean
                  }
    client = MlflowClient()
    run = client.get_run(experiment.run_id)
    print(run.info.run_id)
    spearman_r, spearman_p = spearmanr(fit_params.get('mu'), fit_params.get("y_exp"))
    mse = mean_squared_error(np.atleast_1d(fit_params.get('mu')), np.atleast_1d(fit_params.get("y_exp")))
    client.log_metric(run_id=run.info.run_id, key="spearman r", 
                    value=spearman_r, step=experiment.idx)
    client.log_metric(run_id=run.info.run_id, key="spearman p", 
                    value=spearman_p, step=experiment.idx)
    client.log_metric(run_id=run.info.run_id, key="mse", 
                    value=mse, step=experiment.idx)
    results = {"optimization": optimization_params,
               "regression": fit_params,
               "mutations": mutations,
               "spearman corr": (spearman_r, spearman_p),
               "mse": mse}
    filename = f"./output/{experiment.pdb}_NO_FUSION_pos_lvl_opt_{experiment.optimization}_ref_{experiment.two_sigma}_{experiment.idx}.pkl"
    if write:
        with open(filename, "wb") as filehandle:
            pickle.dump(results, filehandle)
        client.log_artifact(run.info.run_id, filename)
    return


if __name__ == "__main__":
    cv_options = ["pos_lvl", "mut_lvl"]
    data_options = ["tll", "blat", "mgpf"]
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
    parser.add_argument("--no_fusion", action="store_true",
                        help="Run mGP instead of mGPfusion, disregard Rosetta simulations.")
    parser.add_argument("--data", type=str, choices=data_options, help="Select type of experiment run.")
    parser.add_argument("--ref_contact", action="store_true", help="Use reference contactmap from matlab.")
    parser.add_argument("--vae_input", action="store_true", help="Use VAE ELBO input for fusion.")
    parser.add_argument("--vae_kernel", action="store_true", help="Use VAE derived substitution kernel")
    parser.add_argument("--experimental_data", type=str, help="Provide filename for experimental (csv) data.")
    parser.add_argument("--simulated_data", type=str, help="Provide filename for in silico (csv) data.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    experiment = Experiment(pdb=args.pdb, experiment_type=args.data, idx=args.idx, optimization=args.optim, 
                        fusion=bool(not args.no_fusion), reference=args.mode, run_id=args.run_id, 
                        vae_kernel=args.vae_kernel, vae_input=args.vae_input,
                        exp_data_filename=args.experimental_data, is_data_filename=args.simulated_data)
    if args.run == "pos_lvl" and args.no_fusion:
        run_pos_lvl_CV_no_fusion(experiment)
    elif args.run == "pos_lvl":
        run_mgpfusion_experiment_pos_lvl(experiment)
    elif args.run == "mut_lvl":
        run_mgpfusion_experiment_mut_lvl(experiment)
    else:
        parser.print_help()
        raise RuntimeError("Wrong CV option provided. See help.")

    # run_plotting()
    # run_BLAT_experiment()
    # run_BLAT_experiment_2()
