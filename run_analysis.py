import pickle
import os
import numpy as np
import pandas as pd
import torch
from itertools import product
from tqdm import tqdm
from utility import compute_rmse, compute_ρ
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_squared_error
from visualization import parse_hyperparameter_parameters, plot_weights, plot_pos_lvl_gpr_individual, plot_sigmas
from visualization import plot_results_table
from visualization import generate_results_table 
from visualization import plot_mut_lvl_gpr_total, plot_mut_lvl_gpr_individual
from visualization import plot_pos_lvl_gpr_total, plot_pos_lvl_gpr_individual
from visualization import plot_covariance_matrices, plot_mWDK
from visualization import plot_metrics_method, plot_metric_results_table
from visualization import plot_uncertainty_prediction_gpr
  

def load_pkl_results(pdb: str, cv: str="pos_lvl", opt: str="False", ref: str="False", 
directory: str="./output", fusion: str = False, fraction: str = None):
    run_results = []
    if not fusion: # TODO differentiate between 25%, 50%, 100%
        file_prefix = f"{pdb.upper()}_NO_FUSION_{cv}_opt_{opt}_ref_{ref}"
    else:
        file_prefix = f"{pdb.upper()}_{cv}_opt_{opt}_ref_{ref}"
    file_list = [f for f in os.listdir(directory) if f.startswith(file_prefix) and fraction in f]
    for file in file_list:
        with open(os.path.join(directory, file), "rb") as pkl_file:
            result = pickle.load(pkl_file)
        run_results.append(result)
    return file_list, run_results


def parse_optimization_results(results: list):
    weights = []
    params = []
    for r in results:
        params.append((r.get("sigma_S"), r.get("sigma_E"), r.get("t"), r.get("nll_init"), r.get("nll_end")))
        weights.append(r.get("w"))
    return weights, params


def parse_prediction_results(results: list):
    metrics = []
    predictions = []
    for r in results:
        mutations = r.get("n_mut")
        if not r.get('spearman corr'):
            spearman_r = spearmanr(r.get("mu"), r.get("y_exp")).correlation
            mse = mean_squared_error(r.get("mu"), r.get("y_exp"))
            metrics.append((spearman_r, mse))
        else:
            metrics.append((r.get('spearman corr')[0], r.get('mse')))
        if r.get("mu") is None and bool(r.get("regression")): # BLAT parsing is different
            mutations = r.get("mutations")
            r = r.get("regression")
        if r.get("mu") is None:
            continue
        mu = np.atleast_1d(r.get("mu"))
        cov = np.atleast_1d(r.get("cov"))
        if cov.shape[0] >= 2:
            cov = cov.diagonal()
        y_exp = r.get("y_exp").flatten()
        assert len(mu) == len(y_exp) == len(cov)
        predictions.append((mu, cov, y_exp, mutations))
    return predictions, metrics


def annotate_df(df: pd.DataFrame, cv: str, opt: bool, ref: bool, pdb: str, 
                method: str, fraction: str) -> pd.DataFrame:
    df["CV"] = cv
    df["optimization"] = opt
    df["reference"] = ref
    df["pdb"] = pdb
    df["method"] = method
    df["fraction"] = fraction
    return df


def create_evaluation_df(metrics: list, file_list: list, pdb: str, method: str, training: str):
    r = [m[0] for m in metrics]
    mse = [m[1] for m in metrics]
    positions = [int(f.split("_")[-2]) for f in file_list] # derive from file_list
    data = {"spearman r": r, "mse": mse, "position": positions}
    eval_df = pd.DataFrame(data, columns=["spearman r", "mse", "position"])
    eval_df["method"] = method
    eval_df["training"] = training
    eval_df["pdb"] = pdb
    return eval_df


def create_prediction_df(parsed_results: list, pdb: str, ref: bool, opt: bool, cv: str, method: str, fraction: str):
    μ = np.concatenate([e[0] for e in parsed_results])
    cov = np.concatenate([e[1] for e in parsed_results])
    y_exp = np.concatenate([e[2] for e in parsed_results])
    n_mut = np.concatenate([np.atleast_1d(e[3]) for e in parsed_results])
    data = {"mu": μ, "cov": cov, "y": y_exp, "mutations": n_mut}
    pred_df = pd.DataFrame(data, columns=["mu", "cov", "y", "mutations"])
    pred_df = annotate_df(pred_df, cv, opt, ref, pdb, method, fraction)
    return pred_df


def create_params_df(parsed_params: list, pdb:str, ref:bool, opt:bool, cv:str):
    sigma_S = np.concatenate([e[0] for e in parsed_params]).flatten()
    sigma_E = np.concatenate([e[1] for e in parsed_params]).flatten()
    t = np.concatenate([e[2] for e in parsed_params]).flatten()
    nll_init = np.concatenate([e[3] for e in parsed_params]).flatten()
    nll_end = np.concatenate([e[4] for e in parsed_params]).flatten()
    data = {"sigma_S": sigma_S, "sigma_E": sigma_E, "t": t, "neg_ll_init": nll_init, "neg_ll_end": nll_end}
    param_df = pd.DataFrame(data)
    return annotate_df(param_df, cv=cv, opt=opt, ref=ref, pdb=pdb)


def create_weights_df(ws, pdb, ref, opt, cv):
    ws = {"weights": np.array(ws).mean(axis=0).flatten()}
    df = pd.DataFrame(ws)
    return annotate_df(df, cv=cv, opt=opt, ref=ref, pdb=pdb)


def compute_metrics(parsed_results: list):
    μ = np.concatenate([e[0] for e in parsed_results])
    y_exp = np.concatenate([e[2] for e in parsed_results])
    rho = compute_ρ(y_exp, μ)
    rmse = compute_rmse(y_exp, μ)
    r = spearmanr(y_exp, μ)[0]
    return rho, rmse, r


def plot_regression_results(df):
    plot_uncertainty_prediction_gpr(df, method="mGP")
    plot_uncertainty_prediction_gpr(df, method="mGP_dELBO")
    plot_uncertainty_prediction_gpr(df, method="mGP_DESkernel")
    plot_pos_lvl_gpr_individual(df, method="mGP", x_range=(-5, 5), y_range=(-5, 5), x_n=3)
    plot_pos_lvl_gpr_individual(df, method="mGP_dELBO", x_range=(-5, 5), y_range=(-5, 5), x_n=3)
    plot_pos_lvl_gpr_individual(df, method="mGP_DESkernel", x_range=(-5, 5), y_range=(-5, 5), x_n=3)
    plot_pos_lvl_gpr_individual(df, method="mGP_dELBO_DESkernel", x_range=(-5, 5), y_range=(-5, 5), x_n=3)
    # plot total pos lvl
    plot_pos_lvl_gpr_total(df, method="mGP")
    plot_pos_lvl_gpr_total(df, method="mGP_dELBO")
    plot_pos_lvl_gpr_total(df, method="mGP_DESkernel")
    plot_pos_lvl_gpr_total(df, method="mGP_dELBO_DESkernel")


def plot_metrics_results(df):
    # plot individual position lvl
    plot_metrics_method(df, method="mGP")
    plot_metrics_method(df, method="mGP_dELBO")
    plot_metrics_method(df, method="mGP_DESkernel")
    plot_metrics_method(df, method="mGP_dELBO_DESkernel")


def results_table(df, metric_df):
    result_df = generate_results_table(df)
    plot_results_table(result_df)
    plot_metric_results_table(metric_df)
    #print(result_df)
    print(result_df.to_latex(index=False, bold_rows=True))
    average_metric_df = metric_df.groupby(['method', 'pdb', 'training']).mean().reset_index().drop(columns="position")
    print(average_metric_df.to_latex(index=False, bold_rows=True))


def plot_hyperparameters(hyper_df, weight_df):
    plot_weights(weight_df)
    plot_weights(weight_df, opt=True, ref=False)
    plot_weights(weight_df, opt=True, ref=False)
    plot_sigmas(hyper_df)
    #plot_neg_ll(hyper_df)


def load_regression_results(pdbs, method_dirs, optim, refs, cvs, fusion, fractions):
    prediction_frames = []
    metric_frames = []
    parameter_combinations = product(method_dirs, pdbs, optim, refs, cvs, fusion, fractions)
    for method, pdb, opt, ref, cv, f, frac in parameter_combinations:
        results_dir = f"./results/{method}/{pdb.lower()}/{cv}/"
        file_list, results = load_pkl_results(pdb, cv=cv, opt=opt, ref=ref, fusion=f, directory=results_dir,
                                    fraction=frac)
        print(f"{pdb} {method}")
        predictions, metrics = parse_prediction_results(results)
        if not predictions:
            print(f"! No predictions found for: {method}, {cv}, optimization:{opt}, fusion:{f} \n in {results_dir}")
            continue
        metric_df = create_evaluation_df(file_list=file_list, metrics=metrics, pdb=pdb, method=method, training=frac)
        pred_df = create_prediction_df(predictions, pdb, ref, opt, cv=cv, method=method, fraction=frac)
        prediction_frames.append(pred_df)
        metric_frames.append(metric_df)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    metrics_df = pd.concat(metric_frames, ignore_index=True)
    return predictions_df, metrics_df


def load_hyperparameter_results(pdbs, optim, refs):
    frames = []
    w_frames = []
    for pdb in pdbs:
        if pdb == "1FQG": # skip different format
            continue
        for opt in optim:
            for ref in refs:
                results = load_pkl_results(pdb, opt=opt, ref=ref)
                ws, params = parse_optimization_results(results)
                w_df = create_weights_df(ws, pdb, ref, opt, cv="pos_lvl")
                w_frames.append(w_df)
                p_df = create_params_df(params, pdb, ref, opt, cv="pos_lvl")
                frames.append(p_df) 
    params_df = pd.concat(frames, ignore_index=True)
    weight_df = pd.concat(w_frames, ignore_index=True)
    return weight_df, params_df


if __name__ == "__main__":
    #pdbs = ["1BVC", "2LZM", "1PGA", "1CSP", "1BPI", "1RGG", "1RTB", "2RN2", "4LYZ"]# "1FQG"]
    pdbs = ["1FQG", "1UBQ", "1PGA"]
    method_dirs = ["mGP", "mGP_dELBO", "mGP_DESkernel", "mGP_dELBO_DESkernel"]
    optim = [True]
    refs = [False]
    kernel = [False, True]
    fusion = [False, True]
    cvs = ["pos_lvl"]
    fractions = ["0.25", "0.5", "1.0"]
    # TODO load/derive rho of VAE
    regression_df, metrics_df = load_regression_results(pdbs=pdbs, method_dirs=method_dirs, 
                                    optim=optim, refs=refs, cvs=cvs, fusion=fusion,
                                    fractions=fractions)
    plot_metrics_results(metrics_df)
    plot_regression_results(regression_df)
    results_table(regression_df, metrics_df)
    weight_df, hyperparameters_df = load_hyperparameter_results(pdbs, optim, refs)
    #plot_hyperparameters(hyperparameters_df, weight_df)

    # plot_mut_lvl_gpr_total(proteins=pdbs, results_dir="./results/mGPfusion/mut_cv/")
    # plot_mut_lvl_gpr_individual(proteins=pdbs, results_dir="./results/mGPfusion/mut_cv/")
    # plot_mut_lvl_gpr_individual(proteins=pdbs, results_dir="./results/mGPfusion/mut_cv/", suffix="uncertainties",
    #         uncertainties=True)