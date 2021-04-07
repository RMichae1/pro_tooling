import pickle
import os
import numpy as np
import pandas as pd
import torch
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
  

def load_pkl_results(pdb: str, cv: str="pos_lvl", opt: str="False", ref: str="False", directory: str="./output"):
    run_results = []
    file_prefix = f"{pdb.upper()}_{cv}_opt_{opt}_ref_{ref}"
    file_list = [f for f in os.listdir(directory) if f.startswith(file_prefix)]
    for file in file_list:
        with open(os.path.join(directory, file), "rb") as pkl_file:
            result = pickle.load(pkl_file)
        run_results.append(result)
    return run_results


def parse_optimization_results(results: list):
    weights = []
    params = []
    for r in results:
        params.append((r.get("sigma_S"), r.get("sigma_E"), r.get("t"), r.get("nll_init"), r.get("nll_end")))
        weights.append(r.get("w"))
    return weights, params


def parse_prediction_results(results: list):
    predictions = []
    for r in results:
        mutations = r.get("n_mut")
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
    return predictions


def annotate_df(df: pd.DataFrame, cv: str, opt: bool, ref: bool, pdb: str) -> pd.DataFrame:
    df["CV"] = cv
    df["optimization"] = opt
    df["reference"] = ref
    df["pdb"] = pdb
    return df


def create_prediction_df(parsed_results: list, pdb: str, ref: bool, opt: bool, cv: str):
    μ = np.concatenate([e[0] for e in parsed_results])
    cov = np.concatenate([e[1] for e in parsed_results])
    y_exp = np.concatenate([e[2] for e in parsed_results])
    n_mut = np.concatenate([e[3] for e in parsed_results])
    data = {"mu": μ, "cov": cov, "y": y_exp, "mutations": n_mut}
    pred_df = pd.DataFrame(data, columns=["mu", "cov", "y", "mutations"])
    pred_df = annotate_df(pred_df, cv, opt, ref, pdb)
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
    # plot individual position lvl
    plot_pos_lvl_gpr_individual(df, opt=False, ref=False)
    plot_pos_lvl_gpr_individual(df, opt=True, ref=False)
    plot_pos_lvl_gpr_individual(df, opt=False, ref=True)
    plot_pos_lvl_gpr_individual(df, opt=True, ref=True)
    # plot total pos lvl
    plot_pos_lvl_gpr_total(df, opt=False, ref=False)
    plot_pos_lvl_gpr_total(df, opt=True, ref=False, suffix="optimized")
    plot_pos_lvl_gpr_total(df, opt=False, ref=True, suffix="2σ")
    plot_pos_lvl_gpr_total(df, opt=True, ref=True, suffix="optimized 2σ")


def results_table(df):
    result_df = generate_results_table(df)
    plot_results_table(result_df)
    #print(result_df)
    print(result_df.to_latex(index=True, bold_rows=True))


def plot_hyperparameters(hyper_df, weight_df):
    plot_weights(weight_df)
    plot_weights(weight_df, opt=True, ref=False)
    plot_weights(weight_df, opt=True, ref=False)
    plot_sigmas(hyper_df)
    #plot_neg_ll(hyper_df)


def load_regression_results(pdbs, optim, refs):
    frames = []
    for pdb in pdbs:
        for opt in optim:
            for ref in refs:
                results = load_pkl_results(pdb, opt=opt, ref=ref)
                predictions = parse_prediction_results(results)
                if not predictions:
                    continue
                pred_df = create_prediction_df(predictions, pdb, ref, opt, cv="pos_lvl")
                frames.append(pred_df) 
    predictions_df = pd.concat(frames, ignore_index=True)
    return predictions_df


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
    pdbs = ["1TIB"]
    optim = [False, True]
    refs = [False, True]
    
    regression_df = load_regression_results(pdbs, optim, refs)
    #plot_regression_results(regression_df)
    results_table(regression_df)
    weight_df, hyperparameters_df = load_hyperparameter_results(pdbs, optim, refs)
    #plot_hyperparameters(hyperparameters_df, weight_df)

    # plot_mut_lvl_gpr_total(proteins=pdbs, results_dir="./results/mGPfusion/mut_cv/")
    # plot_mut_lvl_gpr_individual(proteins=pdbs, results_dir="./results/mGPfusion/mut_cv/")
    # plot_mut_lvl_gpr_individual(proteins=pdbs, results_dir="./results/mGPfusion/mut_cv/", suffix="uncertainties",
    #         uncertainties=True)