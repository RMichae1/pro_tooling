import pickle
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utility import compute_rmse, compute_ρ
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_squared_error
from visualization import plot_hyperparameters, plot_mean_over_weights, plot_pos_lvl_gpr_individual, plot_sigmas
from visualization import generate_results_table, generate_total_results_table 
from visualization import plot_mut_lvl_gpr_total, plot_mut_lvl_gpr_individual
from visualization import plot_pos_lvl_gpr_total, plot_pos_lvl_gpr_individual
from visualization import plot_covariance_matrices, plot_mWDK

    # TODO handle results
    # POS LVL
    # predictions = np.concatenate([np.atleast_1d(x) for x in [elem.get('mu') for elem in fit_parameters]])
    # experimental = np.concatenate([x for sub in [elem.get('y_exp') for elem in fit_parameters] for x in sub])
    # rho = compute_ρ(y_vec=experimental, y_pred_μ=predictions)
    # rmse = compute_rmse(y=experimental, y_pred_μ=predictions)
    #
    #
    # MUT LVL
    # predictions = np.concatenate([np.atleast_1d(x) for x in [elem.get('mu') for elem in fit_parameters]])
    # experimental = np.concatenate([x for sub in [elem.get('y_exp') for elem in fit_parameters] for x in sub])
    

def load_pkl_results(pdb: str, cv: str="pos_lvl", opt: str="False", ref: str="False", directory: str="./output"):
    run_results = []
    file_prefix = f"{pdb.upper()}_{cv}_opt_{opt}_ref_{ref}"
    file_list = [f for f in os.listdir(directory) if f.startswith(file_prefix)]
    for file in file_list:
        with open(os.path.join(directory, file), "rb") as pkl_file:
            result = pickle.load(pkl_file)
        run_results.append(result)
    return run_results


def create_optimization_df():
    pass


def parse_prediction_results(results: list):
    predictions = []
    for r in results:
        if r.get("mu") is None:
            continue
        mu = np.atleast_1d(r.get("mu"))
        y_exp = r.get("y_exp").flatten()
        assert len(mu) == len(y_exp)
        predictions.append((mu, y_exp, r.get("n_mut")))
    return predictions


def compute_metrics(parsed_results: list):
    μ = np.concatenate([e[0] for e in parsed_results])
    y_exp = np.concatenate([e[1] for e in parsed_results])
    n_mut = np.concatenate([e[2] for e in parsed_results])
    rho = compute_ρ(y_exp, μ)
    rmse = compute_rmse(y_exp, μ)
    r = spearmanr(y_exp, μ)[0]
    return rho, rmse, r


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



if __name__ == "__main__":
    pdbs = ["1BVC", "2LZM", "1PGA", "1CSP", "1BPI", "1RGG", "1RTB", "2RN2", "4LYZ", "1FQG"]
    optim = ["False", "True"]
    refs = ["False", "True"]
    for pdb in pdbs:
        for opt in optim:
            for ref in refs:
                results = load_pkl_results(pdb, opt=opt, ref=ref)
                predictions = parse_prediction_results(results)
                if not predictions:
                    continue
                rho, rmse, r = compute_metrics(predictions)
                print(f"{pdb} - opt:{opt}, ref:{ref}")
                print(f"rho={rho}, rmse={rmse}, r={r}")