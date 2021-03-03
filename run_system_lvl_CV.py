from scipy.io import loadmat
import os
import warnings
import subprocess
import mlflow
from contact_mapper import ContactMapper
from protein_representation import ProteinCollection
from data_scaler import BayesScaler
from gp_regression import GPRegression
import torch
from tqdm import tqdm
from utility import compute_rmse, compute_ρ
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import pickle

todos = []
done = []
pdbs = ["2LZM", "1BVC","1PGA","1CSP", "1BPI", "1RGG", "1RTB", "2RN2", "4LYZ"]


def get_positions(pdb: str) -> str:
    pga_file = loadmat(os.path.join(os.path.dirname(__file__), os.path.join("data/mgp/", f"{pdb}.mat")))
    cm_tri = ContactMapper(pdb_file=f"./pdb/{pdb.lower()}.pdb", tri_dist=True)
    return cm_tri.sequence


def run_sys_CV(pdb, idx, cv, ref=False, optim=True, verbose=False):
    command_lst = ["c:/Users/rmich/miniconda3/envs/mgpfusion/python.exe", 
                    "c:/pro_tooling/run_experiments.py", 
                    "-p", f"{pdb}", "-i", f"{idx}", "-r", f"{cv}", "--seed", 3032021]
    if optim:
        command_lst += ["-o"]
    if ref:
        command_lst += ["-m"]
    if verbose:
        command_lst += ["-v"]
    subprocess.run(command_lst)


def create_mlflow_run(pdb: str, cv: str, optim: bool, ref: str, name: str) -> None:
    sequence = get_positions(pdb)
    experiment = mlflow.create_experiment(f"{pdb}: {name}")
    mlf_run = mlflow.start_run(experiment_id=experiment, run_name="")
    exp_params = {"pdb": pdb, "cv": cv, "optimization": optim, "2σ": ref}
    mlflow.log_params(exp_params)
    for idx, _ in enumerate(sequence):
        # run position lvl no optimization
        run_sys_CV(pdb, idx, cv=cv, ref=ref, optim=optim, verbose=True)
    # TODO compute overall stats for experiment
    return None


def main() -> None:
    mlflow.set_tracking_uri()
    for pdb in pdbs:
        create_mlflow_run(pdb=pdb, cv="pos_lvl", optim=False, ref=False, 
                        name="mGPfusion run - pos-lvl NO OPTIM")
        create_mlflow_run(pdb=pdb, cv="pos_lvl", optim=True, ref=False, 
                        name="mGPfusion run - pos-lvl OPTIM")
        create_mlflow_run(pdb=pdb, cv="pos_lvl", optim=False, ref=True, 
                        name="mGPfusion run - pos-lvl 2σ NO OPTIM")
        create_mlflow_run(pdb=pdb, cv="pos_lvl", optim=True, ref=True, 
                        name="mGPfusion run - pos-lvl 2σ OPTIM")
        done.append(pdb)
    print(f"Done: {done}")
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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()