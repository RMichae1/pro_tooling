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
pdbs = ["1BVC", "2LZM", "1PGA", "1CSP", "1BPI", "1RGG", "1RTB", "2RN2", "4LYZ"]


def get_positions(pdb: str) -> str:
    cm_tri = ContactMapper(pdb_file=f"./pdb/{pdb.lower()}.pdb", tri_dist=True)
    return cm_tri.sequence


def run_sys_CV(pdb, idx, cv, experiment, run_id, data_file, ref=False, optim=True, no_fusion=False, verbose=False):
    command_lst = ["c:/Users/RCML/Anaconda3/envs/mgpfusion/python.exe",
                    "//wsl$/Ubuntu/home/rcml/pro_tooling/run_experiments.py",
                    "-p", f"{pdb}", "-i", f"{idx}", "-r", f"{cv}", "--seed", "3032021", "--experiment", f"{experiment}",
                   "--run_id", f"{run_id}", "--input", f"{data_file}"]
    if optim:
        command_lst += ["-o"]
    if ref:
        command_lst += ["-m"]
    if verbose:
        command_lst += ["-v"]
    if no_fusion:
        command_lst += ["--no_fusion"]
    subprocess.run(command_lst)


def create_mlflow_run(pdb: str, cv: str, optim: bool, ref: str, no_fusion: bool, data_file: str="") -> None:
    sequence = get_positions(pdb)
    experiment_name = f"{pdb}: {cv}"
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment) as run:
        exp_params = {"pdb": pdb, "cv": cv, "optimization": optim, "2σ": ref}
        mlflow.log_params(exp_params)
        for idx, _ in enumerate(sequence):
            # run position lvl no optimization
            run_sys_CV(pdb, idx, cv=cv, ref=ref, optim=optim, experiment=experiment_name, run_id=run.info.run_id, 
                    no_fusion=no_fusion, data_file=data_file, verbose=True)
    # TODO compute overall stats for experiment
    mlflow.end_run()
    return None


def main() -> None:
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    for pdb in pdbs:
        create_mlflow_run(pdb=pdb, cv="pos_lvl", optim=False, ref=False)
        create_mlflow_run(pdb=pdb, cv="pos_lvl", optim=True, ref=False)
        create_mlflow_run(pdb=pdb, cv="pos_lvl", optim=False, ref=True)
        create_mlflow_run(pdb=pdb, cv="pos_lvl", optim=True, ref=True)
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

def run_TLL() -> None:
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    create_mlflow_run(pdb="1TIB", cv="pos_lvl", optim=False, ref=False, no_fusion=True)
    create_mlflow_run(pdb="1TIB", cv="pos_lvl", optim=True, ref=False, no_fusion=True)
    create_mlflow_run(pdb="1TIB", cv="pos_lvl", optim=False, ref=True, no_fusion=True)
    create_mlflow_run(pdb="1TIB", cv="pos_lvl", optim=True, ref=True, no_fusion=True)


def run_BLAT() -> None:
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    create_mlflow_run(pdb="", cv="pos_lvl", optim=False, ref=False, no_fusion=True, data_file="./data/blat/BLAT_ECOLX_Ranganathan2015.csv")
    create_mlflow_run(pdb="", cv="pos_lvl", optim=True, ref=False, no_fusion=True, data_file="./data/blat/BLAT_ECOLX_Ranganathan2015.csv")
    create_mlflow_run(pdb="", cv="pos_lvl", optim=False, ref=True, no_fusion=True, data_file="./data/blat/BLAT_ECOLX_Ranganathan2015.csv")
    create_mlflow_run(pdb="", cv="pos_lvl", optim=True, ref=True, no_fusion=True, data_file="./data/blat/BLAT_ECOLX_Ranganathan2015.csv")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_TLL()
    #main()
    
