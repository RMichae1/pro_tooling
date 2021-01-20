import pickle
import os
import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from graphkernel import KernelLoader
from typing import Tuple

pdbs = ["1PGA"]

def get_mutation_n(protein_representation) -> list:
    """
    gather number of mutations as they occur in gp-regression
    vector has to be as long as predicted mu    
    """
    mutation_n = []
    # mutations include both insilico and experimental
    experimental_mutation_index = get_mutation_idx(protein_representation.mut_ids_exp)
    for pos in range(len(protein_representation.sequence)):
        # check mutations at position and if mutated, add size of mutation
        n = np.array([len(mut) for mut in experimental_mutation_index if bool(pos in mut)])
        mutation_n.append(n)

def load_pickled_file_from_dir(pdb, directory) -> dict:
    for f in os.listdir(directory):
        if not pdb.upper() in f:
            continue
        filename = os.path.join(directory, f)
        with open(filename, "rb") as infile:
            h_file = pickle.load(infile)
    return h_file

def parse_hyperparameter_weights(pdb_id: str, directory: str) -> np.ndarray:
    h_file = load_pickled_file_from_dir(pdb_id, directory)
    weights = h_file.get("w").detach().numpy()
    return weights.reshape(len(weights),).T

def parse_hyperparameter_parameters(pdb_id: str, directory: str):
    h_file = load_pickled_file_from_dir(pdb_id, directory)
    sigma_S = h_file.get("sigma_s").detach().numpy()[0]
    sigma_E = h_file.get("sigma_e").detach().numpy()[0]
    #t = h_file.get("t").detach().numpy()
    return sigma_S, sigma_E

def parse_weights_results(pdb_id, directory="./results/mGPfusion/") -> np.ndarray:
    h_file = load_pickled_file_from_dir(pdb_id, directory)
    weights = [elem.get("w") for elem in h_file.get("optimization")]
    return np.array(weights)

def parse_regression_results(pdb_id, directory="./results/mGPfusion/") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_file = load_pickled_file_from_dir(pdb_id, directory)
    mu = np.array([elem.get("mu") for elem in h_file.get("regression")])
    y_exp = np.arrmu = np.array([elem.get("y_exp") for elem in h_file.get("regression")])
    cov = np.arrmu = np.array([elem.get("cov") for elem in h_file.get("regression")])
    lmls = np.arrmu = np.array([elem.get("lml") for elem in h_file.get("regression")])
    return mu, y_exp, cov, lmls


def plot_hyperparameters(proteins: list, save_fig="./fig/", dir="./results/hyper/"):
    """
    plots weight hyperparameters aquired from neg_ll optimization over all data
    """
    filename = os.path.join(save_fig, "hyper_table.png")
    kernels = KernelLoader()
    ws = np.array([parse_hyperparameter_weights(pdb_id=p, directory=dir) for p in proteins]).T
    assert ws.shape[0] == len(kernels.sub_matrices_ids)
    assert ws.shape[1] == len(proteins)
    df = pd.DataFrame(data=ws, columns=proteins, index=kernels.sub_matrices_ids)
    descriptions = [m_info[0] for _, _, m_info in loadmat("./data/subMats.mat").get('subMats')] 
    df["Description"] = descriptions
    fig, (ax, cbar_ax) = plt.subplots(2, figsize=(5, 20), gridspec_kw={"height_ratios": (.9, .05), "hspace": .3})
    im = sns.heatmap(ws, ax=ax, linewidths=0.5, cmap=sns.cm.rocket_r, #vmax=0.15, 
            cbar_ax=cbar_ax, cbar_kws={"orientation":"horizontal"})
    ax.set_xticklabels(proteins)
    ax.set_yticklabels(df.Description)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, rotation_mode="anchor")
    plt.title("Kernel Weights Table")
    plt.savefig(filename)
    plt.show()

def plot_sigmas(proteins: list, save_fig="./fig", dir="./results/hyper/"):
    """
    plots trainable sigma parameters
    """
    filename = os.path.join(save_fig, "hyper_sigmas.png")
    parameters = np.array([parse_hyperparameter_parameters(pdb_id=p, directory=dir) for p in proteins])
    s_S = parameters[:, 0]
    s_E = parameters[:, 1]
    trainable_params = np.hstack([s_S, s_E]).T
    print(trainable_params)
    print(trainable_params.shape)
    fig, (ax, cbar_ax) = plt.subplots(2, figsize=(5, 5))
    im = sns.heatmap(trainable_params, ax=ax, annot=True, cmap=sns.cm.rocket_r,
            cbar_ax=cbar_ax, cbar_kws={"orientation":"horizontal"})
    ax.set_xticklabels(proteins)
    ax.set_yticklabels(["σ_S", "σ_E"])
    plt.savefig(filename)
    plt.show()

def plot_mean_over_weights(proteins: list, save_fig="./fig", dir="./reults/"):
    filename = os.path.join(save_fig, "weights_mean_table.png")
    kernels = KernelLoader()
    ws = np.array([parse_weights_results(pdb_id=p) for p in proteins])
    mean_ws = np.array([np.mean(weights, axis=1) for weights in ws])
    assert mean_ws.shape[0] == len(kernels.sub_matrices_ids)
    assert mean_ws.shape[1] == len(proteins)
    print(mean_ws)
    df = pd.DataFrame(data=mean_ws, columns=proteins, index=kernels.sub_matrices_ids)
    descriptions = [m_info[0] for _, _, m_info in loadmat("./data/subMats.mat").get('subMats')] 
    df["Description"] = descriptions
    fig, (ax, cbar_ax) = plt.subplots(2, figsize=(5, 20), gridspec_kw={"height_ratios": (.9, .05), "hspace": .3})
    im = sns.heatmap(ws, ax=ax, linewidths=0.5, cmap=sns.cm.rocket_r, #vmax=0.15, 
            cbar_ax=cbar_ax, cbar_kws={"orientation":"horizontal"})
    ax.set_xticklabels(proteins)
    ax.set_yticklabels(df.Description)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, rotation_mode="anchor")
    plt.title("Kernel weights μ \nafter pos.-lvl CV")
    plt.savefig(filename)
    plt.show()


def results_table(rho, rmse, proteins, cv=["pos-lvl. ref", "pos-lvl"], 
    method=["mGPfusion", "mGP"]) -> None:
    """
    rho, rmse are nx2
    """
    # TODO create multiindex: PROT: method1, method2, 
    # Rho: cv-ref, cv
    assert rho.shape[1] == len(cv)
    assert rmse.shape[1] == len(cv)
    df = pd.DataFrame()
    
    x_index = list(zip(proteins, ))
    for idx, p in enumerate(proteins):
        p_df = pd.DataFrame()
        p_df.index = p


def plot_pos_lvl_gpr(proteins:list, results_dir="./results/mGPfusion", 
    save_fig="./fig/", suffix="") -> None:
    filename = os.path.join(save_fig, f"gpr_pos_lvl_{str(proteins)}.png")
    _, ax = plt.subplots(1,1, figsize=(15,10))
    ax.axline((-4, -4), (4,4), color="grey", linestyle="--")
    for p in proteins:
        mu, y_test, _, _ = parse_regression_results(p, directory=results_dir)
        f_μ = np.concatenate([np.atleast_1d(elem) for elem in mu])
        y_test = np.concatenate([elem for sub in y_test for elem in sub])
        plt.scatter(y_test, f_μ, color="indianred")
        # TODO derive mutation color scheme:
        # if mutations:
        #     mutations = [np.repeat(mut, len(y)) for mut, y in zip(mutations, y_test)]
        #     sns.scatterplot(y_test, f_μ, hue=mutations, ax=ax)
    ax.set_xlabel("experimental ΔΔG", fontsize=18)
    ax.set_ylabel("predicted ΔΔG", fontsize=18)
    plt.title(f"GP Regression (position lvl CV) {suffix}")
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_mut_lvl_gpr(proteins:list, results_dir="./results/mGPfusion", save_fig="./fig/") -> None:
    filename = os.path.join(save_fig, f"gpr_mut_lvl_{str(proteins)}.png")
    _, ax = plt.subplots(1,1, figsize=(15,10))
    ax.axline((-4, -4), (4,4), color="grey", linestyle="--")
    for p in proteins:
        mu, y_test, cov, _ = parse_regression_results(p)
        f_μ = np.concatenate([np.atleast_1d(elem) for elem in mu])
        y_test = np.concatenate([elem for sub in y_test for elem in sub])
        plt.scatter(y_test, f_μ, color="indianred")
        # TODO derive mutation color scheme:
        # if mutations:
        #     mutations = [np.repeat(mut, len(y)) for mut, y in zip(mutations, y_test)]
        #     sns.scatterplot(y_test, f_μ, hue=mutations, ax=ax)
    # TODO each mutation has n means and n covariances
    # add gaussians to plot
    for idx, μ, var, y in enumerate(zip(f_μ, cov, y_test)):
        xx = np.arange(-5, 5, 0.1)
        f = norm.pdf(xx, μ, var)
        ax.plot(y+f, xx, "k-")
        # annotate mutations at test point
        # TODO write mutations while doing CV and query here
        # ax.annotate(self.protein.mutation_ids[idx], xy=(y_test, μ), xycoords="data", xytext=(10,10), 
        #     textcoords='offset points')
    ax.set_xlabel("experimental ΔΔG")
    ax.set_ylabel("predicted ΔΔG")
    plt.title(f"GP Regression (mutation lvl CV)")
    plt.legend()
    plt.savefig(filename)
    plt.show()


def plot_log_prob(lml, mutations, x_test, y) -> None:
    _, ax = plt.subplots(1, 1, figsize=(10,10))
    for idx, mut in enumerate(x_test):
        y_train = y.shape[0] - mutations
        log_prob = [(lml.squeeze().detach().numpy()) for lml in lmls]
        sorted_data = np.array(sorted(zip(y_train, log_prob), key= lambda x: x[0]))
        ax.plot(sorted_data[:, 0], sorted_data[:, 1], "r:", alpha=0.5)
        #ax.plot(sorted_data[:, 0], np.mean(sorted_data[:, 1]), "k.")
        # TODO add mean LML
        # TODO add training data points as dots
    ax.set_xlabel("training ΔΔg")
    ax.set_ylabel("log marginal likelihood")
    plt.title(f"Log Marginal Likelihood over training data {self.id}")
    plt.savefig(f"./fig/gpr_logmarginal_{self.id}.png")
    plt.show()
