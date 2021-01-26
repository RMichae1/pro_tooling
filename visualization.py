import pickle
import os
import numpy as np
from numpy.core.shape_base import atleast_1d
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from graphkernel import KernelLoader
from protein_representation import ProteinCollection
from typing import Tuple
from utility import get_mutation_idx
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_squared_error

colormap = ["grey", "black", "yellow", "blue", "red", "pink", "orange", "lightblue", "green", "darkred"]
legend_circles = [Line2D([0], [0], marker="o", markersize=15, color=c, label=str(m)) for c, m in zip(colormap, np.arange(1,11,1))]
mutation_legend_handle = legend_circles

matrix_legend = ['22-29 PAM \n(Benner et al., 1994)', 'Residue Replace \n(Cserzo et al., 1994)', 
    'Initially aligning \n(Gonnet et al., 1992)', 'BLOSUM45 \n(Henikoff-Henikoff, 1992)',
    'BLOSUM62 \n(Henikoff-Henikoff, 1992)', 'BLOSUM80 \n (Henikoff-Henikoff, 1992)', 
    'Structure comparison \n(Luthy et al. 1991)', 'Structure comparison \n(Luthy et al., 1991)', 
    'Structure comparison alpha helix \n(Luthy et al., 1991)', 'Structure comparison beta strand \n(Luthy et al., 1991)',
    'Chemical similarity scores \n (McLachlan, 1972)', 'EMPAR \n(Mohana Rao, 1987)', 
    'Structure correlation matrix 1 \n(Niefind-Schomburg, 1991)', 
    'Cross-correlation main chain \n(Qu et al., 1993)', 'Cross-correlation side chain \n(Qu et al., 1993)',
    'The mutant spatial preference \n(Qu et al., 1993)', 'isomorphicity replacements \n(Tudos et al., 1990)',
    'BLOSUM50 \n(Henikoff-Henikoff, 1992)', 'PHAT \n(Ng et al., 2000)', 
    'SLIM \n(Mueller et al., 2001)', 'Dirichlet Mixture Model \n(Crooks-Brenner, 2005)']

def get_mutation_number_ref_cv(protein_representation) -> list:
    """
    gather number of mutations as they occur in gp-regression
    vector has to be as long as predicted mu    
    """
    mutation_n = []
    # TODO
    # mutations include both insilico and experimental
    experimental_mutation_index = get_mutation_idx(protein_representation.mutation_ids)
    for pos in range(len(protein_representation.sequence)):
        # check mutations at position and if mutated, add size of mutation
        n = np.array([len(mut) for mut in experimental_mutation_index if bool(pos in mut)])
        mutation_n.append(n)

def get_mutation_number_pos_lvl(protein_representation: ProteinCollection) -> list:
    mutation_n = []
    mutation_idx = get_mutation_idx(protein_representation.mut_ids_exp)
    for pos in range(len(protein_representation.sequence)):
        n = np.array([len(mut) for mut in mutation_idx if bool(pos in mut)])
        mutation_n.append(n)

def load_pickled_file_from_dir(pdb, directory) -> dict:
    h_file=None
    for f in os.listdir(directory):
        if not pdb.upper() in f:
            continue
        filename = os.path.join(directory, f)
        with open(filename, "rb") as infile:
            h_file = pickle.load(infile)
        return h_file


def parse_hyperparameter_weights(pdb_id: str, directory: str) -> np.ndarray:
    h_file = load_pickled_file_from_dir(pdb_id, directory)
    if not h_file:
        return None
    weights = h_file.get("w").detach().numpy()
    return weights.reshape(len(weights),).T


def parse_hyperparameter_parameters(pdb_id: str, directory: str):
    h_file = load_pickled_file_from_dir(pdb_id, directory)
    if h_file is None or not h_file.get("sigma_s") or not h_file.get("sigma_e"):
        return None, None
    sigma_S = h_file.get("sigma_s").detach().numpy()[0]
    sigma_E = h_file.get("sigma_e").detach().numpy()[0]
    #t = h_file.get("t").detach().numpy()
    return sigma_S, sigma_E


def parse_mean_weights_results(pdb_id, directory="./results/mGPfusion/") -> np.ndarray:
    h_file = load_pickled_file_from_dir(pdb_id, directory)
    if not h_file:
        return None
    weights = np.array([elem.get("w").detach().numpy() for elem in h_file.get("optimization")])
    weights = weights.mean(axis=0)
    return weights


def parse_regression_results(pdb_id, directory="./results/mGPfusion/") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pdb_id is None:
        raise RuntimeError("No PDB ID provided for parsing results.")
    h_file = load_pickled_file_from_dir(pdb_id, directory)
    if not h_file:
        return None, None, None, None
    mu = np.array([elem.get("mu") for elem in h_file.get("regression")], dtype=object)
    y_exp = np.array([elem.get("y_exp") for elem in h_file.get("regression")], dtype=object)
    cov = np.array([elem.get("cov") for elem in h_file.get("regression")], dtype=object)
    lmls = np.array([elem.get("lml") for elem in h_file.get("regression")], dtype=object)
    return mu, y_exp, cov, lmls

def parse_mutations(pdb_id, directory="./results/mGPfusion") -> np.ndarray:
    h_file = load_pickled_file_from_dir(pdb_id, directory)
    if h_file is None or h_file.get("mutations") is None:
        return None
    return np.concatenate(h_file.get("mutations"))

def parse_regression_metrics(pdb_id, directory="./results/mGPfusion/") -> Tuple[float, float]:
    h_file = load_pickled_file_from_dir(pdb_id, directory)
    if not h_file:
        return 0., 0.
    rho = h_file.get("rho")
    rmse = h_file.get("rmse")
    return rho, rmse

def plot_hyperparameters(proteins: list, save_fig="./fig/", dir="./results/hyper/"):
    """
    plots weight hyperparameters aquired from neg_ll optimization over all data
    """
    filename = os.path.join(save_fig, "hyper_table.png")
    kernels = KernelLoader()
    ws = []
    results_p = []
    for p in proteins:
        w = parse_hyperparameter_weights(pdb_id=p, directory=dir)
        if w is None:
            continue
        ws.append(w)
        results_p.append(p)
    ws = np.array(ws).T
    assert ws.shape[0] == len(kernels.sub_matrices_ids)
    assert ws.shape[1] == len(results_p)
    df = pd.DataFrame(data=ws, columns=results_p, index=kernels.sub_matrices_ids)
    descriptions = [m_info[0] for _, _, m_info in loadmat("./data/subMats.mat").get('subMats')] 
    #df["Description"] = descriptions
    df["Description"] = matrix_legend
    plt.rcParams.update({'figure.autolayout': True})
    fig, (ax, cbar_ax) = plt.subplots(2, figsize=(5, 20), gridspec_kw={"height_ratios": (.9, .05), "hspace": .3})
    im = sns.heatmap(ws, ax=ax, linewidths=0.5, cmap=sns.cm.rocket_r,# vmax=0.025, 
            cbar_ax=cbar_ax, cbar_kws={"orientation":"horizontal"})
    ax.set_xticklabels(results_p)
    ax.set_yticklabels(df.Description)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, rotation_mode="anchor", fontsize=10)
    plt.title("Kernel Weights Table")
    plt.savefig(filename, bbox_inches = "tight")
    plt.show()


def plot_sigmas(proteins: list, save_fig="./fig", dir="./results/hyper/", suffix=""):
    """
    plots trainable sigma parameters
    """
    filename = os.path.join(save_fig, f"hyper_sigmas.png")
    param_list = []
    result_p = []
    for p in proteins:
        p1, p2 = parse_hyperparameter_parameters(pdb_id=p, directory=dir)
        if p1 is not None and p2 is not None:
            param_list.append(np.array([p1, p2]))
            result_p.append(p)
    parameters = np.array(param_list)
    s_S = parameters[:, 0]
    s_E = parameters[:, 1]
    trainable_params = np.hstack([s_S, s_E]).T
    fig, (ax, cbar_ax) = plt.subplots(2, figsize=(5, 5))
    cbar_ax.set_aspect(0.05)
    im = sns.heatmap(trainable_params, ax=ax, annot=True, cmap=sns.cm.rocket_r,
            cbar_ax=cbar_ax, cbar_kws={"orientation":"horizontal"})
    ax.set_xticklabels(result_p)
    ax.set_yticklabels(["σ_S", "σ_E"])
    plt.title(f"Optimized noise parameters {suffix}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_mean_over_weights(proteins: list, save_fig="./fig", dir="./results/", suffix=""):
    plt.rcParams.update({'figure.autolayout': True})
    filename = os.path.join(save_fig, f"weights_mean_table_{suffix}.png")
    kernels = KernelLoader()
    mean_ws = []
    result_p = []
    for p in proteins:
        ws = parse_mean_weights_results(pdb_id=p, directory=dir)
        if ws is not None:
            mean_ws.append(ws)
            result_p.append(p)
    mean_ws = np.array(mean_ws)
    mean_ws = mean_ws[:, :, 0].T
    assert mean_ws.shape[0] == len(kernels.sub_matrices_ids)
    assert mean_ws.shape[1] == len(result_p)
    df = pd.DataFrame(data=mean_ws, columns=result_p, index=kernels.sub_matrices_ids)
    descriptions = [m_info[0] for _, _, m_info in loadmat("./data/subMats.mat").get('subMats')] 
    #df["Description"] = descriptions
    df["Description"] = matrix_legend
    fig, (ax, cbar_ax) = plt.subplots(2, figsize=(5, 20), gridspec_kw={"height_ratios": (.9, .05), "hspace": .3})
    im = sns.heatmap(mean_ws, ax=ax, linewidths=0.5, cmap=sns.cm.rocket_r, #vmax=0.025, 
            cbar_ax=cbar_ax, cbar_kws={"orientation":"horizontal"})
    ax.set_xticklabels(result_p)
    ax.set_yticklabels(df.Description)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, rotation_mode="anchor", fontsize=10)
    plt.title(f"Kernel weights μ \n after pos.-lvl CV {suffix}")
    plt.savefig(filename, bbox_inches = "tight")
    plt.show()


def generate_results_table(proteins, cvs=["all-pos.lvl.", "pos.lvl.", "mut.lvl."], 
    method=["mGPfusion", "2σ mGPfusion"], measures=["ρ", "rmse"], dir="./results/") -> None:
    """
    extract rho rmse from result directory structure
    """
    idx = pd.MultiIndex.from_product([proteins, method], 
        names=["Protein", "Method"])
    cols = pd.MultiIndex.from_product([measures, cvs], names=["measure", "CV"])
    results_df = pd.DataFrame(data=np.zeros([len(idx), len(cols)]), index=idx, columns=cols)
    for p in proteins:
        # all mutations pos-CV
        rho, rmse = parse_regression_metrics(p, directory="./results/mGPfusion/all_pos_cv_error/")
        results_df.loc[(p, "2σ mGPfusion"), ("ρ", "all-pos.lvl.")] = rho
        results_df.loc[(p, "2σ mGPfusion"), ("rmse", "all-pos.lvl.")] = rmse
        rho, rmse = parse_regression_metrics(p, directory="./results/mGPfusion/all_pos_cv/")
        results_df.loc[(p, "mGPfusion"), ("ρ", "all-pos.lvl.")] = rho
        results_df.loc[(p, "mGPfusion"), ("rmse", "all-pos.lvl.")] = rmse
        # experimental mutations pos-CV
        rho, rmse = parse_regression_metrics(p, directory="./results/mGPfusion/pos_cv/")
        results_df.loc[(p, "mGPfusion"), ("ρ", "pos.lvl.")] = rho
        results_df.loc[(p, "mGPfusion"), ("rmse", "pos.lvl.")] = rmse
        rho, rmse = parse_regression_metrics(p, directory="./results/mGPfusion/pos_cv_ref_error/")
        results_df.loc[(p, "2σ mGPfusion"), ("ρ", "pos.lvl.")] = rho
        results_df.loc[(p, "2σ mGPfusion"), ("rmse", "pos.lvl.")] = rmse
        # mut lvl CV
        rho, rmse = parse_regression_metrics(p, directory="./results/mGPfusion/mut_cv")
        results_df.loc[(p, "mGPfusion"), ("ρ", "mut.lvl.")] = rho
        results_df.loc[(p, "mGPfusion"), ("rmse", "mut.lvl.")] = rmse
        rho, rmse = parse_regression_metrics(p, directory="./results/mGPfusion/mut_cv_ref_error")
        results_df.loc[(p, "2σ mGPfusion"), ("ρ", "mut.lvl.")] = rho
        results_df.loc[(p, "2σ mGPfusion"), ("rmse", "mut.lvl.")] = rho
        # # mGP
        # mgp_rho, mgp_rmse = parse_regression_metrics(p, directory="./results/mGP/pos_cv/")
        # results_df.loc[(p, "mGP"), ("ρ", "pos.lvl.")] = mgp_rho
        # results_df.loc[(p, "mGP"), ("rmse", "pos.lvl.")] = mgp_rmse
    return results_df

def get_all_predictions_and_ys(pdbs, dir):
    all_predictions = []
    all_exp_y = []
    for p in pdbs:
        mu, ys, _, _ = parse_regression_results(p, directory=dir)
        if mu is None:
            continue
        all_predictions.append(mu)
        all_exp_y.append(ys)
    all_predictions = np.concatenate([np.atleast_1d(elem) for elem in all_predictions])
    all_predictions = np.concatenate([np.atleast_1d(e) for e in all_predictions])
    all_exp_y = np.concatenate([np.atleast_1d(elem) for elem in all_exp_y])
    all_exp_y = np.concatenate(all_exp_y)
    return all_predictions, all_exp_y


def generate_total_results_table(proteins, dir="./results/", measures=["ρ", "rmse"], 
        cvs=["all-pos.lvl.", "pos.lvl.", "mut.lvl."], methods=["mGPfusion", "2σ mGPfusion"]):
    cols = pd.MultiIndex.from_product([measures, cvs], names=["measure", "CV"])
    results_df = pd.DataFrame(data=np.zeros([len(methods), len(cols)]), index=methods, columns=cols)
    all_predictions, all_exp_y = get_all_predictions_and_ys(pdbs=proteins, 
        dir="./results/mGPfusion/all_pos_cv/")
    rho = spearmanr(all_predictions, all_exp_y)[0]
    rmse = mean_squared_error(all_exp_y, all_predictions)
    results_df.loc["mGPfusion", ("ρ", "all-pos.lvl.")] = rho
    results_df.loc["mGPfusion", ("rmse", "all-pos.lvl.")] = rmse
    all_predictions, all_exp_y = get_all_predictions_and_ys(pdbs=proteins, 
        dir="./results/mGPfusion/all_pos_cv_error/")
    rho = spearmanr(all_predictions, all_exp_y)[0]
    rmse = mean_squared_error(all_exp_y, all_predictions)
    results_df.loc["2σ mGPfusion", ("ρ", "all-pos.lvl.")] = rho
    results_df.loc["2σ mGPfusion", ("rmse", "all-pos.lvl.")] = rmse
    all_predictions, all_exp_y = get_all_predictions_and_ys(pdbs=proteins, 
        dir="./results/mGPfusion/pos_cv/")
    rho = spearmanr(all_predictions, all_exp_y)[0]
    rmse = mean_squared_error(all_exp_y, all_predictions)
    results_df.loc["mGPfusion", ("ρ", "pos.lvl.")] = rho
    results_df.loc["mGPfusion", ("rmse", "pos.lvl.")] = rmse
    all_predictions, all_exp_y = get_all_predictions_and_ys(pdbs=proteins, 
        dir="./results/mGPfusion/pos_cv_ref_error/")
    rho = spearmanr(all_predictions, all_exp_y)[0]
    rmse = mean_squared_error(all_exp_y, all_predictions)
    results_df.loc["2σ mGPfusion", ("ρ", "pos.lvl.")] = rho
    results_df.loc["2σ mGPfusion", ("rmse", "pos.lvl.")] = rmse
    all_predictions, all_exp_y = get_all_predictions_and_ys(pdbs=proteins, 
        dir="./results/mGPfusion/mut_cv/")
    rho = spearmanr(all_predictions, all_exp_y)[0]
    rmse = mean_squared_error(all_exp_y, all_predictions)
    results_df.loc["mGPfusion", ("ρ", "mut.lvl.")] = rho
    results_df.loc["mGPfusion", ("rmse", "mut.lvl.")] = rmse
    all_predictions, all_exp_y = get_all_predictions_and_ys(pdbs=proteins, 
        dir="./results/mGPfusion/mut_cv_ref_error/")
    rho = spearmanr(all_predictions, all_exp_y)[0]
    rmse = mean_squared_error(all_exp_y, all_predictions)
    results_df.loc["2σ mGPfusion", ("ρ", "mut.lvl.")] = rho
    results_df.loc["2σ mGPfusion", ("rmse", "mut.lvl.")] = rmse
    return results_df

def plot_pos_lvl_gpr_individual(proteins:list, results_dir="./results/mGPfusion", 
    save_fig="./fig/", suffix="", title="", y_n=3, x_n=4) -> None:
    assert x_n*y_n >= len(proteins)
    filename = os.path.join(save_fig, f"gpr_pos_lvl_individual_{suffix}.png")
    fig, ax = plt.subplots(y_n, x_n, figsize=(15,15))
    index = [(i,j) for i in range(y_n) for j in range(x_n)]
    for (i,j), p in zip(index, proteins):
        ax[i,j].axline((-4, -4), (4, 4), color="grey", linestyle="--")
        ax[i,j].set_xlim((-11,6))
        ax[i,j].set_ylim((-11,6))
        ax[i,j].grid(True)
        mu, y_test, _, _ = parse_regression_results(p, directory=results_dir)
        mutations = parse_mutations(p, results_dir)
        if mu is None or mutations is None:
            print(f"{p} cannot be found in {results_dir}")
            continue
        f_μ = np.concatenate([np.atleast_1d(elem) for elem in mu])
        y_test = np.concatenate([elem for sub in y_test for elem in sub])
        mapped_color = [colormap[mut-1] for mut in mutations]
        ax[i, j].scatter(y_test, f_μ, s=100., color=mapped_color, edgecolors="darkgrey")
        ax[i, j].set_title(f"{p}", fontsize=12)
    # TODO delete axes from a range of diff values between proteins and provided last axis length
    fig.delaxes(ax[2][3])
    fig.delaxes(ax[2][2])
    fig.delaxes(ax[2][1])
    fig.legend(handles=mutation_legend_handle, loc="lower right", title="Number of mutations")
    for i in range(y_n):
        ax[i,0].set_ylabel("predicted ΔΔG", fontsize=12)
    for i in range(x_n):
        ax[y_n-1,i].set_xlabel("experimental ΔΔG", fontsize=12)
    plt.suptitle(f"GP Regression (position lvl CV) {suffix}")
    plt.savefig(filename)
    plt.show()

def plot_pos_lvl_gpr_total(proteins:list, results_dir="./results/mGPfusion", 
    save_fig="./fig/", suffix="", title="") -> None:
    filename = os.path.join(save_fig, f"gpr_pos_lvl_total_{suffix}.png")
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.axline((-4, -4), (4,4), color="grey", linestyle="--")
    ax.grid(True)
    predictions = []
    ys = []
    mutations_all = []
    for p in proteins:
        mu, y_test, _, _ = parse_regression_results(p, directory=results_dir)
        mutations = parse_mutations(p, results_dir)
        if mu is None:
            print(f"{p} cannot be found in {results_dir}")
            continue
        f_μ = np.concatenate([np.atleast_1d(elem) for elem in mu])
        y_test = np.concatenate([elem for sub in y_test for elem in sub])
        predictions.append(f_μ)
        ys.append(y_test)
        mutations_all.append(mutations)
    predictions = np.concatenate(predictions)
    ys = np.concatenate(ys)
    mutations = np.concatenate(mutations_all)
    # add vertical bars connecting mutations
    stacked_measures = np.vstack([predictions, ys, mutations])
    stacked_measures = stacked_measures[:, mutations!=1] # deselect single mutations
    for x in np.unique(stacked_measures[1,:]):
        # extract equal measurements
        equal_idx = np.where(stacked_measures[1, :] == x)[0]
        # get min and max values for line
        y_min = np.min(stacked_measures[0, equal_idx])
        y_max = np.max(stacked_measures[0, equal_idx])
        mut = stacked_measures[2, equal_idx][0]
        ax.plot([x, x], [y_min, y_max], color=colormap[int(mut-1)], alpha=0.25, linewidth=8)
    mapped_color = [colormap[mut-1] for mut in mutations]
    ax.scatter(ys, predictions, s=100., color=mapped_color, edgecolors="darkgrey")
    fig.legend(handles=mutation_legend_handle, loc="lower right", title="Number of mutations")
    ax.set_xlabel("experimental ΔΔG", fontsize=18)
    ax.set_ylabel("predicted ΔΔG", fontsize=18)
    ax.legend()
    plt.title(f"GP Regression Results (position lvl CV) {suffix}")
    if title:
        plt.title(title)
    plt.savefig(filename)
    plt.show()
    

def plot_mut_lvl_gpr_individual(proteins:list, results_dir="./results/mGPfusion", save_fig="./fig/", suffix="", 
    y_n=3, x_n=4, uncertainties=False) -> None:
    filename = os.path.join(save_fig, f"gpr_mut_lvl_individual_{suffix}.png")
    fig, ax = plt.subplots(y_n, x_n, figsize=(15,15))
    index = [(i,j) for i in range(y_n) for j in range(x_n)]
    for (i,j), p in zip(index, proteins):
        ax[i,j].set_xlim((-11,6))
        ax[i,j].set_ylim((-11,6))
        ax[i,j].axline((-4, -4), (4,4), color="grey", linestyle="--")
        mu, y_test, cov, _ = parse_regression_results(pdb_id=p, directory=results_dir)
        mutations = parse_mutations(p, results_dir)
        if mu is None:
            continue
        f_μ = np.concatenate([np.atleast_1d(elem) for elem in mu])
        color = "black"
        y_test = np.concatenate([elem for sub in y_test for elem in sub])
        if mutations is not None:
            color = [colormap[mut-1] for mut in mutations]
        ax[i,j].scatter(y_test, f_μ, s=100., color=color,
         edgecolors="darkgrey")
        ax[i,j].set_title(f"{p}", fontsize=12)
        if not uncertainties:
            continue # if uncertainties run the loop below
        for idx, (μ, var, y) in enumerate(zip(f_μ, cov, y_test)):
            if idx == 2:
                break
            xx = np.arange(-5, 5, 0.1)
            f = norm.pdf(xx, μ, var)
            x_vals = np.array(y+f)
            ax[i,j].plot(x_vals, xx, "k--", alpha=0.1)
    fig.delaxes(ax[2][3])
    fig.delaxes(ax[2][2])
    fig.delaxes(ax[2][1])
    #fig.legend(handles=mutation_legend_handle, loc="lower right", title="Number of mutations")
    for i in range(y_n):
        ax[i,0].set_ylabel("predicted ΔΔG", fontsize=12)
    for i in range(x_n):
        ax[y_n-1,i].set_xlabel("experimental ΔΔG", fontsize=12)
    plt.suptitle(f"GP Regression (mutation lvl CV) {suffix}")
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_mut_lvl_gpr_total(proteins:list, results_dir="./results/mGPfusion/", save_fig="./fig/", suffix="", uncertainties=False) -> None:
    filename = os.path.join(save_fig, f"gpr_mut_lvl_total_{suffix}.png")
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.axline((-4, -4), (4,4), color="grey", linestyle="--")
    for p in proteins:
        mu, y_test, cov, _ = parse_regression_results(pdb_id=p, directory=results_dir)
        mutations = parse_mutations(p, results_dir)
        if mu is None:
            continue
        f_μ = np.concatenate([np.atleast_1d(elem) for elem in mu])
        y_test = np.concatenate([elem for sub in y_test for elem in sub])
        mapped_colors = "black"
        if mutations is not None:
            mapped_colors = [colormap[mut-1] for mut in mutations]
        ax.scatter(y_test, f_μ, s=100., color=mapped_colors,
            edgecolors="darkgrey")
        if not uncertainties:
            continue # if uncertainties run the loop below
        for idx, (μ, var, y) in enumerate(zip(f_μ, cov, y_test)):
            if idx == 2:
                break
            xx = np.arange(-5, 5, 0.1)
            f = norm.pdf(xx, μ, np.sqrt(var))
            x_vals = np.array(y+f)
            ax.plot(x_vals, xx, "k--", alpha=0.1)
   # fig.legend(handles=mutation_legend_handle, loc="lower right", title="Number of mutations")
    ax.set_ylim((-3, 3))
    ax.set_xlim((-3, 3))
    ax.set_xlabel("experimental ΔΔG", fontsize=18)
    ax.set_ylabel("predicted ΔΔG", fontsize=18)
    plt.suptitle(f"GP Regression (mutation lvl CV) {suffix}")
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

def plot_covariance_matrices(pcol, mats) -> None:
    
    labels = ["".join([m for m in mut]) for mut in pcol.mutation_ids[:10]]
    for mat in mats:
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        sns.heatmap(mat[:10, :10], ax=ax)
        ax.set_xticks(np.arange(0, 10))
        ax.set_xticklabels(labels, rotation=0, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(0, 10))
        ax.set_yticklabels(labels, rotation=0, rotation_mode="anchor", fontsize=10)
    plt.show()

def plot_mWDK(pcol, mWDK) -> None:
    labels = ["".join([m for m in mut]) for mut in pcol.mutation_ids[:10]]
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    sns.heatmap(mWDK[:10, :10], ax=ax)
    ax.set_xticks(np.arange(0, 10))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(0, 10))
    ax.set_yticklabels(labels, rotation=0, rotation_mode="anchor", fontsize=10)
    plt.savefig("./fig/mWDK_matrix.png")
    plt.show()
