import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_gpr_results():
    pass


def parse_hyperparameter_results(pdb_id, directory="./results/hyper/"):
    for f in os.listdir(directory):
        if not pdb_id in f:
            continue
        with open(f, "rb") as infile:
            h_file = pickle.load(infile)
        weights = h_file.get("weights")
    return weights


def plot_hyperparameters(weights: list, matrices: list, matrix_description: list, proteins: list):
    assert weights.shape[0] == len(matrices)
    assert weights.shape[1] == len(proteins)
    df = pd.DataFrame(data=weights, columns=proteins, index=matrices)
    df["Description"] = None 

    fig, ax = None, None


def results_table(rho, rmse, proteins, cv=["pos-lvl. ref", "pos-lvl"], 
    method=["mGPfusion", "mGP"]):
    """
    rho, rmse are nx2
    """
    assert rho.shape[1] == len(cv)
    assert rmse.shape[1] == len(cv)
    df = pd.DataFrame()
    for idx, p in enumerate(proteins):
        p_df = pd.DataFrame()
        p_df.index = p


def plot_gpr(self, f_μ, y_test, cov=None, save_fig="./fig/", pdb_id=None) -> None:
    filename = os.path.join(save_fig, f"gpr_{pdb_id}.png")
    _, ax = plt.subplots(1,1, figsize=(15,10))
    ax.axline((-4, -4), (4,4), color="grey", linestyle="--")
    f_μ = np.concatenate([np.atleast_1d(elem) for elem in f_μ])
    y_test = np.concatenate([elem for sub in y_test for elem in sub])
    plt.scatter(y_test, f_μ, color="indianred")
    if mutations:
        mutations = [np.repeat(mut, len(y)) for mut, y in zip(mutations, y_test)]
        sns.scatterplot(y_test, f_μ, hue=mutations, ax=ax)
    # TODO each mutation has n means and n covariances
    # add gaussians to plot
    if cov is not None:
        for idx, μ, var, y in enumerate(zip(f_μ, cov, y_test)):
            xx = np.arange(-5, 5, 0.1)
            f = norm.pdf(xx, μ, var)
            ax.plot(y+f, xx, "k-")
        # annotate mutations at test point
        # TODO write mutations while doing CV and query here
        ax.annotate(self.protein.mutation_ids[idx], xy=(y_test, μ), xycoords="data", xytext=(10,10), 
            textcoords='offset points')
    ax.set_xlabel("experimental ΔΔG")
    ax.set_ylabel("predicted ΔΔG")
    plt.title(f"GP Regression (position lvl) {self.id}")
    plt.legend()
    plt.savefig(filename)
    plt.show()


def plot_log_prob(self, lml, mutations) -> None:
    _, ax = plt.subplots(1, 1, figsize=(10,10))
    for idx, mut in enumerate(self.x_test):
        y_train = self.y.shape[0] - mutations
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
