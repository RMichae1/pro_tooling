from contact_mapper import ContactMapper
from graphkernel import VaeKernel
from vae import VAE
import numpy as np
import torch
import torch.nn.functional as F
import os
import argparse
import mlflow
from parse_data import parse_BLAT, parse_UBQ, parse_PGA, parse_TLL
from visualization import plot_SVAE_matrix, plot_VAE_kernel_values, plot_eigenvalues

DATA_DICT = {"pga": "1pga",
             #"ubq": "1ubq",
             #"blat": "1fqg",
             }

VAE_TYPES = list(DATA_DICT.keys())
PDB_FILES = list(DATA_DICT.values())
VAE_PARAMETERS = {"LATENT_DIM": 55, "ENCODER_DIM": [1700],
                  "DECODER_DIM": [1200], "EPOCHS": 200, "CUDA": False,
                  "DROPOUT": 0.065}


def run_and_plot_S_matrix(vae_type, vae, sequences, run_suffix=""):
    """
    Compute S Matrix from 1000 sequences
    """
    s_k = VaeKernel(vae)
    # S_mat_fam = s_k.compute_S_matrix(sequences[:1000, :], normalize=False)
    # plot_SVAE_matrix(S_mat_fam, name_suffix=vae_type.upper(), 
    #             run_suffix=run_suffix+" not normalized")
    S_mat_fam_norm = s_k.compute_S_matrix(sequences[:1000, :], normalize=True)
    plot_SVAE_matrix(S_mat_fam_norm, name_suffix=vae_type.upper(), 
                run_suffix=run_suffix+" normalized")


def run_and_plot_kernel(vae_type, vae, family_seqs, test_seqs, adjacencies,
                        normalize_S, normalize_k, subset_n=None):
    family_seqs = family_seqs[:subset_n, :] if subset_n else family_seqs
    test_seqs = test_seqs[:subset_n, :] if subset_n else test_seqs

    v_k = VaeKernel(vae, normalize_S=normalize_S, eigen=True)
    s_vae_val_fam = v_k.k(family_seqs[:1000, :], adjacencies=adjacencies, normalize_k=normalize_k)
    s_fam_eigen = v_k.eigen_values
    v_k.eigen_values = []
    fam_eigen_vec_real = np.stack([eig.real for eig in s_fam_eigen]).flatten()
    fam_eigen_vec_imag = np.stack([eig.imag for eig in s_fam_eigen]).flatten()
    s_vae_val_test = v_k.k(test_seqs[:1000, :], adjacencies=adjacencies, normalize_k=normalize_k)
    s_test_eigen = v_k.eigen_values
    v_k.eigen_values = []
    test_eigen_vec_real = np.stack([eig.real for eig in s_test_eigen]).flatten()
    test_eigen_vec_imag = np.stack([eig.imag for eig in s_test_eigen]).flatten()

    plotname = vae_type.upper() + normalize_S*' S-normalized' + normalize_k*' k-normalized'
    plot_VAE_kernel_values(s_vae_val_fam, s_vae_val_test, name_suffix=plotname)
    plot_eigenvalues(fam_eigen_vec_real, fam_eigen_vec_imag,
                     name_suffix=f"{plotname} MSA ")
    plot_eigenvalues(test_eigen_vec_real, test_eigen_vec_imag,
                     name_suffix=f"{plotname} SSL")


@torch.no_grad()
def experiment_routine(vae_type, family_seqs, test_seqs):
    num_classes = np.unique(family_seqs).shape[0] + 1
    WT = F.one_hot(torch.tensor(family_seqs[0], dtype=torch.int64), num_classes=num_classes).flatten().float()
    # init VAE
    input_dim = WT.shape[0]
    model_FILENAME = f"./models/VAE_t{vae_type}_z{VAE_PARAMETERS['LATENT_DIM']}_h{VAE_PARAMETERS['ENCODER_DIM'] + VAE_PARAMETERS['DECODER_DIM']}_e{VAE_PARAMETERS['EPOCHS']}_d{VAE_PARAMETERS['DROPOUT']}_wTrue.pt"
    vae = VAE(z_dim=VAE_PARAMETERS["LATENT_DIM"],
              encoder_dim=VAE_PARAMETERS["ENCODER_DIM"],
              decoder_dim=VAE_PARAMETERS["DECODER_DIM"],
              input_dims=input_dim,
              use_cuda=VAE_PARAMETERS["CUDA"],
              wt=WT, dropout=VAE_PARAMETERS["DROPOUT"],
              num_categories=num_classes)
    if os.path.exists(model_FILENAME):
        vae.load_state_dict(torch.load(model_FILENAME))
    else:
        raise FileNotFoundError("Model does not exist!")

    vae.eval()
    pdb_name = DATA_DICT.get(vae_type)
    # DERIVE S MATRIX
    # run_and_plot_S_matrix(vae_type=vae_type, vae=vae, sequences=family_seqs, run_suffix="MSA ")
    # run_and_plot_S_matrix(vae_type=vae_type, vae=vae, sequences=test_seqs, run_suffix="SSL ")
    # TEST KERNEL
    contact_map = ContactMapper(pdb_file=f"./pdb/{pdb_name}.pdb", tri_dist=True)
    ref_adj = contact_map.adjacency
    # JOINT LIKELIHOOD P_X_NOT_I 
    # run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs,
    #                     test_seqs=test_seqs, adjacencies=ref_adj,
    #                     normalize_S=False, normalize_k=False, marginal_not_i=False, subset_n=100)
    # run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs,
    #                     test_seqs=test_seqs, adjacencies=ref_adj,
    #                     normalize_S=True, normalize_k=False, marginal_not_i=False, subset_n=100)
    # run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs,
    #                     test_seqs=test_seqs, adjacencies=ref_adj,
    #                     normalize_S=False, normalize_k=True, marginal_not_i=False)
    # run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs,
    #                     test_seqs=test_seqs, adjacencies=ref_adj,
    #                     normalize_S=True, normalize_k=True, marginal_not_i=False, subset_n=100)
    # select subset for plotting DIFFs
    # run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs,
    #                     test_seqs=test_seqs, adjacencies=ref_adj,
    #                     normalize_S=True, normalize_k=False, subset_n=20)
    run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs,
                        test_seqs=test_seqs, adjacencies=ref_adj,
                        normalize_S=True, normalize_k=True, 
                        subset_n=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel Module - train and run VAE.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbosity boolean.")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed for reproducability.")
    parser.add_argument("--experiment", type=str, help="experiment str as ID for tracking.")
    parser.add_argument("--vae_type", nargs="+", type=str, default=VAE_TYPES,
                        help="List of identifiers of VAEs for which to run kernel experiments.")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for v in args.vae_type:
        if v == "blat":
            family_seqs, test_seqs, _ = parse_BLAT()
        elif v == "pga":
            family_seqs, test_seqs, _ = parse_PGA()
        elif v == "ubq":
            family_seqs, test_seqs, _ = parse_UBQ()
        else:
            raise NotImplementedError(
                "Specified type not implemented. Please pick a VAE from the list of options. See help -h.")
        experiment_routine(vae_type=v, family_seqs=family_seqs, test_seqs=test_seqs)
