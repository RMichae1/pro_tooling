from contact_mapper import ContactMapper
from graphkernel import VaeKernel
from vae import VAE
import numpy as np
import torch
import torch.nn.functional as F
import os
import logging
import warnings
import argparse
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
import random
from utility import parse_BLAT, parse_UBQ, parse_PGA, parse_TLL
from reference_alphabet import seq2idx
from visualization import plot_SVAE_matrix, plot_VAE_kernel_values, plot_eigenvalues

# DATA_DICT = {"blat": "2lzm", "sp400": "", "ubq": "1ubq", 
#             "pga": "1pga", "hexo": ""}
DATA_DICT = {"blat": "2lzm", "ubq": "1ubq", 
            "pga": "1pga"}
VAE_TYPES = list(DATA_DICT.keys())
PDB_FILES = list(DATA_DICT.values())
VAE_PARAMETERS = {"LATENT_DIM": 55, "ENCODER_DIM": 1700,
                "DECODER_DIM": 1200, "EPOCHS": 200, "CUDA": False,
                "DROPOUT": 0.065}


def run_and_plot_S_matrix(vae_type, vae, sequences, marginal):
    s_k = VaeKernel(vae, marginal_not_i=marginal)
    S_mat_fam = s_k.compute_S_matrix(sequences, normalize=False)
    S_mat_fam_norm = s_k.compute_S_matrix(sequences, normalize=True)
    plot_SVAE_matrix(S_mat_fam, name_suffix=vae_type.upper())
    plot_SVAE_matrix(S_mat_fam_norm, name_suffix=vae_type.upper())


def run_and_plot_kernel(vae_type, vae, family_seqs, test_seqs, adjacencies, 
                normalize_S, normalize_k, marginal_not_i, subset_n=None):
    family_seqs = family_seqs[:subset_n, :] if subset_n else family_seqs
    test_seqs = test_seqs[:subset_n, :] if subset_n else test_seqs

    v_k = VaeKernel(vae, normalize_S=normalize_S, marginal_not_i=marginal_not_i,
                     eigen=True)
    s_vae_val_fam = v_k.k(family_seqs, adjacencies=adjacencies, 
                        normalize_k=normalize_k)
    s_fam_eigen = v_k.eigen_values
    v_k.eigen_values = []
    fam_eigen_vec_real = np.stack([eig.real for eig in s_fam_eigen]).flatten()
    fam_eigen_vec_imag = np.stack([eig.imag for eig in s_fam_eigen]).flatten()
    s_vae_val_test = v_k.k(test_seqs[:10, :], adjacencies=adjacencies, 
                        normalize_k=normalize_k)
    s_test_eigen = v_k.eigen_values
    v_k.eigen_values = []
    test_eigen_vec_real = np.stack([eig.real for eig in s_test_eigen]).flatten()
    test_eigen_vec_imag = np.stack([eig.imag for eig in s_test_eigen]).flatten()
    
    plot_VAE_kernel_values(s_vae_val_fam, s_vae_val_test)
    plot_eigenvalues(fam_eigen_vec_real, fam_eigen_vec_imag, 
                    name_suffix=f"{vae_type.upper()} MSA")
    plot_eigenvalues(test_eigen_vec_real, test_eigen_vec_imag,
                    name_suffix=f"{vae_type.upper()} SSL")


def experiment_routine(vae_type, family_seqs, test_seqs):
    num_classes = np.unique(family_seqs).shape[0] + 1
    WT = F.one_hot(torch.tensor(family_seqs[0], dtype=torch.int64),
                num_classes=num_classes).flatten().float()
    # init VAE
    input_dim = WT.shape[0]
    model_FILENAME = f"./models/VAE_t{vae_type}_z{VAE_PARAMETERS['LATENT_DIM']}_h[{VAE_PARAMETERS['ENCODER_DIM']}, {VAE_PARAMETERS['DECODER_DIM']}]_e200_d0.065_wTrue.pt"
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
    pdb_name = DATA_DICT.get(args.type)
    # DERIVE S MATRIX
    run_and_plot_S_matrix(vae_type=vae_type, vae=vae, sequences=family_seqs, 
                        marginal=False)
    run_and_plot_S_matrix(vae_type=vae_type, vae=vae, sequences=test_seqs, 
                        marginal=False)
    run_and_plot_S_matrix(vae_type=vae_type, vae=vae, sequences=family_seqs, 
                        marginal=True)
    run_and_plot_S_matrix(vae_type=vae_type, vae=vae, sequences=test_seqs, 
                        marginal=True)
    ## TEST KERNEL
    contact_map = ContactMapper(pdb_file=f"./pdb/{pdb_name}.pdb", tri_dist=True)
    ref_adj = contact_map.adjacency
    # JOINT LIKELIHOOD P_X_NOT_I 
    run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs, 
                        test_seqs=test_seqs, adjacencies=ref_adj, 
                        normalize_S=False, normalize_k=False, marginal_not_i=False)
    run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs, 
                        test_seqs=test_seqs, adjacencies=ref_adj, 
                        normalize_S=True, normalize_k=False, marginal_not_i=False)
    run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs, 
                        test_seqs=test_seqs, adjacencies=ref_adj, 
                        normalize_S=False, normalize_k=True, marginal_not_i=False)            
    run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs, 
                        test_seqs=test_seqs, adjacencies=ref_adj, 
                        normalize_S=True, normalize_k=True, marginal_not_i=False)
    # MARGINAL LIKELIHOOD P_X_NOT_I 
    run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs, 
                        test_seqs=test_seqs, adjacencies=ref_adj, 
                        normalize_S=False, normalize_k=False, marginal_not_i=True)
    run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs, 
                        test_seqs=test_seqs, adjacencies=ref_adj, 
                        normalize_S=True, normalize_k=False, marginal_not_i=True)
    run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs, 
                        test_seqs=test_seqs, adjacencies=ref_adj, 
                        normalize_S=False, normalize_k=True, marginal_not_i=True)            
    run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs, 
                        test_seqs=test_seqs, adjacencies=ref_adj, 
                        normalize_S=True, normalize_k=True, marginal_not_i=True)
    # select subset for plotting DIFFs
    run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs, 
                        test_seqs=test_seqs, adjacencies=ref_adj, 
                        normalize_S=False, normalize_k=False, subset_n=20)
    run_and_plot_kernel(vae_type=vae_type, vae=vae, family_seqs=family_seqs, 
                        test_seqs=test_seqs, adjacencies=ref_adj, 
                        normalize_S=True, normalize_k=True, subset_n=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel Module - train and run VAE.")
    parser.add_argument("-lr", "--learn_rate", type=float, default=0.000027, help="learning rate for optimizer")
    parser.add_argument("--cuda", action="store_true", help="Boolean flag to use cuda.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbosity boolean.")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed for reproducability.")
    parser.add_argument("--experiment", type=str, help="experiment str as ID for tracking.")
    parser.add_argument("--vae_type", nargs="+", type=str, default=VAE_TYPES,
                        help="List of identifiers of VAEs for which to run kernel experiments.")
    parser.add_argument("-t", "--type", choices=VAE_TYPES, default="ubq", help="Type ID of MSA used to create VAE.")
    parser.add_argument("-p", "--plot", action="store_false", help="Plot low-latent-representation outputs and feature correlation.")
    args = parser.parse_args()  # TODO change weighting to store_true
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.type == "blat":
        family_seqs, test_seqs, _ = parse_BLAT()
    elif args.type == "sp400":
        family_seqs, test_seqs, _ = parse_TLL()
    elif args.type == "pga":
        family_seqs, test_seqs, _ = parse_PGA()
    elif args.type == "ubq":
        family_seqs, test_seqs, _ = parse_UBQ()
    else:
        raise NotImplementedError(
            "Specified type not implemented. Please pick a VAE from the list of options. See help -h.")

    for v in args.vae_type:
        experiment_routine(vae_type=v, family_seq=family_seqs, test_seq=test_seqs)
