from vae import VAE
from vae import train, evaluate
from utility import aa2index, one_hot_encoding
import pickle
import os
import reference_alphabet
import pyro
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam, ClippedAdam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from utility import compute_ρ
from utility import WeightedMSADataset, seq_collate
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import warnings
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import random
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from run_vae import parse_TLL

np.random.seed(42)
torch.manual_seed(42)
cuda = torch.device('cuda')


if __name__ == "__main__":
    family_seqs, test_seqs, test_y = parse_TLL()
    n, length = family_seqs.shape
    test_n = test_seqs.shape[0]
    num_classes = np.unique(family_seqs).shape[0]
    indices = list(range(n))
    random.shuffle(indices)
    test_size = int(0.1 * n)
    train_idx = indices[:(n - test_size)]
    test_idx = indices[(n - test_size):]
    # load and encode data set
    seq_train = WeightedMSADataset(family_seqs[train_idx], num_classes=num_classes)
    seq_test = WeightedMSADataset(family_seqs[test_idx], num_classes=num_classes)
    sampler = torch.utils.data.WeightedRandomSampler(seq_train.weights, 
                                                    num_samples=len(seq_train), replacement=True)
    test_seq_dataset = WeightedMSADataset(test_seqs, num_classes=num_classes)
    train_loader = torch.utils.data.DataLoader(seq_train, batch_size=128, 
                                            sampler=sampler, collate_fn=seq_collate)
    test_loader = torch.utils.data.DataLoader(seq_test, batch_size=128, 
                                            shuffle=True, collate_fn=seq_collate)
    WT = F.one_hot(torch.tensor(family_seqs[0], dtype=torch.int64), 
                    num_classes=num_classes).flatten().float()
    torch.autograd.set_detect_anomaly(True)

    def fit_model(svi, vae, train_loader=train_loader, test_loader=test_loader, epochs=250):
        vae.train()
        for epoch in tqdm(range(epochs)):
            total_epoch_loss_train = train(svi, train_loader, False)
            print(f"[epoch {epoch}] avrg. train loss: {total_epoch_loss_train}")

            if epoch % VALIDATE == 0:
                total_epoch_loss_test = evaluate(svi, test_loader, False)
                print(f"[epoch {epoch}] avrg. test loss: {total_epoch_loss_test}")
        return svi, vae

    def correlation(vae, test_y=test_y):
        vae.eval()
        wt_log_prob = vae.log_p(WT.cuda())[1].cpu().detach().numpy()
        log_likelihoods = []
        for s, _, _ in test_seq_dataset:
            loss = vae.log_p(s.flatten().cuda())
            log_likelihoods.append(loss[1].cpu().detach().numpy())
        delta_log_p = np.array([(l-wt_log_prob) for l in log_likelihoods], dtype=float)
        return spearmanr(delta_log_p, test_y)[0]

    search_space  = [Integer(900, 2000, name='encoder_dim'),
                    Integer(100, 2000, name="decoder_dim"),
                    Integer(2, 100, name='latent_dim'),
                    Real(0.00001, 0.5, "log-uniform", name="dropout"),
                    Real(10**-5, 10**-1, "log-uniform", name='learning_rate'),
                    Real(10**-5, 10**-1, "log-uniform", name='weight_decay')]
    @use_named_args(search_space)
    def objective(encoder_dim, decoder_dim, 
                latent_dim, dropout, learning_rate, weight_decay):
        pyro.clear_param_store()
        vae = VAE(z_dim=latent_dim, encoder_dim=[encoder_dim], decoder_dim=[decoder_dim],
                    input_dims=WT.shape[0], use_cuda=True, wt=WT, dropout=dropout, num_categories=num_classes)
        optimizer = Adam({"lr": learning_rate, "weight_decay": weight_decay})
        svi = SVI(vae.model, vae.guide, optimizer, loss=JitTrace_ELBO())
        try:
            svi, vae = fit_model(svi=svi, vae=vae)
        except:
            return 9000
        return -correlation(vae)

    # call optimization
    res_gp = gp_minimize(objective, search_space, n_calls=20, random_state=101)

    print(f"Best score={res_gp.fun}")
    print(f"""Best parameters:
        - encoder_dim={res_gp.x[0]},
        - decoder_dim1={res_gp.x[1]},
        - latent_dim={res_gp.x[2]}
        - dropout={res_gp.x[3]}
        - lr={res_gp.x[4]}
        - weight decay (l2) = {res_gp.x[5]}""")
        
    plot_convergence(res_gp)