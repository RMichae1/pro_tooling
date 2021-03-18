from vae import VAE
import pickle
import os
from utility import aa2index, one_hot_encoding
from vae import train, evaluate
import pyro
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam, ClippedAdam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import torch
from tqdm import tqdm
import logging
import warnings
import argparse
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' #TODO figure out what caused OMP Error #15


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    pyro.clear_param_store()
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="VAE Module - train and run VAE.")
    parser.add_argument("-lr", "--learn_rate", type=float, default=5e-4, help="learning rate for optimizer")
    parser.add_argument("--cuda", action="store_true", help="Boolean flag to use cuda.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbosity boolean.")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed for reproducability.")
    parser.add_argument("-e", "--epochs", type=int, default=500, help="Training epochs.")
    parser.add_argument("--latent_dim", type=int, default=20, help="Dimensionality of hidden latent random variable.")
    parser.add_argument("-s", "--save", type=str, help="Destination for model output.")
    parser.add_argument("--hidden_dim", type=int, default=500, help="Hidden dimension for VAE internals.")
    parser.add_argument("--test_split", type=float, default=0.1, choices=np.arange(0, 1, 0.001), help="Fraction of test data from total data-set.")
    parser.add_argument("--validate", type=int, default=10, help="Frequency of validation step.")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Int size of batches.")
    parser.add_argument("--experiment", type=str, help="experiment str as ID for tracking.")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0., help="Adam Optimizer weight decay.")
    parser.add_argument("-d", "--dropout", type=float, default=None, help="Add Dropout layer with dropout probability.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open("./data/blat/BLAT_data_df.pkl", "rb") as infile:
        blat_df = pickle.load(infile)
    # stored values without assay entries are BLAT TEM1 ECOLX family data
    family_df = blat_df[blat_df.assay.isna()]
    test_blat_df = blat_df[~blat_df.assay.isna()]
    # cast sequence labels to int
    family_seqs = np.array([[int(elem) for elem in seq] for seq in family_df.seqs])
    test_blat_seqs = np.array([[int(elem) for elem in seq] for seq in test_blat_df.seqs])
    test_y = np.array(test_blat_df.assay)

    n, length = family_seqs.shape
    test_n = test_blat_seqs.shape[0]
    categories = np.unique(family_seqs).shape[0]
    one_seq = family_seqs.reshape(n*length,)
    test_one_seq = test_blat_seqs.reshape(test_n*length,)
    # TODO refactor the one-hot encoding - this code is non-pythonic!
    one_hot_sequence_all = one_hot_encoding(np.concatenate((one_seq, test_one_seq), axis=None))
    one_hot_sequences = one_hot_sequence_all[:len(one_seq)].reshape(n, length, categories)
    one_hot_test_sequences = one_hot_sequence_all[len(one_seq):].reshape(test_n, length, categories)
    assert one_hot_sequences.shape[0] == len(family_df)
    assert one_hot_test_sequences.shape[0] == len(test_blat_df)

    # load and encode data set
    seq_dataset = torch.utils.data.TensorDataset(torch.tensor(one_hot_sequences,
                                                              dtype=torch.float))
    test_seq_dataset = torch.utils.data.TensorDataset(torch.tensor(one_hot_test_sequences, 
                                                                    dtype=torch.float))
    test_size = int(args.test_split * n)
    seq_train, seq_test = torch.utils.data.random_split(seq_dataset, [n - test_size, test_size])
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(seq_train, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(seq_test, batch_size=batch_size)

    WT = torch.tensor(one_hot_sequences[0], dtype=torch.float)

    # parameters
    param_dict = {"LEARNING_RATE": args.learn_rate,
            "USE_CUDA": args.cuda,
            "NUM_EPOCHS": args.epochs,
            "TEST_FREQ": args.validate,
            "LATENT_DIM": args.latent_dim,
            "HIDDEN_DIM": args.hidden_dim,
            "INPUT_DIM": int(one_hot_sequences.shape[1] * one_hot_sequences.shape[2]),
            "test_split": args.test_split, 
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout}

    # TODO VAE of different flavors - sparse, dropout, etc.
    experiment_name = args.experiment if args.experiment else f"VAE_Adam_z{args.latent_dim}"
    mlflow.set_experiment(experiment_name)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    model_FILENAME = f"./models/VAE_z{args.latent_dim}_h{args.hidden_dim}_e{args.epochs}_d{args.dropout}.pt"
    optimizer_FILENAME = f"./models/Adam_z{args.latent_dim}_h{args.hidden_dim}_e{args.epochs}_d{args.dropout}.pt"
    vae = VAE(z_dim=param_dict["LATENT_DIM"], hidden_dim=param_dict["HIDDEN_DIM"], 
                    input_dims=param_dict["INPUT_DIM"], use_cuda=args.cuda, wt=WT)
    optimizer = Adam({"lr": param_dict["LEARNING_RATE"], "weight_decay": param_dict["weight_decay"]})
    #optimizer = ClippedAdam({"lr": LEARNING_RATE})
    svi = SVI(vae.model, vae.guide, optimizer, loss=JitTrace_ELBO())
    
    if os.path.exists(model_FILENAME) and os.path.exists(optimizer_FILENAME):
        vae.load_state_dict(torch.load(model_FILENAME))
        optimizer.load(optimizer_FILENAME)
    else:
        mlflow.start_run()
        mlflow.log_params(param_dict)
        mlflow.set_tag("out", "Categorical")
        for epoch in tqdm(range(args.epochs)):
            total_epoch_loss_train = train(svi, train_loader, args.cuda)
            mlflow.log_metric(key="neg loss train", value=-total_epoch_loss_train, 
                                step=epoch)
            print(f"[epoch {epoch}] avrg. train loss: {total_epoch_loss_train}")

            if epoch % args.validate == 0:
                total_epoch_loss_test = evaluate(svi, test_loader, args.cuda)
                mlflow.log_metric(key="neg loss validate", value=-total_epoch_loss_test, 
                                    step=epoch)
                print(f"[epoch {epoch}] avrg. test loss: {total_epoch_loss_test}")
        
        
        torch.save(vae.state_dict(), model_FILENAME)
        optimizer.save(optimizer_FILENAME)
        mlflow.log_artifact(model_FILENAME)
        mlflow.log_artifact(optimizer_FILENAME)
        mlflow.end_run()
    
    wt_log_prob = vae.log_p(WT)[1].detach().numpy()
    wt_elbo = vae.log_p(WT)[0].detach().numpy()
    elbo_values = []
    kld_values = []
    log_likelihoods = []
    samples = []
    for s in test_seq_dataset:
        samples.append(vae.latent_sample(s[0], n=1).reshape(-1).detach().numpy())
        loss = vae.log_p(s[0])
        elbo_values.append(loss[0].detach().numpy())
        log_likelihoods.append(loss[1].detach().numpy())
        kld_values.append(loss[2].detach().numpy())
    # PLOT first two dimensions
    # samples = np.array(samples)
    # plt.scatter(samples[:, 0], samples[:, 1], c=log_likelihoods, alpha=0.25, s=1.5)
    # plt.title("VAE z=20 latent representation on 2D")
    # plt.show()
    # WT is first element
    delta_log_p = [(l-wt_log_prob) for l in log_likelihoods]

    fig, ax = plt.subplots(1, 1)
    sns.regplot(delta_log_p, test_y, ax=ax[0], color="grey", scatter_kws={"alpha": 0.25})
    ax[0].set_ylabel("measured growth (2500 ampicillin dose)")
    ax[0].set_xlabel("delta log likelihood")
    plt.suptitle("VAE loss to measured values \n (2500 ampicillin dose)")
    plt.show()
    print(spearmanr(delta_log_p, test_y))
    
    
    

        