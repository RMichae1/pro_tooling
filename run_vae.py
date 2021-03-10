from vae import VAE
import pickle
import os
from utility import one_hot_encoding
from vae import train, evaluate
import pyro
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam, ClippedAdam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import logging
import warnings
import argparse
import mlflow

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
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="Training epochs.")
    parser.add_argument("--latent_dim", type=int, help="Dimensionality of hidden latent random variable.")
    parser.add_argument("-s", "--save", type=str, help="Destination for model output.")
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension for VAE internals.")
    parser.add_argument("--test_split", type=float, default=0.1, choices=np.arange(0, 1, 0.001), help="Fraction of test data from total data-set.")
    parser.add_argument("--validate", type=int, default=10, help="Frequency of validation step.")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Int size of batches.")
    parser.add_argument("--experiment", type=str, help="experiment str as ID for tracking.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open("./data/blat/BLAT_data_df.pkl", "rb") as infile:
        blat_df = pickle.load(infile)
    # stored values without assay entries are BLAT TEM1 ECOLX
    blat_df = blat_df[blat_df.assay.isna()]
    # cast sequence labels to int
    all_seqs = []
    for seq in blat_df.seqs:
        all_seqs.append([int(elem) for elem in seq])
    all_seqs = np.array(all_seqs)
    x, y = all_seqs.shape
    one_seq = all_seqs.reshape(x*y, )
    one_hot_sequence = one_hot_encoding(one_seq).reshape(x, y, 23)

    # load and encode data set
    seq_dataset = torch.utils.data.TensorDataset(torch.tensor(one_hot_sequence,
                                                              dtype=torch.float))
    test_size = int(args.test_split * x)
    seq_train, seq_test = torch.utils.data.random_split(seq_dataset, [ x - test_size, test_size])
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(seq_train, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(seq_test, batch_size=batch_size)

    wt = torch.tensor(one_hot_sequence[0], dtype=torch.float)
    seq_1 = torch.tensor(one_hot_sequence[1], dtype=torch.float)

    # parameters
    param_dict = {"LEARNING_RATE": args.learn_rate,
            "USE_CUDA": args.cuda,
            "NUM_EPOCHS": args.epochs,
            "TEST_FREQ": args.validate,
            "LATENT_DIM": args.latent_dim,
            "HIDDEN_DIM": args.hidden_dim,
            "INPUT_DIM": int(one_hot_sequence.shape[1] * one_hot_sequence.shape[2]),
            "test_split": args.test_split, 
            "batch_size": args.batch_size}

    # TODO VAE of different flavors - sparse, dropout, etc.
    experiment_name = args.experiment if args.experiment else f"VAE_Adam_z{args.latent_dim}"
    mlflow.set_experiment(experiment_name)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    model_FILENAME = f"./models/VAE_{args.latent_dim}_{args.hidden_dim}_{args.epochs}.pt"
    optimizer_FILENAME = f"./model/Adam_{args.latent_dim}_{args.hidden_dim}_{args.epochs}.pt"
    vae = VAE(z_dim=param_dict["LATENT_DIM"], hidden_dim=param_dict["HIDDEN_DIM"], 
                    input_dims=param_dict["INPUT_DIM"], use_cuda=args.cuda, wt=wt)
    optimizer = Adam({"lr": param_dict["LEARNING_RATE"]})
    #optimizer = ClippedAdam({"lr": LEARNING_RATE})
    svi = SVI(vae.model, vae.guide, optimizer, loss=JitTrace_ELBO())
    
    if os.path.exists(model_FILENAME):
        vae.load_state_dict(torch.load(model_FILENAME))
        svi.load(optimizer_FILENAME)
    else:
        mlflow.start_run()
        mlflow.log_params(param_dict)
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
        
    print(vae.log_p(wt))
    print(vae.log_p(seq_1))
        