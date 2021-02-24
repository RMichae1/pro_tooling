from vae import VAE
import pickle
from utility import one_hot_encoding
from vae import train, evaluate
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam, ClippedAdam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

if __name__ == "__main__":
    with open("./data/BLAT_data_df.pkl", "rb") as infile:
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

    seq_dataset = torch.utils.data.TensorDataset(torch.tensor(one_hot_sequence,
                                                              dtype=torch.float))
    test_size = int(0.1 * x)
    seq_train, seq_test = torch.utils.data.random_split(seq_dataset, [ x - test_size, test_size])
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(seq_train, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(seq_test, batch_size=batch_size)

    LEARNING_RATE = 5e-4
    USE_CUDA = False
    NUM_EPOCHS = 1000
    TEST_FREQ = 10
    INPUT_DIM = int(one_hot_sequence.shape[1] * one_hot_sequence.shape[2])

    vae = VAE(z_dim=50, hidden_dim=400, input_dims=INPUT_DIM, use_cuda=USE_CUDA)
    optimizer = Adam({"lr": LEARNING_RATE})
    #optimizer = ClippedAdam({"lr": LEARNING_RATE})
    svi = SVI(vae.model, vae.guide, optimizer, loss=JitTrace_ELBO())

    train_elbo = []
    test_elbo = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        total_epoch_loss_train = train(svi, train_loader, USE_CUDA)
        train_elbo.append(-total_epoch_loss_train)
        print(f"[epoch {epoch}] avrg. train loss: {total_epoch_loss_train}")

        if epoch % TEST_FREQ == 0:
            total_epoch_loss_test = evaluate(svi, test_loader, USE_CUDA)
            test_elbo.append(-total_epoch_loss_test)
            print(f"[epoch {epoch}] avrg. test loss: {total_epoch_loss_test}")

    plt.plot(train_elbo, label="train")
    plt.legend()
    plt.title("ELBO Training over iterations")
    plt.ylabel("neg. training loss [ELBO]")
    plt.xlabel("validation step")
    plt.show()

    plt.plot(test_elbo, label="test")
    plt.legend()
    plt.title("ELBO validation over validation-steps")
    plt.ylabel("neg. validation loss [ELBO]")
    plt.xlabel("validation steps")
    plt.xticks(np.arange(0, int(NUM_EPOCHS / TEST_FREQ) + .1, step=1))
    plt.show()