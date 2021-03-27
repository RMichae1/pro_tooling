import pytest
import pickle
import numpy as np
import torch
import pyro
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam
from utility import one_hot_encoding
from reference_vae.diku_thesis.torch_vae import VAE as ReferenceVAE
from vae import VAE, train, evaluate

def test_vae():
    """
    Tests the VAE Pyro implementation against the reference PyTorch implementation.
    Dataset is extraction from BLAT_ECOLX data - 100 Sequences.
    """
    pass

###
# SETUP AND INITIALIZE
###
np.random.seed(123)
torch.manual_seed(123)

with open("../data/blat/BLAT_data_df.pkl", "rb") as infile:
    blat_df = pickle.load(infile)

# stored values without assay entries are BLAT TEM1 ECOLX
blat_df = blat_df[blat_df.assay.isna()]
# cast sequence labels to int
all_seqs = []
for i, seq in enumerate(blat_df.seqs):
    all_seqs.append([int(elem) for elem in seq])
    if i == 99:
        break
all_seqs = np.array(all_seqs)

x, y = all_seqs.shape
categories = np.unique(all_seqs).shape[0]
one_seq = all_seqs.reshape(x*y, )
one_hot_sequence = one_hot_encoding(one_seq).reshape(x, y, categories)

# load and encode data set
seq_dataset = torch.utils.data.TensorDataset(torch.tensor(one_hot_sequence,
                                                              dtype=torch.float))
test_size = int(0.1 * x)
seq_train, seq_test = torch.utils.data.random_split(seq_dataset, [ x - test_size, test_size])
train_loader = torch.utils.data.DataLoader(seq_train)
test_loader = torch.utils.data.DataLoader(seq_test)

wt = torch.tensor(one_hot_sequence[0], dtype=torch.float)
seq_1 = torch.tensor(one_hot_sequence[1], dtype=torch.float)

EPOCHS = 200
VALIDATION = 10
LEARN_RATE = 0.0125
Z_DIM = 20
HIDDEN_DIM = 500
INPUT_DIM = int(one_hot_sequence.shape[1] * one_hot_sequence.shape[2])

vae = VAE(Z_DIM, HIDDEN_DIM, INPUT_DIM)
ref_vae = ReferenceVAE(layer_sizes=INPUT_DIM, num_tokens=categories)
optimizer =  Adam({"lr": LEARN_RATE})
svi = SVI(vae.model, vae.guide, optimizer, loss=JitTrace_ELBO())

### TRAIN MODELS

def test_training():
    train_loss = []
    test_loss = []
    for epoch in range(EPOCHS):
        total_epoch_loss_train = train(svi, train_loader)
        train_loss.append(total_epoch_loss_train)
        print(f"[epoch {epoch}] avrg. train loss: {total_epoch_loss_train}")

        if epoch % VALIDATION == 0:
            total_epoch_loss_test = evaluate(svi, test_loader)
            test_loss.append(total_epoch_loss_test)
            print(f"[epoch {epoch}] avrg. test loss: {total_epoch_loss_test}")
