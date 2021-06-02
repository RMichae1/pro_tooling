import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pyro.infer import JitTrace_ELBO, SVI
from pyro.optim import Adam
import numpy as np
from utility import WeightedMSADataset, seq_collate
from vae import VAE, train, evaluate
from parse_data import parse_UBQ
from kernel.linear_kernel import



def setup_dummy_data_train_test():
    N = 250  # number of sequences in MSA
    L = 10  # length of the sequence
    AA = 4  # number of amino acids
    BATCHSIZE = 128
    dummy_sequences = np.random.randint(0, AA, size=[N, L])
    indices = list(range(N))
    random.shuffle(indices)
    test_size = int(0.1 * N)  # 10% test split
    train_idx = indices[:(N - test_size)]
    test_idx = indices[(N - test_size):]
    seq_train = WeightedMSADataset(dummy_sequences[train_idx], num_classes=AA+1)
    seq_test = WeightedMSADataset(dummy_sequences[test_idx], num_classes=AA+1)
    train_loader = torch.utils.data.DataLoader(seq_train, batch_size=BATCHSIZE, shuffle=True, collate_fn=seq_collate)
    test_loader = torch.utils.data.DataLoader(seq_test, batch_size=BATCHSIZE, shuffle=True, collate_fn=seq_collate)
    return dummy_sequences, train_loader, test_loader


def setup_dummy_VAE():
    TRAIN_EPOCHS = 100
    VALIDATION = 10
    dummy_sequences, train_loader, test_loader = setup_dummy_data_train_test()
    num_classes = np.unique(dummy_sequences).shape[0] + 1
    WT = F.one_hot(torch.Tensor(dummy_sequences[0]).to(torch.int64), num_classes=num_classes).flatten().float()
    vae = VAE(z_dim=2, encoder_dim=[100], decoder_dim=[100], input_dims=WT.shape[0], use_cuda=False, wt=WT,
              dropout=0.01,
              num_categories=num_classes)
    optimizer = Adam({"lr": 0.001, "weight_decay": 0.000001})
    svi = SVI(vae.model, vae.guide, optimizer, loss=JitTrace_ELBO())
    vae.train()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(TRAIN_EPOCHS):
        total_epoch_loss_train = train(svi, train_loader, False)
        print(f"[epoch {epoch}] avrg. train loss: {total_epoch_loss_train}")
        if epoch % VALIDATION == 0:
            total_epoch_loss_test = evaluate(svi, test_loader, False)
            print(f"[epoch {epoch}] avrg. test loss: {total_epoch_loss_test}")
    vae.eval()
    return vae


def setup_UBQ_VAE():
    # LOAD AND TEST VAE ON 1UBQ SEQUENCE
    family_seqs, test_seqs, test_y = parse_UBQ()
    num_classes = np.unique(family_seqs).shape[0] + 1
    WT = F.one_hot(torch.tensor(family_seqs[0], dtype=torch.int64),
                   num_classes=num_classes).flatten().float()
    model_FILENAME = f"/home/rimichael/pro_tooling/models/VAE_tubq_z55_h[1700, 1200]_e200_d0.065_wTrue.pt"
    vae = VAE(z_dim=55, encoder_dim=[1700], decoder_dim=[1200], input_dims=WT.shape[0],
              use_cuda=False, wt=WT, dropout=0.065,
              num_categories=num_classes)
    vae.load_state_dict(torch.load(model_FILENAME))
    vae.eval()
    return family_seqs, vae


vae = setup_dummy_VAE()
test_dummy_sequences, _, _ = setup_dummy_data_train_test()
L = test_dummy_sequences.shape[1]
num_classes = np.unique(test_dummy_sequences).shape[0] + 1


def linear_kernel(x: torch.Tensor, y: torch.Tensor):
    return torch.dot(x.T, y) + 1


def test_naive_linear_kernel():
    x = F.one_hot(torch.tensor(test_dummy_sequences[0], dtype=torch.int64), num_classes=num_classes).flatten().float()
    y = F.one_hot(torch.tensor(test_dummy_sequences[0], dtype=torch.int64), num_classes=num_classes).flatten().float()
    k = linear_kernel(x, y)
    print(test_dummy_sequences[0])
    print(x)
    print(k)
    assert k.shape == ()
    assert k > 0
    assert k < L +1


def test_naive_sequence_computation_linear_kernel():
    n = test_dummy_sequences.shape[0]
    K = np.zeros([n, n])
    for p in range(n):
        for q in range(n):
            s_x = F.one_hot(torch.tensor(test_dummy_sequences[p], dtype=torch.int64), num_classes=num_classes).flatten().float()
            s_y = F.one_hot(torch.tensor(test_dummy_sequences[q], dtype=torch.int64), num_classes=num_classes).flatten().float()
            K[p, q] += linear_kernel(s_x, s_y)
    print(K)
    vec_k =
    assert np.testing.assert_almost_equal(K, )

