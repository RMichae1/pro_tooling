import numpy as np
from contact_mapper import ContactMapper
from graphkernel import VaeKernel
from utility import parse_UBQ, WeightedMSADataset, seq_collate
import torch.nn.functional as F
import torch
from torch.distributions import Normal, Categorical
from vae import VAE, train, evaluate
import pyro
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader
import random


def setup_dummy_data_train_test():
    N = 250  # number of sequences in MSA
    L = 10  # length of the sequence
    AA = 19  # number of amino acids
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
    # LOAD AND TEST VAE ON 1 UBQ SEQUENCE
    family_seqs, test_seqs, test_y = parse_UBQ()
    num_classes = np.unique(family_seqs).shape[0] + 2
    WT = F.one_hot(torch.tensor(family_seqs[0], dtype=torch.int64),
                   num_classes=num_classes).flatten().float()
    model_FILENAME = f"/home/rimichael/pro_tooling/models/VAE_tubq_z2_h[1700, 1200]_e200_d0.065_wTrue.pt"
    vae = VAE(z_dim=2, encoder_dim=[1700], decoder_dim=[1200], input_dims=WT.shape[0],
              use_cuda=False, wt=WT, dropout=0.065,
              num_categories=num_classes)
    vae.load_state_dict(torch.load(model_FILENAME))
    vae.eval()
    return family_seqs, vae


def build_normalization_vec(seq, idx, cat, p_x_z_not_i, p_z, q_z_x, AAs=20):
    normalization_vec = []
    for aa in range(AAs):
        _seq = seq.copy()
        _seq[:, idx] = aa
        cat_ll_i = cat.log_prob(torch.Tensor(_seq)).detach().numpy()[:, idx]  # log likelihood of categorical at pos idx
        normalization_vec.append((np.exp(cat_ll_i) * p_x_z_not_i * p_z) / q_z_x)
    return np.array(normalization_vec)


def likelihood(_vae: VAE, seq: np.array, idx: int, latent_sample: np.array) -> np.array:
    """
    _vae is VAE object for S-computations
    seq is n_sequences x length of sequence
    idx is int position in sequence
    """
    oh_x = F.one_hot(torch.Tensor(seq).to(torch.int64), num_classes=_vae.num_categories).float()
    n_samples = latent_sample.shape[0]
    p_z = Normal(loc=torch.zeros(_vae.z_dim), scale=torch.ones(_vae.z_dim)).log_prob(latent_sample).sum(1).exp().detach().numpy()
    z_x_loc, z_x_scale = _vae.encoder(oh_x)
    # transform latent sample z' => z
    z = z_x_loc + torch.Tensor(latent_sample)*torch.sqrt(z_x_scale)
    q_z_x = Normal(z_x_loc, z_x_scale).log_prob(z).sum(-1).exp().detach().numpy()
    ll_p_x_z = Categorical(_vae.decoder(z).exp()).log_prob(torch.Tensor(seq)).detach().numpy()
    joint_p_left_x_not_i = np.sum(ll_p_x_z[:, :idx])  # log_prob already evaluates sequence value expressions
    joint_p_right_x_not_i = np.sum(ll_p_x_z[:, idx+1:])
    p_x_z_not_i = np.exp(joint_p_left_x_not_i + joint_p_right_x_not_i)
    normalization_vec = build_normalization_vec(seq=seq, idx=idx, cat=Categorical(_vae.decoder(z).exp()),
                                                p_x_z_not_i=p_x_z_not_i, p_z=p_z, q_z_x=q_z_x)
    p_x_not_i = np.mean(normalization_vec)
    # p_x_not_i = (1/n_samples) * (np.exp(np.sum(ll_p_x_z[idx, :])) * p_x_z_not_i * p_z) / q_z_x
    p_x_i_x_not_i = (1/p_x_not_i) * (1 / n_samples) * np.sum((normalization_vec[idx] * normalization_vec))
    return np.array(p_x_i_x_not_i)


def naive_v_K(sequences: np.ndarray, adj: np.ndarray, vae: VAE, sample_size=1) -> np.ndarray:
    """
    Kernel as described in the paper
    """
    latent_sample = torch.normal(0, 1, size=(sample_size, vae.z_dim)).float()
    n = sequences.shape[0]
    K = np.zeros([n, n])
    temp_K = np.zeros([n, n])
    for p in range(n):
        for q in range(n):
            for idx in range(sequences.shape[1]):
                nbps = adj[idx]
                temp_K.fill(0.)
                for l in nbps:
                    temp_K[p, q] += likelihood(vae, sequences, l, latent_sample)
                temp_K[p, q] *= likelihood(vae, sequences, idx, latent_sample)
                K += temp_K
    print(K)
    # normalize
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            K[p, q] /= (np.sqrt(K[p, p]) * np.sqrt(K[q, q]))
    # set diagonal explicitly
    # for i in range(0, n):
    #     K[i, i] = 1  # TODO validate if this is still true
    return K


def test_naive_VAE_kernel():
    vae = setup_dummy_VAE()
    test_dummy_sequences, _, _ = setup_dummy_data_train_test()
    L = test_dummy_sequences.shape[1]
    adj = [np.random.randint(0, L, [np.random.randint(0, L)]) for _ in range(0, L)]
    k_mat = naive_v_K(test_dummy_sequences[0][np.newaxis, :], adj, vae)

    np.testing.assert_almost_equal(k_mat, np.zeros(k_mat.shape))


def test_vectorized_VAE_kernel():
    family_seqs, vae = setup_UBQ_VAE()
    contact_map = ContactMapper(pdb_file=f"/home/rimichael/pro_tooling/pdb/1ubq.pdb", tri_dist=True)
    num_classes = np.unique(family_seqs).shape[0] + 2
    sequence = family_seqs[0][np.newaxis, :]
    ref_adj = [c for elem, c in contact_map.adjacency]
    naive_vae_val = naive_v_K(sequence, ref_adj, vae)
    v_k = VaeKernel(vae, sample_size=1)
    s_vae_val = v_k.k(sequence, adjacencies=ref_adj)
    np.testing.assert_almost_equal(s_vae_val, naive_vae_val)


def test_tiny_categorical():
    cat = Categorical(torch.Tensor([0.75, 0.2, 0.05]))
    sequences = torch.Tensor([[0, 1, 2, 1, 2, 0, 1]])
    for s, sequence in enumerate(sequences):
        cat_ll = cat.log_prob(sequence).detach().numpy()
        for idx in range(len(sequence)):
            left = []
            right = []
            for i, elem in enumerate(sequence):
                if i == idx:
                    continue
                elif i < idx:
                    left.append(cat_ll[i])
                elif i > idx:
                    right.append(cat_ll[i])
            assert np.sum(np.array(left)) == np.sum(cat_ll[:idx][sequence[:idx]])
            print(cat_ll[idx+1:])
            assert np.sum(np.array(right)) == np.sum(cat_ll[idx+1:][sequence[idx+1:]])