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

torch.manual_seed(0)


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


@torch.no_grad()
def stable_likelihoods(_vae: VAE, seq: np.array, z_dist, idx: int, latent_sample: np.array) -> np.array:
    """
    COMPUTE NUMERICALLY STABLE LIKELIHOODS
    _vae is VAE object for S-computations
    seq is n_sequences x length of sequence
    idx is int position in sequence
    """
    L = len(seq)
    z_loc, z_scale = z_dist
    # transform latent sample z' => z
    z = z_loc + torch.Tensor(latent_sample) * torch.sqrt(z_scale)
    p_z = Normal(loc=torch.zeros(_vae.z_dim), scale=torch.ones(_vae.z_dim)).log_prob(z).sum(1).detach().numpy()
    q_z_x = Normal(z_loc, z_scale).log_prob(z).sum(-1).detach().numpy()
    cat_p = Categorical(_vae.decoder(z).exp()).probs.log()
    ll_left_p_x_z = np.sum([cat_p[:, :idx][:, i, s] - q_z_x/L for i, s in enumerate(seq[:idx])])
    ll_right_p_x_z = np.sum([cat_p[:, idx+1:][:, i, s] - q_z_x/L for i, s in enumerate(seq[idx+1:])])
    cat_log_prob_x_not_i = ll_left_p_x_z + ll_right_p_x_z
    # TODO: pull p(z) into normalization as well
    p_x_i_x_not_i = cat_p[:, idx] + cat_log_prob_x_not_i + p_z - q_z_x/L
    return p_x_i_x_not_i


@torch.no_grad()
def likelihoods(_vae: VAE, seq: np.array, z_dist, idx: int, latent_sample: np.array) -> np.array:
    """
    NAIVE LIKELIHOOD COMPUTATION:
    "exp, ..., exp everywhere" - calculating directly with the likelihoods - NOT the log-likelihoods.
    """
    z_loc, z_scale = z_dist
    z = z_loc + torch.Tensor(latent_sample)*torch.sqrt(z_scale)
    p_z = Normal(loc=torch.zeros(_vae.z_dim), scale=torch.ones(_vae.z_dim)).log_prob(z).sum(1).exp().detach().numpy()
    q_z_x = Normal(z_loc, z_scale).log_prob(z).sum(-1).exp().detach().numpy()
    cat_p = Categorical(_vae.decoder(z).exp()).probs
    p_left_x_not_i_z = np.prod([cat_p[:, :idx][:, i, s] for i, s in enumerate(seq[:idx])])
    p_right_x_not_i_z = np.prod([cat_p[:, idx+1:][:, i, s] for i, s in enumerate(seq[idx+1:])])
    cat_p_x_not_i_z = p_left_x_not_i_z * p_right_x_not_i_z
    p_x_i_x_not_i = ((cat_p[:, idx] * cat_p_x_not_i_z) * p_z) / q_z_x
    return p_x_i_x_not_i


@torch.no_grad()
def S_stable(_vae, seq_x, seq_y, idx: int, n_samples=1, fixed_sample=False):
    latent_samples = torch.normal(0, 1, size=(n_samples, _vae.z_dim)).float()
    if fixed_sample:
        latent_samples = torch.ones((n_samples, _vae.z_dim)).float()
    oh_x = F.one_hot(torch.Tensor(seq_x).to(torch.int64), num_classes=_vae.num_categories).float()
    oh_y = F.one_hot(torch.Tensor(seq_y).to(torch.int64), num_classes=_vae.num_categories).float()
    z_x_dist = _vae.encoder(oh_x)
    z_y_dist = _vae.encoder(oh_y)
    # Normal(z_x_dist[0], z_x_dist[1]).loc == z_x_dist[0]
    p_x = Categorical(_vae.decoder(z_x_dist[0]).exp()).log_prob(torch.Tensor(seq_x))
    p_y = Categorical(_vae.decoder(z_y_dist[0]).exp()).log_prob(torch.Tensor(seq_y))
    ll_x_i_x_not_i_vec = []
    ll_y_i_y_not_i_vec = []
    for sample in latent_samples:
        log_prob_x_i_x_not_i = stable_likelihoods(_vae, seq_x, z_dist=z_x_dist, idx=idx, latent_sample=sample)
        ll_x_i_x_not_i_vec.append(log_prob_x_i_x_not_i[0])
        log_prob_y_i_z_y_not_i = stable_likelihoods(_vae, seq_y, z_dist=z_y_dist, idx=idx, latent_sample=sample)
        ll_y_i_y_not_i_vec.append(log_prob_y_i_z_y_not_i[0])
    ll_x_i_x_not_i_vec = torch.stack(ll_x_i_x_not_i_vec)
    ll_y_i_y_not_i_vec = torch.stack(ll_y_i_y_not_i_vec)
    # SUM OF PROBABILITIES - hence exp operation
    ll_x_not_i = torch.log(torch.sum(torch.exp(torch.mean(ll_x_i_x_not_i_vec, axis=0))))
    ll_y_not_i = torch.log(torch.sum(torch.exp(torch.mean(ll_y_i_y_not_i_vec, axis=0))))
    normalized_ll_x_i_x_not_i = torch.mean(ll_x_i_x_not_i_vec, axis=0)[idx] - p_x[:, idx] - ll_x_not_i
    normalized_ll_y_i_y_not_i = torch.mean(ll_y_i_y_not_i_vec, axis=0)[idx] - p_y[:, idx] - ll_y_not_i
    return normalized_ll_x_i_x_not_i + normalized_ll_y_i_y_not_i


@torch.no_grad()
def S(_vae, seq_x, seq_y, idx: int, n_samples=1, fixed_sample=False):
    latent_samples = torch.normal(0, 1, size=(n_samples, _vae.z_dim)).float()
    if fixed_sample:
        latent_samples = torch.ones((n_samples, _vae.z_dim)).float()
    oh_x = F.one_hot(torch.Tensor(seq_x).to(torch.int64), num_classes=_vae.num_categories).float()
    oh_y = F.one_hot(torch.Tensor(seq_y).to(torch.int64), num_classes=_vae.num_categories).float()
    z_x_dist = _vae.encoder(oh_x)
    z_y_dist = _vae.encoder(oh_y)
    # Normal(z_x_dist[0], z_x_dist[1]).loc == z_x_dist[0]
    p_x = Categorical(_vae.decoder(z_x_dist[0]).exp()).log_prob(torch.Tensor(seq_x)).exp()
    p_y = Categorical(_vae.decoder(z_y_dist[0]).exp()).log_prob(torch.Tensor(seq_y)).exp()
    p_x_i_x_not_i_vec = []
    p_y_i_y_not_i_vec = []
    for sample in latent_samples:
        p_x_i_x_not_i = likelihoods(_vae, seq_x, z_dist=z_x_dist, idx=idx, latent_sample=sample)
        p_x_i_x_not_i_vec.append(p_x_i_x_not_i[0])
        p_y_i_y_not_i = likelihoods(_vae, seq_y, z_dist=z_y_dist, idx=idx, latent_sample=sample)
        p_y_i_y_not_i_vec.append(p_y_i_y_not_i[0])
    p_x_i_x_not_i_vec = torch.stack(p_x_i_x_not_i_vec)
    p_y_i_y_not_i_vec = torch.stack(p_y_i_y_not_i_vec)
    p_x_not_i = torch.sum(torch.mean(p_x_i_x_not_i_vec, axis=0))
    p_y_not_i = torch.sum(torch.mean(p_y_i_y_not_i_vec, axis=0))
    normalized_p_x_i_x_not_i = 1/p_x_not_i * torch.mean(p_x_i_x_not_i_vec, axis=0)[idx] / p_x[:, idx]
    normalized_p_y_i_y_not_i = 1/p_y_not_i * torch.mean(p_y_i_y_not_i_vec, axis=0)[idx] / p_y[:, idx]
    return np.log(normalized_p_x_i_x_not_i * normalized_p_y_i_y_not_i)


def naive_v_K(sequences: np.ndarray, adj: np.ndarray, vae: VAE, sample_size=1, stable=False, fixed_sample=False) -> np.ndarray:
    """
    Kernel as described in the paper
    """
    n = sequences.shape[0]
    K = np.zeros([n, n])
    temp_K = np.zeros([n, n])
    for p in range(n):
        for q in range(n):
            for idx in range(sequences.shape[1]):
                nbps = adj[idx]
                temp_K.fill(0.)
                for l in nbps:
                    temp_K[p, q] += S(vae, seq_x=sequences[p], seq_y=sequences[q], idx=l, n_samples=sample_size,
                                      fixed_sample=fixed_sample) if not stable else S_stable(
                        vae, sequences[p], sequences[q], idx=l, n_samples=sample_size, fixed_sample=fixed_sample)
                temp_K[p, q] *= S(vae, seq_x=sequences[p], seq_y=sequences[q], idx=idx, n_samples=sample_size,
                                  fixed_sample=fixed_sample) if not stable else S_stable(
                    vae, sequences[p], sequences[q], idx=idx, n_samples=sample_size, fixed_sample=fixed_sample)
                K += temp_K
    print("NAIVE KERNEL:")
    print(K)
    return K


def normalize_K(K):
    K=K.copy()
    # normalize
    for p in range(len(K)):
        for q in range(len(K)):
            if p == q:
                continue
            K[p, q] /= (np.sqrt(K[p, p]) * np.sqrt(K[q, q]))
    # # set diagonal explicitly
    # for i in range(0, n):
    #     K[i, i] = 1
    print(K)
    return K


vae = setup_dummy_VAE()
test_dummy_sequences, _, _ = setup_dummy_data_train_test()
L = test_dummy_sequences.shape[1]
adj = [np.random.randint(0, L, [np.random.randint(0, L)]) for _ in range(0, L)]


def test_naive_VAE_kernel():
    k_mat = naive_v_K(test_dummy_sequences[0][np.newaxis, :], adj, vae, fixed_sample=True)
    k_mat_stable = naive_v_K(test_dummy_sequences[0][np.newaxis, :], adj, vae, stable=True, fixed_sample=True)
    np.testing.assert_almost_equal(k_mat, k_mat_stable, decimal=6)


def test_naive_VAE_kernel_multiple_sequences():
    k_mat = naive_v_K(test_dummy_sequences[:3], adj, vae, fixed_sample=True)
    k_mat_stable = naive_v_K(test_dummy_sequences[:3], adj, vae, stable=True, fixed_sample=True)
    np.testing.assert_almost_equal(k_mat, k_mat_stable, decimal=6)


def test_naive_VAE_kernel_sampling():
    L = test_dummy_sequences.shape[1]
    adj = [np.random.randint(0, L, [np.random.randint(0, L)]) for _ in range(0, L)]
    k_mat = naive_v_K(test_dummy_sequences[0][np.newaxis, :], adj, vae, sample_size=100, fixed_sample=True)
    k_mat_stable = naive_v_K(test_dummy_sequences[0][np.newaxis, :], adj, vae, sample_size=100, stable=True,
                             fixed_sample=True)
    np.testing.assert_almost_equal(k_mat, k_mat_stable)


def test_vectorized_VAE_kernel():
    sequences = test_dummy_sequences[:3]
    #naive_vae_val = naive_v_K(sequences, adj, vae, sample_size=10, stable=True, fixed_sample=True)
    naive_vae_val = naive_v_K(sequences, adj, vae, sample_size=10, fixed_sample=True)
    v_k = VaeKernel(vae, sample_size=10, fixed_sample=True)
    s_vae_val = v_k.k(sequences, adjacencies=adj, normalize=False)
    norm = np.sqrt(np.diag(s_vae_val))
    norm_s_vae_val = s_vae_val / norm.dot(norm.T)
    # TEST UNNORMALIZED
    # np.testing.assert_almost_equal(naive_vae_val, s_vae_val)
    normalized_naive_vae_val = normalize_K(naive_vae_val)
    # TEST NORMALIZED
    np.testing.assert_almost_equal(normalized_naive_vae_val, norm_s_vae_val)


def test_vectorized_VAE_kernel_on_UBQ():
    family_seqs, vae = setup_UBQ_VAE()
    contact_map = ContactMapper(pdb_file=f"/home/rimichael/pro_tooling/pdb/1ubq.pdb", tri_dist=True)
    sequences = family_seqs[:3]
    ref_adj = [c for elem, c in contact_map.adjacency]
    naive_vae_val = naive_v_K(sequences, ref_adj, vae, sample_size=10, stable=True, fixed_sample=True)
    v_k = VaeKernel(vae, sample_size=10, fixed_sample=True)
    s_vae_val = v_k.k(sequences, adjacencies=ref_adj, normalize=False)
    norm = np.sqrt(np.diag(s_vae_val))
    norm_s_vae_val = s_vae_val / norm.dot(norm.T)
    # TEST UNNORMALIZED
    # np.testing.assert_almost_equal(naive_vae_val, s_vae_val)
    # TEST NORMALIZED
    normalized_naive_vae_val = normalize_K(naive_vae_val)
    np.testing.assert_almost_equal(normalized_naive_vae_val, norm_s_vae_val)
