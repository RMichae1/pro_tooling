import numpy as np
from vae import VAE
from contact_mapper import ContactMapper
from graphkernel import VaeKernel
from utility import parse_UBQ
import torch.nn.functional as F
import torch
from torch.distributions import Normal, Categorical


def likelihood(_vae: VAE, seq: np.array, idx: int, latent_sample: np.array) -> np.array:
    """
    _vae is VAE object for S-computations
    seq is n_sequences x length of sequence
    idx is int position in sequence
    """
    oh_x = F.one_hot(torch.Tensor(seq).to(torch.int64), num_classes=_vae.num_categories).float()
    n_samples = latent_sample.shape[0]
    p_z = Normal(loc=torch.zeros(1), scale=torch.ones(1)).log_prob(latent_sample).sum(1).exp().detach().numpy()
    z_x_loc, z_x_scale = _vae.encoder(oh_x)
    # transform latent sample z' => z
    z = z_x_loc + torch.Tensor(latent_sample)*torch.sqrt(z_x_scale)
    q_z_x = Normal(z_x_loc, z_x_scale).log_prob(z).sum(1).exp().detach().numpy()
    ll_p_x_z = Categorical(_vae.decoder(z).exp()).log_prob(torch.Tensor(seq)).permute(1, 0).detach().numpy()
    # print(ll_p_x_z[:, :idx][:, :idx, seq[:, :idx]])
    joint_p_left_sequence = np.sum(ll_p_x_z[:idx][:idx, seq[:idx]])
    joint_p_right_sequence = np.sum(ll_p_x_z[idx+1:][idx+1:, seq[idx+1:]])
    p_x_z_not_i = np.exp(joint_p_left_sequence + joint_p_right_sequence)
    # p(x_i==a) is exp(log_likelihood(p_x[n_sample, position, residue]))
    p_x_i_x_not_i = (1/n_samples) * (np.exp(ll_p_x_z[idx, seq[idx]]) * p_x_z_not_i * p_z) / q_z_x
    return np.array(p_x_i_x_not_i)


def naive_v_K(sequences: np.ndarray, adj: np.ndarray, vae: VAE, sample_size=1) -> np.ndarray:
    """
    Kernel as described in the paper
    """
    latent_sample = torch.normal(0, 1, size=(sample_size, vae.z_dim)).float()
    n = sequences.shape[0]
    elements = vae.num_categories
    K = np.zeros([elements, n, n])
    temp_K = np.zeros([elements, n, n])
    Ks = []
    for seq in sequences:
        for p in range(n):
            for q in range(n):
                for idx in range(sequences.shape[1]):
                    nbps = adj[idx]
                    temp_K.fill(0.)
                    for l in nbps:
                        p_vals = likelihood(vae, seq, l, latent_sample)
                        for i in range(elements):
                            temp_K[i, p, q] += p_vals[i]
                    p_vals = likelihood(vae, seq, idx, latent_sample)
                    for i in range(elements):
                        temp_K[i, p, q] *= p_vals[i]
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
        Ks.append(K)
    return Ks


# LOAD AND TEST VAE ON 1 UBQ SEQUENCE
family_seqs, test_seqs, test_y = parse_UBQ()
num_classes = np.unique(family_seqs).shape[0] + 2
WT = F.one_hot(torch.tensor(family_seqs[0], dtype=torch.int64),
               num_classes=num_classes).flatten().float()
model_FILENAME = f"/home/rimichael/pro_tooling/models/VAE_tubq_z2_h[1700, 1200]_e200_d0.065_wTrue.pt"
contact_map = ContactMapper(pdb_file=f"/home/rimichael/pro_tooling/pdb/1ubq.pdb", tri_dist=True)
vae = VAE(z_dim=2, encoder_dim=[1700], decoder_dim=[1200], input_dims=WT.shape[0],
          use_cuda=False, wt=WT, dropout=0.065,
          num_categories=num_classes)
vae.load_state_dict(torch.load(model_FILENAME))
vae.eval()


def test_vectorized_VAE_kernel():
    sequence = family_seqs[0][np.newaxis, :]
    ref_adj = [c for elem, c in contact_map.adjacency]
    naive_vae_val = naive_v_K(sequence, ref_adj, vae, num_classes)
    v_k = VaeKernel(vae, sample_size=1)
    s_vae_val = v_k.k(sequence, adjacencies=ref_adj)
    np.testing.assert_almost_equal(s_vae_val, naive_vae_val)


def test_tiny_categorical():
    sequence = np.array([[1, 2, 1, 2, 5, 6]])
    cat_ll = Categorical(logits=sequence).log_prob(sequence)
    elementwise = []
    for idx in range(sequence.shape[1]):
        left = []
        right = []
        for elem in sequence[0]:
            if elem == idx:
                continue
            elif elem < idx:
                left.append(cat_ll[elem])
            elif elem > idx:
                right.append(cat_ll[elem])
        one_line = np.exp(np.sum(cat_ll[0, :idx][:idx, sequence[:idx]]) + np.sum(cat_ll[0, idx+1:][idx+1:, sequence[idx+1]]))
        assert one_line == np.sum(np.array(left))+np.sum(np.array(right))