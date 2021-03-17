import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
import numpy as np
import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.nn.functional import nll_loss
pyro.enable_validation()


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, input_dims):
        super().__init__()
        self.sequence_dims = input_dims
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.reshape(-1, self.sequence_dims)
        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden)) # TODO multiply with 0.5?
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, input_dims, dropout):
        super().__init__()
        self.seq_length = input_dims[0]
        self.categories = input_dims[1]
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, self.seq_length*self.categories)
        self.softplus = nn.Softplus()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, z):
        batch_size = z.shape[0]
        hidden = self.softplus(self.fc1(z))
        if self.dropout:
            seq_space = self.dropout(self.fc21(hidden)).view(batch_size, self.seq_length, -1)
        else:
            seq_space = self.fc21(hidden).view(batch_size, self.seq_length, -1)
        assert seq_space.shape[2] == self.categories
        loc_img = self.log_softmax(seq_space)
        return loc_img


class VAE(nn.Module):
    def __init__(self, z_dim, hidden_dim, input_dims, wt, use_cuda=False, dropout=None):
        super().__init__()
        self.input_dims = input_dims
        self.seq_length = wt.shape[0]
        self.categories = wt.shape[1]
        self.encoder = Encoder(z_dim, hidden_dim, input_dims)
        self.decoder = Decoder(z_dim, hidden_dim, 
                                input_dims=(self.seq_length, self.categories), dropout=dropout)
        self.mse = nn.MSELoss()

        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.wt = wt

    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale, constraints.positive).to_event(1))
            loc_seq = self.decoder.forward(z).exp()
            categorical_x = x.argmax(-1)
            pyro.sample("obs", dist.Categorical(loc_seq, validate_args=True).to_event(1), obs=categorical_x)

    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale, constraints.positive).to_event(1))

    def representation(self, z: dist) -> torch.Tensor:
        z_repr = self.decoder(z.loc).exp()
        sample = dist.Categorical(z_repr).sample()
        return sample

    def reconstruct(self, x):
        z_loc, z_scale = self.encoder(x)
        z_dist = dist.Normal(z_loc, z_scale)
        reconstruction = self.representation(z_dist)
        return reconstruction

    def log_p(self, x): 
        z_loc, z_scale = self.encoder(x)
        z_dist = dist.Normal(z_loc, z_scale)
        kld = self.kld_loss(z_dist)
        reconstruction = self.decoder(z_dist.loc)
        # nll loss input requires: (batch, categories, data)
        log_p = nll_loss(reconstruction.permute(0, 2, 1), x.argmax(-1)[np.newaxis, :], reduction="none").mul(-1).sum(1)
        # log_p = dist.Categorical(self.decoder(z_dist.loc).exp()).log_prob(x.argmax(-1)).sum(1) 
        elbo = log_p + kld
        return elbo, log_p, kld

    @staticmethod
    def kld_loss(z_dist: dist):
        prior = dist.Normal(torch.zeros_like(z_dist.loc), torch.ones_like(z_dist.scale))
        kld = kl_divergence(z_dist, prior).sum(dim=1)
        return kld

    def mse_loss(self, x):
        x_construct = self.reconstruct(x).argmax(-1).to(torch.float)
        return self.mse(x_construct, x)

    def mse_diff(self, x, y=None):
        if y is None:
            y = self.wt
        x_construct = self.reconstruct(x).argmax(-1).to(torch.float)
        y_construct = self.reconstruct(y).argmax(-1).to(torch.float)
        return self.mse(x_construct, y_construct)

    def latent_sample(self, x, n=10):
        z_loc, z_scale = self.encoder(x)
        return dist.Normal(z_loc, z_scale).rsample([n])


def train(svi, train_loader, use_cuda=False):
    epoch_loss = 0
    for tensor_list in train_loader:
        for x in tensor_list:
            if use_cuda:
                x = x.cuda()
            epoch_loss += svi.step(x)
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


def evaluate(svi, test_loader, use_cuda=False):
    test_loss = 0
    for tensor_list in test_loader:
        for x in tensor_list:
            if use_cuda:
                x = x.cuda()
            test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_train = test_loss / normalizer_test
    return total_epoch_loss_train

