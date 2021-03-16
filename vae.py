import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
import pandas as pd
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
    def __init__(self, z_dim, hidden_dim, input_dims):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, input_dims)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img


class VAE(nn.Module):
    def __init__(self, z_dim, hidden_dim, input_dims, wt, use_cuda=False):
        super().__init__()
        self.input_dims = input_dims
        self.encoder = Encoder(z_dim, hidden_dim, input_dims)
        self.decoder = Decoder(z_dim, hidden_dim, input_dims)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.mse = nn.MSELoss()
        self.sequence_length = wt.shape[0]

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
            loc_seq = self.decoder.forward(z)
            pyro.sample("obs", dist.Bernoulli(loc_seq, validate_args=True).to_event(1), obs=x.reshape(-1,
                                                                                                       self.input_dims))

    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale, constraints.positive).to_event(1))

    def representation(self, z: dist) -> torch.Tensor:
        z_repr = self.decoder(z.loc)
        sample = dist.Bernoulli(z_repr).sample()
        return sample

    def reconstruct(self, x):
        z_loc, z_scale = self.encoder(x)
        z_dist = dist.Normal(z_loc, z_scale)
        reconstruction = self.representation(z_dist)
        return reconstruction

    def log_p(self, x): 
        z_dist = self.encoder(x)
        z_dist = dist.Normal(z_dist[0], z_dist[1])
        kld = self.kld_loss(z_dist)
        reconstruction = self.representation(z_dist).view(self.sequence_length, 23)
        # TODO ensure input: (N, C)
        log_p = self.ce_loss(reconstruction, x)
        # log_p = nll_loss(reconstruction, x, reduction="none").mul(-1).sum(1) # TODO requires log_softmax in Encoder
        # log_p = dist.Bernoulli(self.decoder(z_dist.loc)).log_prob(x.flatten()).sum(1) # TODO CORRECT THAT
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

