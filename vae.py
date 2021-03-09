import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
import pandas as pd
import torch
from torch import nn
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
        z_scale = torch.exp(self.fc22(hidden))
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

    def reconstruct_seq(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        seq = self.decoder(z)
        return seq

    def likelihood(self, x): 
        z_loc, z_scale = self.encoder(x)
        z_dist = dist.Normal(z_loc, z_scale)
        # TODO check if sampling is correct here - there is no y given
        samples = pyro.sample("y", z_dist)
        p = z_dist.log_prob(samples).exp()
        return p

    def log_odd_ratio(self, x):
        wt_loc, wt_scale = self.encoder(self.wt)
        wt_dist = dist.Normal(wt_loc, wt_scale)
        wt_log_odds = wt_dist.log_prob(pyro.sample("y_wt", wt_dist)).exp()
        x_loc, x_scale = self.encoder(x)
        x_dist = dist.Normal(x_loc, x_scale)
        x_log_odds = x_dist.log_prob(pyro.sample("y_x", x_dist)).exp()
        ratio = x_log_odds/wt_log_odds
        return ratio


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

