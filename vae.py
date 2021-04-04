import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
import numpy as np
import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.nn.functional import nll_loss, relu
pyro.enable_validation()


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, input_dims, dropout=0.5):
        super().__init__()
        self.sequence_dims = input_dims
        encoding_layers = []
        current_dim = input_dims
        for hidden_dim in hidden_dims:
            encoding_layers.append(nn.Linear(current_dim, hidden_dim))
            encoding_layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim
        self.encoding_nn = nn.Sequential(*encoding_layers)
        self.mean = nn.Linear(current_dim, z_dim)
        self.log_var = nn.Linear(current_dim, z_dim)

    def forward(self, x):
        x = x.reshape(-1, self.sequence_dims)
        z_loc = self.mean(self.encoding_nn(x))
        z_scale = torch.exp(self.log_var(self.encoding_nn(x))) # TODO multiply with 0.5?
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, input_dims, num_categories, dropout=0.5):
        super().__init__()
        # TODO replace dropout with sparse layer
        decoding_layers = []
        self.categories = num_categories
        self.sequence_length = int(input_dims / num_categories)
        current_dim = z_dim
        for hidden_dim in hidden_dims:
            decoding_layers.append(nn.Linear(current_dim, hidden_dim))
            decoding_layers.append(nn.ReLU(inplace=True))
            decoding_layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        decoding_layers.append(nn.Linear(current_dim, input_dims))
        self.decoding_nn = nn.Sequential(*decoding_layers)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        z = self.decoding_nn(x)
        seq_space = z.view(batch_size, self.sequence_length, -1)
        assert seq_space.shape[2] == self.categories
        loc_img = self.log_softmax(seq_space)
        return loc_img


class VAE(nn.Module):
    def __init__(self, z_dim, encoder_dim, decoder_dim, input_dims, wt, num_categories, use_cuda=False, dropout=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.num_categories = num_categories
        self.sequence_length = int(input_dims / num_categories)
        self.encoder = Encoder(z_dim, encoder_dim, input_dims, dropout)
        self.decoder = Decoder(z_dim, decoder_dim, input_dims=input_dims, 
                                num_categories=num_categories, dropout=dropout)
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
        log_p = nll_loss(reconstruction.permute(0, 2, 1), 
                            x.view(self.sequence_length, self.num_categories).argmax(-1)[np.newaxis, :], reduction="none").mul(-1).sum(1)
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
        """ MSE loss is unintuitive/uninformative in a classification task"""
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
    for x, _, _ in train_loader:
        if use_cuda:
            x = x.cuda()
        epoch_loss += svi.step(x)
    total_epoch_loss_train = epoch_loss / len(train_loader.dataset)
    return total_epoch_loss_train


def evaluate(svi, test_loader, use_cuda=False):
    test_loss = 0
    for x, _, _ in test_loader:
        if use_cuda:
            x = x.cuda()
        test_loss += svi.evaluate_loss(x)
    total_epoch_loss_train = test_loss / len(test_loader.dataset)
    return total_epoch_loss_train

