import pytest
import numpy as np
import torch
import os
from scipy.io import loadmat
from graphkernel import ContactMapper
from protein_representation import ProteinCollection
from utility import parse_matlab_mutation_file, parse_mutations 
from utility import preprocess_observations, convert_aa_sequence
from utility import convert_graph_from_matlab_file, get_split_training_and_test_data
from gp_regression import GPRegression
from scipy.linalg import solve_triangular


### TEST PREDICTIONS
# load 1pga small reference prediction
ref_file = loadmat(os.path.join(os.path.dirname(__file__), os.path.join("data", "1PGAsmall_pred.mat")))
ref_mean_pred = ref_file["ypred"]
ref_unscaled_pred = ref_file["unscaled_pred"]
ref_uncertainty = ref_file["stdPred"]
ref_noise = np.square(ref_file["noise"][:, 0])
ref_beta = ref_file["beta"]
sigma_T = ref_file["model"]["stdT"][0, 0]
a, b, c, d = ref_file["model"]["theta"][0, 0][0, :]

# Bayesian scaling
def f(y):
    return b * y + a * np.exp(c * y) + d

ref_K = ref_file["Kfull"]
num_exp_obs = ref_K.shape[0] - sigma_T.shape[0] - 1 
contact_graph, X_wt, y_wt, X_exp, y_exp, X_is, y_is, y_train_exp_match, y_is_match, X_test_, y_test = get_split_training_and_test_data("1PGA", 
                                                                                                    cutoff_distance=5., p=np.arange(num_exp_obs))
X_test = torch.Tensor(X_test_).type(torch.float64)
X_is = X_is[:20, :]
y_is = y_is[:20, :]
def test_parsed_ys_against_target():
    assert np.vstack([y_wt, y_exp, y_is, y_test]).shape[0] == ref_file["training_targets"].shape[0]

mut_exp = parse_matlab_mutation_file("./data/ddg_protherm.mat", query="ddg_protherm")
mut_is = parse_matlab_mutation_file("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")
cm = ContactMapper(pdb_file="./pdb/1pga.pdb", tri_dist=True)
cm.adjacency = contact_graph # set adjacency here as well for downstream consistency
prot = ProteinCollection(cm, pdb_ID="1PGA", mutations_exp=mut_exp, mutations_sim=mut_is)

y_scaled = f(y_is)
mean_y, max_y, y_wt, y_exp, y_scaled = preprocess_observations(y_wt, y_exp, y_scaled)

def test_predict_ymax():
    assert max_y == pytest.approx(ref_file["model"]["ymax"][0, 0][0, 0], rel=0.02)

model = GPRegression(prot, X_wt.reshape(1, len(X_wt)), np.vstack([X_test_, X_exp]), X_is, 
            y_wt, np.vstack([y_test, y_exp]), y_scaled, max_y, mean_y, contact_graph, torch.Tensor(sigma_T))
# set model parameters to account for training/testing specs
model.set_train_index(np.concatenate([np.array([0]), np.arange(21, model.y.shape[0])]))
model.set_test_index(np.arange(1, 21))

def test_model_dimensions():
    assert model.X_train.shape[0] == len(model.idx_train)
    assert np.vstack([X_wt, model.X_exp[20:,:], X_is]).shape[0] == len(model.idx_train)
    assert len(model.y_train) == len(model.idx_train) 
    assert X_test_.shape[0] == len(model.idx_test)

def test_targets():
    np.testing.assert_allclose(model.y[0, :], ref_file["training_targets"][0, :], atol=0.01)
    assert model.y.shape[0] == ref_file["training_targets"].shape[0]
    np.testing.assert_allclose(model.y[0:, :], ref_file["training_targets"][0:, :], atol=0.015)

noise = model.set_noise_term().detach().numpy()[model.idx_train]

def test_noise_equal():
    np.testing.assert_almost_equal(noise[0], ref_noise[0])

K = np.zeros([noise.size, noise.size])
KZX_ref = np.hstack([ref_K[1:21, [0]], ref_K[1:21, 21:]])
K[0, 0] = ref_K[0, 0]
K[0, 1:] = ref_K[0, 21:]
K[1:, 0] = K[0, 1:].T
K[1:, 1:] = ref_K[21:, 21:]

#beta = np.linalg.solve(K + np.diag(noise), KZX_ref.T)

# def test_beta_ref():
#     # TODO beta has large difference
#     np.testing.assert_allclose(beta.T, ref_beta, atol=0.01)

# def test_mean_pred_local():
#     model.y_test = np.vstack([y_wt, y_exp])
#     ref_mean_pred_local = beta.T.dot(model.y_test)
#     np.testing.assert_allclose(ref_mean_pred_local, ref_unscaled_pred)

def test_mean_prediction_against_ref():
    print(model.y.shape)
    ref_mean_corrected = np.linalg.solve(K, KZX_ref.T).T.dot(model.y_train)
    ref_mean_corrected *= max_y
    ref_mean_corrected += mean_y
    mu, _ = model._fit()
    np.testing.assert_allclose(mu.detach().numpy(), ref_mean_corrected)