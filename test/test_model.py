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

def test_model():
    """
    Tests the mGPfusion model versus the matlab reference implementation.
    :return:
        None
    """
    pass

# Initialization
# Loading required target values from mat file
ref_file = loadmat(os.path.join(os.path.dirname(__file__), os.path.join("data", "1PGAsmall.mat")))
ref_loss = ref_file["model"]["mll"][0, 0]
ref_noise = np.squeeze(ref_file["noise"])  # [0, 0]
ref_K = ref_file["kernel_matrix"]#[0, 0]
ref_det = ref_file["mll_components_struct"]["logdet"][0, 0]
ref_yKy = ref_file["mll_components_struct"]["square_form"][0, 0]
ref_prior_E = ref_file["mll_components_struct"]["prior_E"][0, 0]
ref_prior_R = ref_file["mll_components_struct"]["prior_R"][0, 0]
sigma_T = ref_file["model"]["stdT"][0, 0]
a, b, c, d = ref_file["model"]["theta"][0, 0][0, :]  # parameters for the Bayesian scaling
# correct adjacencies
pga_file = loadmat(os.path.join(os.path.dirname(__file__), os.path.join("data", "1PGA.mat")))
ref_contact_graph = convert_graph_from_matlab_file(pga_file["al"])

# some consistency checks on our loaded data
ref_L = np.linalg.cholesky(ref_K + np.square(np.diag(ref_noise)))  # reference Cholesky
# make sure the reference determinant is in line with the reference Cholesky
_ref_det = 2 * np.sum(np.log(np.diag(ref_L)))

# the loss without the contributions from the prior
ref_gp_loss = ref_K.shape[0] / 2 * np.log(2 * np.pi) + ref_det/2 + ref_yKy/2
_ref_loss = ref_gp_loss - ref_prior_E - ref_prior_R

num_wet_lab_obs = ref_K.shape[0] - sigma_T.shape[0] - 1

# Bayesian scaling
def f(y):
    """
    Bayesian Scaling from reference values
    """
    return b * y + a * np.exp(c * y) + d
 
def test_reference_determinant_correct():
    """  
    test correctness of reference determinant
    """
    assert _ref_det == pytest.approx(ref_det[0, 0])  

def test_reference_loss_correct():
    """
    test corrtetness of reference loss
    """
    assert ref_loss[0, 0] == pytest.approx(_ref_loss[0, 0])

# load "our" data and derive mutations
cm = ContactMapper(pdb_file="./pdb/1pga.pdb", tri_dist=True)
cm.adjacency = ref_contact_graph # set adjacency here as well for downstream consistency
mut_exp = parse_matlab_mutation_file("./data/ddg_protherm.mat", query="ddg_protherm")
mut_is = parse_matlab_mutation_file("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")

prot = ProteinCollection(cm, pdb_ID="1PGA", mutations_exp=mut_exp, mutations_sim=mut_is)
mut_S_exp, mut_adj_exp, ΔΔg_exp, mut_ids_exp = parse_mutations(mutation_dict=mut_exp.get(prot.pdb_ID), 
                                                sequence=prot.sequence, adjacency=ref_contact_graph)
mut_S_is, mut_adj_is, ΔΔg_is, mut_ids_is = parse_mutations(mutation_dict=mut_is.get(prot.pdb_ID), 
                                                sequence=prot.sequence, adjacency=ref_contact_graph)
# select only for 20 insilico mutations
X_wetlab = convert_aa_sequence(mut_S_exp)
X_insilico = convert_aa_sequence(mut_S_is[:20])
X_wild_type = convert_aa_sequence([prot.sequence])
y_wild_type = np.array([prot.ΔΔg[0]])[:, np.newaxis]
y_wetlab = np.array(ΔΔg_exp)[:, np.newaxis]
y_insilico = np.array(ΔΔg_is[:20])[:, np.newaxis]

# apply preprocessing
y_scaled = f(y_insilico)
mean_y, max_y, y_wild_type, y_wetlab, y_scaled = preprocess_observations(y_wild_type, y_wetlab, y_scaled)

# build model from loaded data
model = GPRegression(protein_representation=prot, X_wt=X_wild_type, X_exp=X_wetlab, X_is=X_insilico,
                    y_wt=y_wild_type, y_exp=y_wetlab, y_is=y_scaled, y_max=max_y, y_mean=mean_y,
                    σ_T=torch.Tensor(sigma_T), adjacencies=ref_contact_graph)
    
def test_y_scaling_and_normalization():
    assert max_y == pytest.approx(ref_file["model"]["ymax"][0, 0][0, 0])

def test_model_parameters():
    assert len(model.trainable_parameters) == (3 + 21)

#cov_mats = [mat[:model.X.shape[0], :model.X.shape[0]] for mat in model.covariance_matrices]
K = model.mWDK(X=model.X, covariance_matrices=model.covariance_matrices)
def test_K_computation_agains_ref():
    np.testing.assert_almost_equal(K.detach().numpy(), ref_K, decimal=3)

L = np.linalg.cholesky(K.detach().numpy() + np.square(np.diag(ref_noise)))
def test_determinant_against_ref():
    assert pytest.approx(2 * np.sum(np.log(np.diag(L)))) == ref_det[0, 0] 

def test_quadratic_form():
    t = solve_triangular(L, model.y, lower=True)
    assert pytest.approx(t.T.dot(t)[0, 0]) == ref_yKy[0, 0]

def test_noise_term():
    model_noise = model.set_noise_term().detach().numpy().flatten()
    np.testing.assert_almost_equal(np.square(ref_noise), model_noise)

# def test_log_likelihood_loss_w_prior():
#     gp_loss = model.neg_ll().detach().numpy()
#     np.testing.assert_almost_equal(gp_loss[0][0], ref_gp_loss[0,0])

def test_training_loss():
    model.parameter_optimization()
    loss = model.neg_ll().detach().numpy()
    loss = loss - 2 * np.log(max_y)  # a contribution from the Gamma priors
    np.testing.assert_almost_equal(loss, ref_loss, decimal=3)


