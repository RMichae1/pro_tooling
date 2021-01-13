import pytest
import numpy as np
import os
from scipy.io import loadmat
from graphkernel import ContactMapper
from protein_representation import ProteinCollection
from utility import parse_mutations
from gp_regression import GPRegression
 
def preprocess_observations(y_wild_type, y_wetlab, y_scaled):
    y = np.vstack([y_wild_type, y_wetlab, y_scaled])
    mean_y = np.mean(y)
    y -= mean_y
    max_y = np.max(np.abs(y))
    y /= max_y
    return mean_y, max_y, y[[0], :], y[1:y_wetlab.shape[0]+1, :], y[1+y_wetlab.shape[0]:, :]
 
def test_model():
    """
    Tests the mGPfusion model versus the matlab reference implementation.
    :return:
        None
    """

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

    # some consistency checks on our loaded data
    ref_L = np.linalg.cholesky(ref_K + np.square(np.diag(ref_noise)))  # reference Cholesky
    # make sure the reference determinant is in line with the reference Cholesky
    _ref_det = 2 * np.sum(np.log(np.diag(ref_L)))
    # FIRST TEST: TEST CORRECTNESS OF PROVIDED DATA
    pytest.approx(_ref_det, ref_det[0, 0])  
    # the loss without the contributions from the prior
    ref_gp_loss = ref_K.shape[0] / 2 * np.log(2 * np.pi) + ref_det/2 + ref_yKy/2
    _ref_loss = ref_gp_loss - ref_prior_E - ref_prior_R
    # SECOND TEST: TEST CORRECTNESS OF REFERENCE
    pytest.approx(ref_loss[0, 0], _ref_loss[0, 0])

    # Bayesian scaling
    def f(y):
        return b * y + a * np.exp(c * y) + d
    num_wet_lab_obs = ref_K.shape[0] - sigma_T.shape[0] - 1

    # load our data
    cm = ContactMapper(pdb_file="./pdb/1pga.pdb", tri_dist=True)
    mut_exp = parse_mutations("./data/ddg_protherm.mat", query="ddg_protherm")
    mut_is = parse_mutations("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")
    assert isinstance(cm.contact_map, np.ndarray)

    prot = ProteinCollection(cm, pdb_ID="1PGA", mutations_exp=mut_exp, mutations_sim=mut_is)
    y_wild_type = np.array([prot.ΔΔg[0]])[:, np.newaxis]
    X_wetlab = prot.mut_ids_exp
    y_wetlab = np.array(prot.ΔΔg_exp)[:, np.newaxis]
    X_insilico = prot.mut_ids_is[:20]
    y_insilico = np.array(prot.ΔΔg_is[:20])[:, np.newaxis]

    # apply preprocessing
    y_scaled = f(y_insilico)
    mean_y, max_y, y_wild_type, y_wetlab, y_scaled = preprocess_observations(y_wild_type, y_wetlab, y_scaled)
    # THIRD TEST: SCALING AND NORMALIZATION OF y-VALUES
    assert max_y == pytest.approx(ref_file["model"]["ymax"][0, 0][0, 0], rel=0.003)

    # build model
    model = GPRegression(protein_representation=prot)
    m = MGPFusionModel(x_wild_type=x_wild_type, X_wetlab=X_wetlab, X_in_silico=X_insilico, y_wild_type=y_wild_type,
                       y_wetlab=y_wetlab, y_in_silico=y_scaled, sigma_T=sigma_T, contact_map=contact_graph,
                       ymax=max_y, ymean=mean_y)

    # now comes the actual testing
    # self.assertEqual(len(m.trainable_parameters), 3 + 21) # TODO GPRegression trainable parameters = weights + 2xsigma+t

    K = m.kernel(m.X)
    np.testing.assert_almost_equal(K.numpy(), ref_K)
    L = np.linalg.cholesky(K.numpy() + np.square(np.diag(ref_noise)))

    # check determinant
    self.assertAlmostEqual(ref_det[0, 0], 2 * np.sum(np.log(np.diag(L))))

    # check quadratic form
    t = solve_triangular(L, m.Y, lower=True)
    self.assertAlmostEqual(ref_yKy[0, 0], t.T.dot(t)[0, 0])

    # check loss with contributions from prior
    gp_loss = m.maximum_log_likelihood_objective().numpy()
    self.assertAlmostEqual(-gp_loss, ref_gp_loss[0, 0])
    loss = m.training_loss().numpy()

    loss = loss - 2 * np.log(max_y)  # a contribution from the Gamma priors

    self.assertAlmostEqual(ref_prior_R[0, 0] + ref_prior_E[0, 0], -loss - gp_loss)
    np.testing.assert_almost_equal(loss, ref_loss)