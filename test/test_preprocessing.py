def test_preprocessing(self):
        ref_file = loadmat(os.path.join(os.path.dirname(__file__), os.path.join("model", "1PGAsmall.mat")))
        ref_K = ref_file["kernel_matrix"]#[0, 0]
        ref_model = ref_file["model"]
        sigma_T = ref_model["stdT"][0, 0]
        num_wet_lab_obs = ref_K.shape[0] - sigma_T.shape[0] - 1

        a, b, c, d = ref_model["theta"][0, 0][0, :]
        def f(y):
            return b * y + a * np.exp(c * y) + d

        contact_graph, x_wild_type, y_wild_type, X_wetlab, y_wetlab, X_insilico, y_insilico, y_train_wetlab_matching, \
            y_insilico_matching, X_test, y_test = get_split_training_and_test_data("1PGA", cutoff_distance=5.,
                                                                                   p=np.arange(num_wet_lab_obs))
        y_wetlab = np.vstack([y_test, y_wetlab])

        y_insilico = y_insilico[:20, :]

        y_scaled = f(y_insilico)
        mean_y, max_y, y_wild_type, y_wetlab, y_scaled = preprocess_observations(y_wild_type, y_wetlab, y_scaled)

        self.assertAlmostEqual(mean_y, ref_model["my"][0, 0][0, 0])
        self.assertAlmostEqual(max_y, ref_model["ymax"][0, 0][0, 0])
        target_y = ref_file["target_y"]
        np.testing.assert_allclose(y_wild_type, target_y[[0], :])
        np.testing.assert_allclose(y_wetlab, target_y[1:num_wet_lab_obs+1, :])
        np.testing.assert_allclose(y_scaled, target_y[num_wet_lab_obs+1:, :])