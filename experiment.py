import os
import pickle
import numpy as np
import pandas as pd
from contact_mapper import ContactMapper
from protein_representation import ProteinCollection
from gp_regression import GPRegression
from data_scaler import BayesScaler
from vae import VAE
from scipy.io import loadmat
from utility import convert_graph_from_matlab_file, convert_aa_sequence
from utility import parse_mutations, preprocess_observations
from utility import WeightedMSADataset, parse_matlab_mutation_file
from parse_data import filter_alignment, parse_BLAT, parse_TLL, parse_UBQ, parse_PGA, parse_HEX
import torch
import torch.nn.functional as F


class Experiment:
    """
    Wrapper Class that encapsulates experiment configurations
    """

    def __init__(self, pdb: str, experiment_type: str, idx: int, optimization: bool,
                 fusion: bool, reference: bool, vae_input: bool, vae_kernel: bool, fraction: float,
                 exp_data_filename: str, is_data_filename: str, run_id: str, **vae_params) -> None:
        self.pdb = pdb
        self.experiment_type = experiment_type
        self.idx = idx
        self.fusion = fusion
        self.optimization = optimization
        self.fraction = fraction
        self.vae = None
        self.vae_input = vae_input
        self.vae_kernel = vae_kernel
        self.exp_data_filename = exp_data_filename
        self.is_data_filename = is_data_filename
        self.run_id = run_id
        self.two_sigma = reference
        self.contact_map = ContactMapper(pdb_file=f"./pdb/{pdb.lower()}.pdb", tri_dist=True)
        self.experimental_data = self.prepare_experimental_data()
        self.ref_adj = self.contact_map.adjacency
        if vae_input or vae_kernel:
            self.family_seqs = self.prepare_family_sequences()
            self.vae_model_FILENAME = f"./models/VAE_t{experiment_type}_z55_h[1700, 1200]_e200_d0.065_wTrue.pt"
            self.vae = self.prepare_vae(**vae_params)
        if vae_kernel:
            self.protein = ProteinCollection(self.contact_map, pdb_ID=pdb, mutations_exp=self.experimental_data,
                                             kernel_vae=self.vae)
        else:
            self.protein = ProteinCollection(self.contact_map, pdb_ID=pdb, mutations_exp=self.experimental_data)
        if idx == 0:
            print("Assertion Run:")
            self.assert_structure_to_experiment_integrity()
        self.in_silico_data = self.prepare_in_silico_data() if self.fusion else {}
        self.X_wt, self.X_exp, self.X_is, self.y_wt, self.ΔΔg_exp, self.ΔΔg_is_scaled, self.scaler_σ, self.max_y, self.mean_y = self.init_experiment_run()
        if not self.fusion:
            self.X_is = np.array([])
            self.scaler_σ = torch.Tensor([0.])
        self.gpr = self.init_mgp_regression()

    def assert_structure_to_experiment_integrity(self):
        mutations = {}
        for m, _ in self.experimental_data.get(self.protein.pdb_ID):
            try:
                location = int(str(m)[1:-1]) - 1  # adjust index as done in ProteinCollection parsing
            except ValueError as e:  # hidden wild-type or unknown mutation notation
                continue
            mutations[location] = m[0]
        for idx, s_elem in enumerate(self.contact_map.sequence):
            #print(f"S-elem: {s_elem} <=> {mutations.get(idx)}")
            if mutations.get(idx) and s_elem != mutations.get(idx):
                print(f"MISMATCH pos: {idx}")
                #assert s_elem == mutations.get(idx)

    def prepare_in_silico_data(self):
        if self.experiment_type == "blat":
            return self.prepare_blat_in_silico_data()
        elif self.experiment_type == "tll":
            return self.prepare_tll_in_silico_data()
        elif self.experiment_type == "mgpf":
            return parse_matlab_mutation_file(self.is_data_filename, query="ddg_rosetta_single")
        elif self.experiment_type == "ubq":
            return self.prepare_ubq_in_silico_data()
        elif self.experiment_type == "pga":
            return self.prepare_pga_in_silico_data()
        else:
            raise NotImplementedError("Specified In-Silico data configuration not available.")

    def prepare_experimental_data(self):
        if self.experiment_type == "blat":
            return self.prepare_blat_experimental_data()
        elif self.experiment_type == "tll":
            return self.prepare_tll_experimental_data()
        elif self.experiment_type == "mgpf":
            return parse_matlab_mutation_file(self.exp_data_filename, query="ddg_protherm")
        elif self.experiment_type == "ubq":
            return self.prepare_ubq_experimental_data()
        elif self.experiment_type == "pga":
            return self.prepare_pga_experimental_data()
        elif self.experiment_type == "hex":
            return self.prepare_hexo_experimental_data()
        else:
            raise NotImplementedError("Specified experimental data configuration not available.")

    def prepare_family_sequences(self):
        if self.experiment_type == "blat":
            family_seq, _, _ = parse_BLAT()
        elif self.experiment_type == "tll":
            family_seq, _, _ = parse_TLL()
        elif self.experiment_type == "ubq":
            family_seq, _, _ = parse_UBQ()
        elif self.experiment_type == "hex":
            family_seq, _, _ = parse_HEX()
        elif self.experiment_type == "pga":
            family_seq, _, _ = parse_PGA()
        else:
            raise NotImplementedError(f"Specified experiment {self.experiment_type} has no family sequences available.")
        return family_seq

    def prepare_blat_in_silico_data(self, load_existing=True):
        if not self.vae:
            raise NotImplementedError("Rosetta in silico data for BLAT is not yet implemented.")
        if os.path.exists(self.is_data_filename) and load_existing:
            with open(self.is_data_filename, "rb") as filehandle:
                is_mutation_dict = pickle.load(filehandle)
            return is_mutation_dict
        else:
            return self.generate_in_silico_mutations_from_vae()

    def generate_in_silico_mutations_from_vae(self, write_data=True):
        mutations_tuples = self.derive_vae_mutations()
        is_mutation_dict = {self.protein.pdb_ID.upper(): mutations_tuples}
        if write_data:
            with open(self.is_data_filename, "wb") as filehandle:
                pickle.dump(is_mutation_dict, filehandle)
        return is_mutation_dict

    def prepare_vae(self, **vae_params):
        num_classes = np.unique(self.family_seqs).shape[0] + 1
        WT = F.one_hot(torch.tensor(self.family_seqs[0], dtype=torch.int64), num_classes=num_classes).flatten().float()
        vae = VAE(z_dim=vae_params["latent_dim"], encoder_dim=[vae_params["encoder_dim"]],
                  decoder_dim=[vae_params["decoder_dim"]],
                  input_dims=WT.shape[0], use_cuda=vae_params["cuda"], wt=WT,
                  dropout=vae_params["dropout"], num_categories=num_classes)
        if os.path.exists(self.vae_model_FILENAME):
            vae.load_state_dict(torch.load(self.vae_model_FILENAME))
        else:
            raise FileNotFoundError(f"Specified model does not exist!\n {self.vae_model_FILENAME}")
        return vae

    def derive_vae_mutations(self, sample_n=1):
        assert isinstance(self.vae, VAE)
        mut_S_exp, mut_adj_exp, ΔΔg_exp, mut_ids_exp = parse_mutations(
            mutation_dict=self.experimental_data.get(self.pdb), sequence=self.protein.sequence, adjacency=self.ref_adj)
        X_exp = convert_aa_sequence(mut_S_exp)
        sequence_dataset = WeightedMSADataset(X_exp, num_classes=self.vae.num_categories)
        self.vae.eval()
        wt_log_prob = self.vae.log_p(self.vae.wt)[1].detach().numpy()
        log_likelihoods = []
        samples = []
        for seq, _, _ in sequence_dataset:
            samples.append(self.vae.latent_sample(seq.flatten(), n=sample_n).reshape(
                -1).detach().numpy())
            loss = self.vae.log_p(seq.flatten())
            log_likelihoods.append(loss[1].detach().numpy())
        delta_log_p = np.array([(l - wt_log_prob) for l in log_likelihoods], dtype=float)
        mutation_values = list(zip([m for m, _ in self.experimental_data.get(self.pdb)], delta_log_p))
        return mutation_values

    def load_blat_experimental_mutations_from_csv(self, save_file="./data/blat/blat_exp_mutations.pkl"):
        """
        !!! WARN: EXPERIMENTAL INDEX IS OFF BY 24 W.R.T. PDB SEQUENCE !!!
        """
        blat_df = pd.read_csv(self.exp_data_filename)
        blat_df["growth"] = blat_df["2500"]
        blat_df["mutation_idx"] = blat_df.mutant.str[1:-1].astype(int) - 23
        blat_df["mutant"] = blat_df.mutant.str[0] + blat_df.mutation_idx.astype(str) + blat_df.mutant.str[-1]
        mutations = list(zip(blat_df.mutant, blat_df.growth))
        mutation_dict = {"1FQG": mutations}
        if save_file:
            with open(save_file, "wb") as filehandle:
                pickle.dump(mutation_dict, filehandle)
        return mutation_dict

    def prepare_blat_experimental_data(self, load_existing=True) -> dict:
        exp_mutations_filename = "./data/blat/blat_exp_mutations.pkl"
        if os.path.exists(exp_mutations_filename) and load_existing:
            with open(exp_mutations_filename, "rb") as filehandle:
                mutation_dict = pickle.load(filehandle)
            return mutation_dict
        else:
            return self.load_blat_experimental_mutations_from_csv()

    def load_hexo_experimental_mutations_from_csv(self, save_file="./data/hex/hex_exp_mutations.pkl"):
        """
        Load experimental observations from file.
        """
        exp_df = pd.read_excel("./data/hex/Hexosaminidase_SSL_data_simple.xlsx")
        exp_df = exp_df[["Origin", "Target", "ddG (HIF)"]].dropna()
        exp_df["pdb_position"] = exp_df.Origin.str[1:].astype(int)
        exp_df['mutation_origin'] = exp_df.Origin.str[0]
        exp_df["mutations"] = exp_df.mutation_origin + exp_df.pdb_position.astype(str) + exp_df.Target
        mutation_dict = {"D45": [(mut, y) for (mut, y) in zip(exp_df.mutations, exp_df["ddG (HIF)"])]}
        if save_file:
            with open(save_file, "wb") as filehandle:
                pickle.dump(mutation_dict, filehandle)
        return mutation_dict

    def prepare_hexo_experimental_data(self, load_existing=True) -> dict:
        exp_mutations_filename = "./data/hex/hex_exp_mutations.pkl"
        # AUGMENT CONTACT MAP, MISSING FIRST Q
        self.contact_map.sequence = np.insert(self.contact_map.sequence, 0, "Q")
        self.contact_map.adjacency = [("Q", self.contact_map.adjacency[0][1])] + self.contact_map.adjacency
        if os.path.exists(exp_mutations_filename) and load_existing:
            with open(exp_mutations_filename, "rb") as filehandle:
                mutation_dict = pickle.load(filehandle)
            return mutation_dict
        else:
            return self.load_hexo_experimental_mutations_from_csv()

    def prepare_tll_in_silico_data(self):
        assert os.path.exists(self.exp_data_filename)
        assert os.path.exists(self.is_data_filename)
        tll_df = pd.read_excel(self.exp_data_filename)
        is_df = pd.read_excel(self.is_data_filename)
        is_df = is_df.merge(tll_df, how="left", left_on="var_name",
                            right_on="TSA.sample")[["mut2wt_1ein_join", "ddG"]].dropna()
        is_df["mutations"] = is_df.mut2wt_1ein_join.str.replace(" ", "")
        is_df["ddG"] = is_df.ddG.astype(float)
        is_mutations = list(zip(is_df.mutations, is_df.ddG))
        return {"1TIB": is_mutations}

    def prepare_tll_experimental_data(self):
        tll_df = pd.read_excel(self.exp_data_filename)
        # filter out more than 10 mutations
        tll_df = tll_df[tll_df.mut2wt_1ein_join.str.count(" ") <= 9].dropna()
        tll_df = tll_df[["mut2wt_1ein_join", "TSA.Tm"]]
        # mean over experimental measurements
        tll_df = tll_df.groupby("mut2wt_1ein_join").mean().reset_index()
        tll_df["mutations"] = tll_df.mut2wt_1ein_join.str.replace(" ", "")
        tll_df["TSA"] = tll_df["TSA.Tm"].astype(float)
        exp_mutations = [(mut, y) for (mut, y) in zip(tll_df.mutations, tll_df.TSA)]
        return {"1TIB": exp_mutations}

    def prepare_ubq_experimental_data(self, load_existing=True):
        exp_mutations_filename = "./data/ubq/ubq_exp_mutations.pkl"
        if os.path.exists(exp_mutations_filename) and load_existing:
            with open(exp_mutations_filename, "rb") as filehandle:
                mutation_dict = pickle.load(filehandle)
            return mutation_dict
        else:
            return self.load_ubq_experimental_data_from_csv()

    def load_ubq_experimental_data_from_csv(self, save_file="./data/ubq/ubq_exp_mutations.pkl"):
        ubq_df = pd.read_csv(self.exp_data_filename, delimiter=";")
        ubq_df = ubq_df[["mutant", "selection_coefficient"]].dropna()
        ubq_df["growth"] = ubq_df["selection_coefficient"].str.replace(",", ".").astype(float)
        mutation_dict = {"1UBQ": list(zip(ubq_df.mutant, ubq_df.growth))}
        if save_file:
            with open(save_file, "wb") as filehandle:
                pickle.dump(mutation_dict, filehandle)
        return mutation_dict

    def prepare_ubq_in_silico_data(self, load_existing=True):
        if not self.vae:
            raise NotImplementedError("Rosetta in silico data for UBQ is not implemented.")
        if os.path.exists(self.is_data_filename) and load_existing:
            with open(self.is_data_filename, "rb") as filehandle:
                is_mutation_dict = pickle.load(filehandle)
            return is_mutation_dict
        else:
            return self.generate_in_silico_mutations_from_vae()

    def prepare_pga_experimental_data(self, load_existing=True):
        exp_mutations_filename = "./data/pga/pga_exp_mutations.pkl"
        if os.path.exists(exp_mutations_filename) and load_existing:
            with open(exp_mutations_filename, "rb") as filehandle:
                mutation_dict = pickle.load(filehandle)
            return mutation_dict
        else:
            return self.load_pga_experimental_data_from_csv()

    def load_pga_experimental_data_from_csv(self, save_file="./data/pga/pga_exp_mutations.pkl"):
        test_df = pd.read_csv("./data/pga/Nisthal_Mayo_2019_updated_3xESLyS9.csv", delimiter=",")
        test_df = test_df[~test_df["Assay/Protocol"].str.contains("SD ")]  # exclude standard-deviation
        test_df = test_df[test_df.Units == "kcal/mol"]
        test_df = test_df[test_df["Assay/Protocol"].str.contains("^ddG")]  # select only ddG values
        test_df = test_df[["Description", "Data", "Assay/Protocol"]].dropna()
        mutation_dict = {"1PGA": list(zip(test_df.Description, test_df.Data))}
        if save_file:
            with open(save_file, "wb") as filehandle:
                pickle.dump(mutation_dict, filehandle)
        return mutation_dict

    def prepare_pga_in_silico_data(self, load_existing=True):
        if not self.vae:
            raise NotImplementedError("Rosetta in silico data for PGA is not implemented.")
        if os.path.exists(self.is_data_filename) and load_existing:
            with open(self.is_data_filename, "rb") as filehandle:
                is_mutation_dict = pickle.load(filehandle)
            return is_mutation_dict
        else:
            return self.generate_in_silico_mutations_from_vae()

    @staticmethod
    def prepare_tll_family_sequences():
        with open("./data/tll/seqs_in_int_nogaps_sp400_Mar14_data_all_jaks_Apr3_trimmed.pkl", "rb") as infile:
            family_seqs = np.array(pickle.load(infile))
        return family_seqs

    @staticmethod
    def prepare_ubq_family_sequences():
        ubq_df = filter_alignment("./data/ubq/UBQ_combined_UBC_ISG15.a2m")
        family_seqs = np.array([[int(elem) for elem in seq] for seq in ubq_df.seq])
        return family_seqs

    @staticmethod
    def prepare_pga_family_sequences():
        pga_df = filter_alignment("./data/pga/FINAL_PGA_n1133.a3m")
        family_seqs = np.array([s for s in pga_df.seq])
        return family_seqs

    def init_mgp_regression(self):
        gpr = GPRegression(protein_representation=self.protein, X_wt=self.X_wt, X_exp=self.X_exp, X_is=self.X_is,
                           y_wt=self.y_wt, y_exp=self.ΔΔg_exp, y_is=self.ΔΔg_is_scaled, adjacencies=self.ref_adj,
                           σ_T=self.scaler_σ, y_max=self.max_y, y_mean=self.mean_y, cached=True, fusion=self.fusion,
                           kernel_vae=self.vae_kernel)
        # TODO fix scaler sigma (to array of sigmas)
        return gpr

    def init_experiment_run(self, load_reference_adjaciencies=True) -> tuple:
        assert self.experimental_data and bool(self.fusion == bool(self.in_silico_data))
        ref_mat_file = os.path.join(os.path.dirname(__file__), os.path.join("data/mgp/", f"{self.pdb.upper()}.mat"))
        if os.path.isfile(ref_mat_file) and load_reference_adjaciencies:
            pga_file = loadmat(ref_mat_file)
            self.ref_adj = convert_graph_from_matlab_file(
                pga_file["contact_map"])  # in case precalculated contacts exist
            self.contact_map.adjacency = self.ref_adj  # propagate contactmap to all dependencies
        mut_S_exp, _, ΔΔg_exp, mut_ids_exp = parse_mutations(mutation_dict=self.experimental_data.get(self.pdb),
                                                             sequence=self.protein.sequence,
                                                             adjacency=self.ref_adj)
        mut_S_is, _, ΔΔg_is, mut_ids_is = parse_mutations(mutation_dict=self.in_silico_data.get(self.pdb),
                                                          sequence=self.protein.sequence,
                                                          adjacency=self.ref_adj)
        X_exp, X_is = convert_aa_sequence(mut_S_exp), convert_aa_sequence(mut_S_is)
        y_wt = np.array([0])[:, np.newaxis]
        X_wt = convert_aa_sequence([self.protein.sequence])
        # scale using Bayesian Regression
        if self.fusion:
            bs_rosetta = BayesScaler(is_mutations=mut_ids_is, ΔΔg=ΔΔg_is, exp_mutations=mut_ids_exp,
                                     experimentally_observed_ΔΔg=ΔΔg_exp, TESTING=False, pdb_ID=self.pdb, cached=True,
                                     vae=self.vae_input)
            bs_rosetta.plot_scaling()

        ΔΔg_is_scaled = bs_rosetta.transform(ΔΔg_is)[:, np.newaxis] if self.fusion else ΔΔg_is[:, np.newaxis]
        sigma_T = bs_rosetta.σ_T if self.fusion else torch.Tensor([0.])
        ΔΔg_exp = ΔΔg_exp[:, np.newaxis]

        # Scale y-values as done in the implementation by normalizing with mean and max
        mean_y, max_y, y_wt, ΔΔg_exp, ΔΔg_is_scaled = preprocess_observations(y_wt, ΔΔg_exp, ΔΔg_is_scaled)
        # TODO replace bs_rosetta.σ_T with σ_T_samples
        return X_wt, X_exp, X_is, y_wt, ΔΔg_exp, ΔΔg_is_scaled, sigma_T, max_y, mean_y

    def __str__(self):
        experiment_str = f"Experiment: {self.pdb}, {self.experiment_type} ({self.idx}) "
        experiment_str += "optimized " if self.optimization else ""
        experiment_str += "vae " if self.vae_input else ""
        experiment_str += "fusion " if self.fusion else ""
        experiment_str += "VAE-S " if self.vae_kernel else "S-matrices "
        experiment_str += f"\n Data: {self.exp_data_filename} \n {self.is_data_filename}"
        return experiment_str
