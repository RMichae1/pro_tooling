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
from utility import filter_alignment
import torch
import torch.nn.functional as F


class Experiment:
    """
    Wrapper Class that encapsulates experiment configurations
    """

    def __init__(self, pdb: str, experiment_type: str, idx: int, optimization: bool,
                 fusion: bool, reference: bool, vae_input: bool, vae_kernel: bool,
                 exp_data_filename: str, is_data_filename: str, run_id: str, **vae_params) -> None:
        self.pdb = pdb
        self.experiment_type = experiment_type
        self.idx = idx
        self.fusion = fusion
        self.optimization = optimization
        self.vae = None
        self.vae_input = vae_input
        self.vae_kernel = vae_kernel
        self.exp_data_filename = exp_data_filename
        self.is_data_filename = is_data_filename
        self.run_id = run_id
        self.two_sigma = reference
        self.experimental_data = self.prepare_experimental_data()
        self.contact_map = ContactMapper(pdb_file=f"./pdb/{pdb.lower()}.pdb", tri_dist=True)
        self.ref_adj = self.contact_map.adjacency
        if vae_input or vae_kernel:
            self.family_seqs = self.prepare_family_sequences()
            self.vae_model_FILENAME = f"./models/VAE_t{experiment_type}_z55_h[1700, 1200]_e200_d0.065_wTrue.pt"
            self.vae = self.prepare_vae(vae_params)
        self.protein = ProteinCollection(self.contact_map, pdb_ID=pdb,
                                         mutations_exp=self.experimental_data, vae=self.vae, TESTING=False)
        self.in_silico_data = self.prepare_in_silico_data() if self.fusion else {}
        self.X_wt, self.X_exp, self.X_is, self.y_wt, self.ΔΔg_exp, self.ΔΔg_is_scaled, self.scaler_σ, self.max_y, self.mean_y = self.init_experiment_run()
        if not self.fusion:
            self.X_is = np.array([])
            self.scaler_σ = torch.Tensor([0.])
        self.gpr = self.init_mgp_regression()

    def prepare_in_silico_data(self):
        if self.experiment_type == "blat":
            return self.prepare_blat_in_silico_data()
        elif self.experiment_type == "tll":
            return self.prepare_tll_in_silico_data()
        elif self.experiment_type == "mgpf":
            return parse_matlab_mutation_file(self.is_data_filename, query="ddg_rosetta_single")
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
        else:
            raise NotImplementedError("Specified experimental data configuration not available.")

    def prepare_family_sequences(self):
        if self.experiment_type == "blat":
            return self.prepare_blat_family_sequences()
        elif self.experiment_type == "tll":
            return self.prepare_tll_family_sequences()
        elif self.experiment_type == "ubq":
            return self.prepare_ubq_family_sequences()
        else:
            raise NotImplementedError(f"Specified experiment {self.experiment_type} has no family sequences available.")

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
        num_classes = np.unique(self.family_seqs).shape[0]
        WT = F.one_hot(torch.tensor(self.family_seqs[0], dtype=torch.int64),
                       num_classes=num_classes).flatten().float()
        vae = VAE(z_dim=vae_params["latent_dim"], encoder_dim=list(vae_params["encoder_dim"]),
                  decoder_dim=list(vae_params["decoder_dim"]),
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
            mutation_dict=self.experimental_data.get(self.pdb),
            sequence=self.protein.sequence,
            adjacency=self.ref_adj)
        X_exp = convert_aa_sequence(mut_S_exp)  # TODO also run seq2idx and test for identity
        sequence_dataset = WeightedMSADataset(X_exp, num_classes=self.vae.num_categories)
        self.vae.eval()
        wt_log_prob = self.vae.log_p(self.vae.wt)[1].detach().numpy()
        log_likelihoods = []
        samples = []
        for seq, _, _ in sequence_dataset:
            samples.append(self.vae.latent_sample(seq.flatten(), n=sample_n).reshape(
                -1).detach().numpy())  # TODO why is there a sequence when sampling??
            loss = self.vae.log_p(seq.flatten())
            log_likelihoods.append(loss[1].detach().numpy())
        delta_log_p = np.array([(l - wt_log_prob) for l in log_likelihoods], dtype=float)
        mutation_values = list(zip([m for m, _ in self.experimental_data.get(self.pdb)], delta_log_p))
        return mutation_values

    def load_blat_experimental_mutations_from_csv(self, save_file="./data/blat/blat_exp_mutations.pkl"):
        blat_df = pd.read_csv(self.exp_data_filename)
        blat_df["growth"] = blat_df["2500"]
        clipped_mutations = list(filter(lambda x: int(x[0][1:-1]) <= 263,
                                        zip(blat_df.mutant, blat_df.growth)))
        # WARNING: we clip mutations at position 263 - mutations go until 286, however pdb is only 263 (A chain) long
        mutation_dict = {"1FQG": clipped_mutations}
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

    def prepare_tll_in_silico_data(self):
        # TODO load rosetta TLL data here
        assert os.path.exists(self.exp_data_filename)
        assert os.path.exists(self.is_data_filename)
        tll_df = pd.read_excel(self.exp_data_filename)
        is_df = pd.read_excel(self.is_data_filename)
        is_df = is_df.merge(tll_df, how="left", left_on="var_name",
                            right_on="TSA.sample")[["mut2wt_1ein_join", "ddG"]].dropna()
        is_df["mutations"] = is_df.mut2wt_1ein_join.str.replace(" ", "")
        is_df["ddG"] = is_df.ddG.astype(float)
        is_mutations = [(mut, y) for (mut, y) in zip(is_df.mutations, is_df.ddG)]
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
        clipped_mutations = list(filter(lambda x: int(x[0][1:-1]) <= 74, zip(ubq_df.mutant, ubq_df.growth)))
        # WARNING: we clip mutations at position 74 - mutations go until 76, however pdb is only 74 (A chain) long
        mutation_dict = {"1UBQ": clipped_mutations}
        if save_file:
            with open(save_file, "wb") as filehandle:
                pickle.dump(mutation_dict, filehandle)
        return mutation_dict

    @staticmethod
    def prepare_blat_family_sequences():
        with open("./data/blat/BLAT_data_df.pkl", "rb") as infile:
            blat_df = pickle.load(infile)
        family_df = blat_df[blat_df.assay.isna()]
        family_seqs = np.array([[int(elem) for elem in seq] for seq in family_df.seqs])
        return family_seqs

    @staticmethod
    def prepare_tll_family_sequences():
        with open("./data/tll/seqs_in_int_nogaps_sp400_Mar14_data_all_jaks_Apr3_trimmed.pkl", "rb") as infile:
            family_seqs = np.array(pickle.load(infile))
        return family_seqs

    @staticmethod
    def prepare_ubq_family_sequences():
        ubq_df = filter_alignment("./data/ubq/UBC_HUMAN_P0CG48_ubiquitin.a2m")
        family_seqs = np.array([[int(elem) for elem in seq] for seq in ubq_df.seq])
        return family_seqs

    @staticmethod
    def prepare_pga_family_sequences():
        pga_df = filter_alignment("./data/pga/hmmer_PGA_msa_n42.a3m")
        family_seqs = np.array([s for s in pga_df.seq])
        return family_seqs

    def init_mgp_regression(self):
        gpr = GPRegression(protein_representation=self.protein, X_wt=self.X_wt, X_exp=self.X_exp, X_is=self.X_is,
                           y_wt=self.y_wt, y_exp=self.ΔΔg_exp, y_is=self.ΔΔg_is_scaled, adjacencies=self.ref_adj,
                           σ_T=self.scaler_σ, y_max=self.max_y, y_mean=self.mean_y, cached=True, fusion=self.fusion)
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
        # scale using Bayesian Scaling
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
