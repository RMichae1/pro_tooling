from vae import VAE
import pickle
import os
from utility import aa2index, one_hot_encoding
from vae import train, evaluate
import pyro
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam, ClippedAdam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from utility import compute_ρ
from utility import WeightedMSADataset, seq_collate 
from utility import parse_mutations, parse_alignment, filter_alignment
from utility import convert_aa_sequence
from protein_representation import ProteinCollection
from contact_mapper import ContactMapper
from graphkernel import VaeKernel
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import warnings
import argparse
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
import random
from reference_alphabet import seq2idx

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # TODO figure out what caused OMP Error #15

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

VAE_TYPES = ["blat", "sp400", "pga", "ubq"]


def parse_BLAT():
    with open("./data/blat/BLAT_data_df.pkl", "rb") as infile:
            blat_df = pickle.load(infile)
    # stored values without assay entries are BLAT TEM1 ECOLX family data
    family_df = blat_df[blat_df.assay.isna()]
    test_blat_df = blat_df[~blat_df.assay.isna()]
    # cast sequence labels to int
    family_seqs = np.array([[int(elem) for elem in seq] for seq in family_df.seqs])
    test_seqs = np.array([[int(elem) for elem in seq] for seq in test_blat_df.seqs])
    test_y = np.array(test_blat_df.assay, dtype=float)
    return family_seqs, test_seqs, test_y


def parse_TLL():
    with open("./data/tll/seqs_in_int_nogaps_sp400_Mar14_data_all_jaks_Apr3_trimmed.pkl", "rb") as infile:
        family_seqs = np.array(pickle.load(infile))
    test_df = pd.read_excel("./data/tll/lipase_variants_tll_tm_tapo_20nov2020.xlsx")
    test_df = test_df[["mut2wt_1ein_join", "TSA.Tm"]]
    test_df = test_df.groupby("mut2wt_1ein_join").mean().reset_index()
    test_df["mutations"] = test_df.mut2wt_1ein_join.str.replace(" ", "")
    test_df["TSA"] = test_df["TSA.Tm"].astype(float)
    exp_mutations = {"1TIB" : [(mut, y) for (mut, y) in zip(test_df.mutations, test_df.TSA)]}
    contact_map = ContactMapper(pdb_file=f"./pdb/1tib.pdb", tri_dist=True)
    protein = ProteinCollection(contact_map, pdb_ID="1TIB", mutations_exp=exp_mutations, TESTING=False)
    test_seqs = convert_aa_sequence(protein.mut_S_exp) # TODO also run seq2idx and test for identity
    test_y = test_df.TSA  # get y values
    return family_seqs, test_seqs, test_y


def parse_PGA():
    test_df = pd.read_csv("./data/pga/Nisthal_Mayo_2019_updated_3xESLyS9.csv", delimiter=",")
    test_df = test_df[~test_df["Assay/Protocol"].str.contains("SD ")]  # exclude standard-deviation
    test_df = test_df[test_df.Units == "kcal/mol"]
    test_df = test_df[test_df["Assay/Protocol"].str.contains("^ddG")]  # select only ddG values
    test_df = test_df[["Sequence", "Data", "Assay/Protocol"]].dropna()  # select relevant columns
    pga_df = filter_alignment("./data/pga/hmmer_PGA_msa_n42.a3m")
    family_seqs = np.array([s for s in pga_df.seq])
    # build sequences from test_df
    test_seqs = np.array([seq2idx(seq) for seq in test_df.Sequence])
    test_y =test_df.Data.astype(float)
    return family_seqs, test_seqs, test_y


def parse_UBQ():
    ubq_df = filter_alignment("./data/ubq/UBC_HUMAN_P0CG48_ubiquitin.a2m")
    family_seqs = np.array([[int(elem) for elem in seq] for seq in ubq_df.seq])
    # for testing combine protabank sequences with DeepSequence Bolon 2013 data
    protabank_df = pd.read_csv("./data/ubq/RL401_Bolon2013_YHUnpqbw.csv", delimiter=",")
    # drop SD values
    protabank_df = protabank_df[~protabank_df["Assay/Protocol"].str.contains("SD ")]
    # drop last two elements from sequence "...GG" and duplicate last residues
    protabank_df["Sequence"] = protabank_df.Sequence
    # measurements as used in DeepSequence paper
    deep_seq_df = pd.read_csv("./data/ubq/RL401_Bolon2013.csv", delimiter=";")
    deep_seq_df = deep_seq_df[["mutant", "selection_coefficient"]].dropna()
    test_df = deep_seq_df.merge(protabank_df[["Description", "Data", "Sequence"]], 
                                "inner", left_on="mutant", right_on="Description")
    test_df["Data"] = test_df.Data.astype(float)
    test_df["selection_coefficient"] = test_df.selection_coefficient.str.replace(",", ".").astype(float)
    #np.testing.assert_array_equal(test_df.selection_coefficient.values, test_df.Data.values)
    test_seqs = np.array([seq2idx(seq) for seq in test_df.Sequence])
    test_y = test_df.selection_coefficient  # use DeepSequence reported values
    return family_seqs, test_seqs, test_y


if __name__ == "__main__":
    pyro.clear_param_store()
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="VAE Module - train and run VAE.")
    parser.add_argument("-lr", "--learn_rate", type=float, default=0.000027, help="learning rate for optimizer")
    parser.add_argument("--cuda", action="store_true", help="Boolean flag to use cuda.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbosity boolean.")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed for reproducability.")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--latent_dim", type=int, default=2, help="Dimensionality of hidden latent random variable.")
    parser.add_argument("-s", "--save", type=str, help="Destination for models output.")
    parser.add_argument("--encoder_dim", nargs="+", type=int, default=[1700],
                        help="Hidden dimension(s) for VAE encoder module.")
    parser.add_argument("--decoder_dim", nargs="+", type=int, default=[1200],
                        help="Hidden dimension(s) for the VAE decoder module.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction of test data from total data-set.")
    parser.add_argument("--validate", type=int, default=10, help="Frequency of validation step.")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Int size of batches.")
    parser.add_argument("--experiment", type=str, help="experiment str as ID for tracking.")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0007, help="Adam Optimizer weight decay.")
    parser.add_argument("-d", "--dropout", type=float, default=0.065, help="Add Dropout layer with dropout probability.")
    parser.add_argument("-sw", "--sequence_weighting", action="store_false", # TODO reverse action
                        help="Weighing input sequences in the training procedure.")
    parser.add_argument("-t", "--type", choices=VAE_TYPES, default="ubq", help="Type ID of MSA used to create VAE.")
    parser.add_argument("-p", "--plot", action="store_false", help="Plot low-latent-representation outputs and feature correlation.")
    parser.add_argument("--sample_vae", action="store_true", help="Prepare in-silico sample.")
    args = parser.parse_args()  # TODO change weighting to store_true

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.type == "blat":
        family_seqs, test_seqs, test_y = parse_BLAT()
    elif args.type == "sp400":
        family_seqs, test_seqs, test_y = parse_TLL()   # TODO
    elif args.type == "pga":
        family_seqs, test_seqs, test_y = parse_PGA()
    elif args.type == "ubq":
        family_seqs, test_seqs, test_y = parse_UBQ()
    else:
        raise NotImplementedError(
            "Specified type not implemented. Please pick a VAE from the list of options. See help -h.")

    n, length = family_seqs.shape
    test_n = test_seqs.shape[0]
    num_classes = np.unique(family_seqs).shape[0] + 2  # TODO double check this.. PGA has 20 classes Error
    indices = list(range(n))
    random.shuffle(indices)
    test_size = int(args.test_split * n)
    train_idx = indices[:(n - test_size)]
    test_idx = indices[(n - test_size):]

    # load and encode data set
    seq_train = WeightedMSADataset(family_seqs[train_idx], num_classes=num_classes)
    seq_test = WeightedMSADataset(family_seqs[test_idx], num_classes=num_classes)
    sampler = torch.utils.data.WeightedRandomSampler(seq_train.weights, num_samples=len(seq_train),
                                                     replacement=True) if args.sequence_weighting else None
    test_seq_dataset = WeightedMSADataset(test_seqs, num_classes=num_classes)

    train_loader = torch.utils.data.DataLoader(seq_train, batch_size=args.batch_size, sampler=sampler,
                                               collate_fn=seq_collate)
    test_loader = torch.utils.data.DataLoader(seq_test, batch_size=args.batch_size, shuffle=True,
                                              collate_fn=seq_collate)

    WT = F.one_hot(torch.tensor(family_seqs[0], dtype=torch.int64),
                   num_classes=num_classes).flatten().float()
    print(WT.shape)

    # parameters
    param_dict = {"LEARNING_RATE": args.learn_rate,
                  "USE_CUDA": args.cuda,
                  "NUM_EPOCHS": args.epochs,
                  "TEST_FREQ": args.validate,
                  "LATENT_DIM": args.latent_dim,
                  "ENCODER_DIM": args.encoder_dim,
                  "DECODER_DIM": args.decoder_dim,
                  "INPUT_DIM": WT.shape[0],
                  "test_split": args.test_split,
                  "batch_size": args.batch_size,
                  "weight_decay": args.weight_decay,
                  "dropout": args.dropout,
                  "sequence_weighting": args.sequence_weighting,
                  "seed": args.seed}

    # TODO VAE of different flavors - sparse, dropout, etc.
    experiment_name = args.experiment if args.experiment else f"VAE_Adam_z{args.latent_dim}_t{args.type}"
    mlflow.set_experiment(experiment_name)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    model_FILENAME = f"./models/VAE_t{args.type}_z{args.latent_dim}_h{args.encoder_dim + args.decoder_dim}_e{args.epochs}_d{args.dropout}_w{args.sequence_weighting}.pt"
    optimizer_FILENAME = f"./models/Adam_t{args.type}_z{args.latent_dim}_h{args.encoder_dim + args.decoder_dim}_e{args.epochs}_d{args.dropout}_w{args.sequence_weighting}.pt"
    vae = VAE(z_dim=param_dict["LATENT_DIM"], encoder_dim=param_dict["ENCODER_DIM"],
              decoder_dim=param_dict["DECODER_DIM"],
              input_dims=param_dict["INPUT_DIM"], use_cuda=args.cuda, wt=WT, dropout=param_dict["dropout"],
              num_categories=num_classes)
    optimizer = Adam({"lr": param_dict["LEARNING_RATE"], "weight_decay": param_dict["weight_decay"]})
    svi = SVI(vae.model, vae.guide, optimizer, loss=JitTrace_ELBO())

    if os.path.exists(model_FILENAME) and os.path.exists(optimizer_FILENAME):
        vae.load_state_dict(torch.load(model_FILENAME))
        optimizer.load(optimizer_FILENAME)
    else:
        mlflow.start_run()
        mlflow.log_params(param_dict)
        mlflow.set_tag("out", "Categorical")
        vae.train()
        torch.autograd.set_detect_anomaly(True)
        for epoch in tqdm(range(args.epochs)):
            total_epoch_loss_train = train(svi, train_loader, args.cuda)
            mlflow.log_metric(key="neg loss train", value=-total_epoch_loss_train,
                              step=epoch)
            print(f"[epoch {epoch}] avrg. train loss: {total_epoch_loss_train}")

            if epoch % args.validate == 0:
                total_epoch_loss_test = evaluate(svi, test_loader, args.cuda)
                mlflow.log_metric(key="neg loss validate", value=-total_epoch_loss_test,
                                  step=epoch)
                print(f"[epoch {epoch}] avrg. test loss: {total_epoch_loss_test}")

        torch.save(vae.state_dict(), model_FILENAME)
        optimizer.save(optimizer_FILENAME)
        mlflow.log_artifact(model_FILENAME)
        mlflow.log_artifact(optimizer_FILENAME)
        mlflow.end_run()

    vae.eval()
    wt_log_prob = vae.log_p(WT)[1].detach().numpy()
    wt_elbo = vae.log_p(WT)[0].detach().numpy()
    elbo_values = []
    kld_values = []
    log_likelihoods = []
    samples = []
    for s, _, _ in test_seq_dataset:
        samples.append(vae.latent_sample(s.flatten(), n=1).reshape(-1).detach().numpy())
        loss = vae.log_p(s.flatten())
        elbo_values.append(loss[0].detach().numpy())
        log_likelihoods.append(loss[1].detach().numpy())
        kld_values.append(loss[2].detach().numpy())

    delta_log_p = np.array([(l - wt_log_prob) for l in log_likelihoods], dtype=float)
    print(f"Corr. (Spearman) Δ ELBO and data: {spearmanr(delta_log_p, test_y)}")

    ## TEST KERNEL
    contact_map = ContactMapper(pdb_file=f"./pdb/1ubq.pdb", tri_dist=True)
    ref_adj = contact_map.adjacency
    v_k = VaeKernel(vae)
    print(v_k.k(family_seqs, adjacencies=ref_adj))

    if args.plot:
        samples = [vae.latent_sample(s.flatten(), n=1).reshape(-1).detach().numpy() for s, _, _ in seq_train]
        samples = np.array(samples)
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.25, s=1.5)
        plt.title(f"VAE z={args.latent_dim} latent representation of training data in 2D \n {args.type}")
        plt.savefig(f"./fig/vae_z{args.latent_dim}_2d_{args.type}.png")
        plt.show()
        
        fig, ax = plt.subplots(1, 1)
        sns.regplot(delta_log_p, test_y, ax=ax, color="grey", 
                    scatter_kws={"alpha": 0.125}, line_kws={"color": "darkred"})
        ax.set_ylabel("measured effect")
        ax.set_xlabel("delta log likelihood")
        plt.suptitle("VAE loss to measured values")
        plt.savefig(f"./fig/vae_z{args.latent_dim}_correlation_{args.type}.png")
        plt.show()
    
    # if args.sample_vae:
    #     # TODO REDO THIS 
    #     data_filename = "./data/blat/BLAT_ECOLX_Ranganathan2015.csv"
    #     data_df = pd.read_csv(data_filename)
    #     #assert len(data_df) == len(test_blat_df)
    #     mutations = list(filter(lambda x: int(x[1:-1]) <= 263, data_df.mutant))
    #     mutations_tuples = list(zip([m for m in mutations], delta_log_p))
    #     is_mutations_dict = {"1FQG": mutations_tuples}
    #     with open("./data/blat/vae_blat_mutations.pkl", "wb") as outfile:
    #         pickle.dump(is_mutations_dict, outfile)
    

    
