import numpy as np
import pandas as pd
from protein_representation import ProteinCollection
from contact_mapper import ContactMapper
from utility import seq2idx
import pickle
from utility import convert_aa_sequence



def parse_alignment(a2m_filename: str, drop_lowercase=True) -> pd.DataFrame:
    with open(a2m_filename, "r") as filehandle:
        alignment = filehandle.read().splitlines()
    identifier = []
    sequence = []
    seq = []
    for line in alignment:
        if line.startswith(">"):
            sequence.append(seq)
            identifier.append(line[1:])
            seq = []
        else:
            seq.append(line)
    sequence.append(seq)  # add last
    sequence = sequence[1:]  # eliminate first empty entry
    wt_seq = ''.join(sequence[0])
    uppercase_idx = [idx for idx in range(len(wt_seq)) if wt_seq[idx].isupper()]
    # convert to string and encode
    encoded_sequence = list(map(seq2idx, map(lambda x: "".join(x), sequence)))
    if drop_lowercase:
        encoded_sequence = np.array([np.array(s) for s in encoded_sequence])
        encoded_sequence = list(encoded_sequence[:, uppercase_idx])
    df = pd.DataFrame({"seq": encoded_sequence, "identifier": identifier})
    return df


def filter_alignment(a2m_filename: str, gap_code=22, wt_idx=0) -> pd.DataFrame:
    """
    Returns matrix of sequences by filtering gaps for the 
    """
    alignment_df = parse_alignment(a2m_filename)
    seqs = np.array([np.array(s) for s in alignment_df.seq])
    # select wildtype against which we select
    wt_sequence = seqs[wt_idx]
    # cut gaps
    ungapped_idx = np.argwhere(wt_sequence != gap_code).flatten()
    filtered_sequences = seqs[:, ungapped_idx]
    alignment_df["seq"] = list(filtered_sequences)
    return alignment_df


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


def parse_HEX():
    """
    !!! PROVIDED PDB AGAINST WT Sequence is off by the first position !!!
    """
    test_df = pd.read_excel("./data/hex/Hexosaminidase_SSL_data_simple.xlsx")
    test_df = test_df[["Origin", "Target", "ddG (HIF)"]].dropna()
    # ADJUSTMENT BY ONE POSITION - FIX W.R.T. PDB PARSING
    test_df["pdb_position"] = test_df.Origin.str[1:].astype(int) - 1
    test_df['mutation_origin'] = test_df.Origin.str[0]
    test_df["mutations"] = test_df.mutation_origin + test_df.pdb_position.astype(str) + test_df.Target
    exp_mutations = {"D45": [(mut, y) for (mut, y) in zip(test_df.mutations, test_df["ddG (HIF)"])]}
    contact_map = ContactMapper(pdb_file="./pdb/d45.pdb", tri_dist=True)
    protein = ProteinCollection(contact_map, pdb_ID="D45", mutations_exp=exp_mutations, TESTING=False)
    test_seqs = convert_aa_sequence(protein.mut_S_exp)
    test_y = test_df["ddG (HIF)"]
    family_seqs = np.loadtxt("./data/hex/uniref90_MSA_.aln", dtype=str)
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
    ubq_df = filter_alignment("/home/rimichael/pro_tooling/data/ubq/UBC_HUMAN_P0CG48_ubiquitin.a2m")
    family_seqs = np.array([[int(elem) for elem in seq] for seq in ubq_df.seq])
    # for testing combine protabank sequences with DeepSequence Bolon 2013 data
    protabank_df = pd.read_csv("/home/rimichael/pro_tooling/data/ubq/RL401_Bolon2013_YHUnpqbw.csv", delimiter=",")
    # drop SD values
    protabank_df = protabank_df[~protabank_df["Assay/Protocol"].str.contains("SD ")]
    # drop last two elements from sequence "...GG" and duplicate last residues
    protabank_df["Sequence"] = protabank_df.Sequence
    # measurements as used in DeepSequence paper
    deep_seq_df = pd.read_csv("/home/rimichael/pro_tooling/data/ubq/RL401_Bolon2013.csv", delimiter=";")
    deep_seq_df = deep_seq_df[["mutant", "selection_coefficient"]].dropna()
    test_df = deep_seq_df.merge(protabank_df[["Description", "Data", "Sequence"]],
                                "inner", left_on="mutant", right_on="Description")
    test_df["Data"] = test_df.Data.astype(float)
    test_df["selection_coefficient"] = test_df.selection_coefficient.str.replace(",", ".").astype(float)
    #np.testing.assert_array_equal(test_df.selection_coefficient.values, test_df.Data.values)
    test_seqs = np.array([seq2idx(seq) for seq in test_df.Sequence])
    test_y = test_df.selection_coefficient  # use DeepSequence reported values
    return family_seqs, test_seqs, test_y
