from collections import OrderedDict
import torch


### ONLY FOR REFERENCE PURPOSES

IUPAC_IDX_AMINO_PAIRS_decoding_REFERENCE = list(enumerate([
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "-",
    'B'
]))

IUPAC_IDX_AMINO_PAIRS_REFERENCE = list(enumerate([
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "<mask>",
    'B'
]))
IUPAC_AMINO_IDX_PAIRS = [(a, i) for (i, a) in IUPAC_IDX_AMINO_PAIRS_REFERENCE]

alphabet_size = len(IUPAC_AMINO_IDX_PAIRS)

IUPAC_SEQ2IDX = OrderedDict(IUPAC_AMINO_IDX_PAIRS)
IUPAC_IDX2SEQ = OrderedDict(IUPAC_IDX_AMINO_PAIRS_REFERENCE)

# Add gap tokens as the same as mask
IUPAC_SEQ2IDX["-"] = IUPAC_SEQ2IDX["<mask>"]
IUPAC_SEQ2IDX["."] = IUPAC_SEQ2IDX["<mask>"]

IUPAC_IDX2SEQ_decoding = OrderedDict(IUPAC_IDX_AMINO_PAIRS_decoding_REFERENCE)

# !! WARNING: THIS IS FOR REFERENCE PURPOSES ONLY - THIS IS HOW THE BLAT DF WAS ENCODED - NOTE THAT FOR mGPf RUNS THE
# ORDER OF THE RESIDUES MATTERS !!

# def seq2idx(seq, device = None):
#     return [IUPAC_SEQ2IDX[s.upper() if len(s) < 2 else s] for s in seq]


def idx2seq(seq):
    return [IUPAC_IDX2SEQ[s] for s in seq]
