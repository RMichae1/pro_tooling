from collections import OrderedDict
import torch

IUPAC_IDX_AMINO_PAIRS_decoding = list(enumerate([
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

IUPAC_IDX_AMINO_PAIRS = list(enumerate([
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
IUPAC_AMINO_IDX_PAIRS = [(a, i) for (i, a) in IUPAC_IDX_AMINO_PAIRS]

alphabet_size = len(IUPAC_AMINO_IDX_PAIRS)

IUPAC_SEQ2IDX = OrderedDict(IUPAC_AMINO_IDX_PAIRS)
IUPAC_IDX2SEQ = OrderedDict(IUPAC_IDX_AMINO_PAIRS)

# Add gap tokens as the same as mask
IUPAC_SEQ2IDX["-"] = IUPAC_SEQ2IDX["<mask>"]
IUPAC_SEQ2IDX["."] = IUPAC_SEQ2IDX["<mask>"]

IUPAC_IDX2SEQ_decoding = OrderedDict(IUPAC_IDX_AMINO_PAIRS_decoding)

def seq2idx(seq, device = None):
    return torch.tensor([IUPAC_SEQ2IDX[s.upper() if len(s) < 2 else s] for s in seq], device = device)
