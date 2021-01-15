import os
import pytest
import numpy as np
from utility import convert_graph_from_matlab_file, get_sequence_and_contact_graph
from utility import parse_mutations, parse_matlab_mutation_file, convert_aa_sequence
from graphkernel import MatrixKernel
from scipy.io import loadmat
from protein_representation import ProteinCollection
from contact_mapper import ContactMapper
from utility import get_split_training_and_test_data

cm = ContactMapper(pdb_file="./pdb/1pga.pdb", tri_dist=True)
mut_exp = parse_matlab_mutation_file("./data/ddg_protherm.mat", query="ddg_protherm")
mut_is = parse_matlab_mutation_file("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")

ref_file = loadmat(os.path.join(os.path.dirname(__file__), os.path.join("data", "1PGAkernel_matrices.mat")))
matrices = ref_file["subMats"]
m = matrices[0]
kernel = MatrixKernel(matrix=m[0], matrix_id=None)


def naive_K(seq: np.ndarray, adj: np.ndarray, S:np.ndarray) -> np.ndarray:
    """
    Kernel as described in the paper
    """
    n = seq.shape[0]
    K = np.zeros([n, n])
    for p in range(n):
        for q in range(n):
            for idx in range(seq.shape[1]):
                nbps = adj[idx]
                for l in nbps:
                    K[p, q] += S[seq[p, l], seq[q, l]]
                K[p, q] *= S[seq[p, idx], seq[q, idx]]
    print(K)
    # normalize
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            K[p, q] /= (np.sqrt(K[p, p]) * np.sqrt(K[q, q]))
    # set diagonal explicitly 
    for i in range(0, n):
        K[i, i] = 1
    return K


def test_vectorized_kernel():
    N = 50 # mutations
    L = 20 # sequence length
    AA = 19 # amino acids
    S = kernel.matrix
    seqs = np.random.randint(0, AA, size=[N, L])
    adj = [np.random.randint(0, L, [np.random.randint(0, L)]) for _ in range(0, L)]
    k = kernel.k(sequences=seqs, adjacencies=adj)
    k_ref = naive_K(seq=seqs, adj=adj, S=S)
    np.testing.assert_almost_equal(k, k_ref)

ref_file = loadmat(os.path.join(os.path.dirname(__file__), os.path.join("data", "1PGAkernel_matrices.mat")))
ref_K_list = ref_file["kernel_matrices"]
matrices = ref_file["subMats"]
ref_contact_graph = convert_graph_from_matlab_file(ref_file["al"])
num_wet_lab_obs = ref_K_list[0][0].shape[0] - 1

sequence_WT = get_sequence_and_contact_graph(pdb_id="1PGA", cutoff_distance=5., chain_id=None)[0]
sequence_WT = list(sequence_WT)

prot = ProteinCollection(cm, pdb_ID="1PGA", mutations_exp=mut_exp, mutations_sim=mut_is)

def test_parsed_seq_against_ref():
        """
        is reference sequence equal to own parsed sequence
        """
        assert np.all([x == y for x,y in zip(prot.sequence, sequence_WT)])

def test_adjacency_against_ref():
        # TEST adjacencies
        # THIS FAILS
        contacts = np.array([contacts for res, contacts in cm.adjacency])
        assert len(ref_contact_graph) == len(contacts)
        # assert np.all([elem_ref == elem for elem_ref, elem in zip(ref_contact_graph, contacts)])

### PARSING MUTATIONS
# Richard Code:
mut_S_exp, _, _, _ = parse_mutations(mutation_dict=mut_exp.get(prot.pdb_ID),
                                            sequence=sequence_WT, adjacency=ref_contact_graph)
X = np.vstack([sequence_WT, mut_S_exp])
X = convert_aa_sequence(X)
# Simon Code:
num_wet_lab_obs = ref_K_list[0][0].shape[0] - 1
_, x_wild_type, _, X_wetlab, _, _, _, _, _, X_test, _ = get_split_training_and_test_data(
                        "1PGA", cutoff_distance=5., p=np.arange(num_wet_lab_obs))
_X = np.vstack([x_wild_type, X_test, X_wetlab])

def test_mutations_consistent():
    """
    Test against gp_modeling parsing reference for experimental mutations
    """
    assert len(X) == len(_X)
    assert np.all([x == y for x, y in zip(X, _X)])

import matlab
import matlab.engine
K_script = """
function K=mWDK(WTSeq,mutations,al,depth,normalize,S)
%% MWDK calculates a graph kernel matrix with fixed adjacency matrix
%
% INPUT:    WTSeq = sequence of the wild type protein. (chars or integers)
%           mutations = cell array of strings containing the mutations in
%               format AiB, where A is the amino acid in the WTSeq and B is
%               the mutated residue. i is the residue number. Multiple
%               mutations where there are several of these triplets are
%               also possible (eg. M1AL5I).
%           al = adjacency list. Cell array that contains an array of
%               neighbours for each residue.
%           depth = depth of the neighbourhood. If 1, then only residues
%               in contact are considered, i
%           normalize = boolean. If true, the matrix is normalized
%           S = subsitution matrix (PSD)
%
% OUTPUT:   K = the calculated kernel matrix

sequences=constructSequences(WTSeq,mutations,1); % includes a sequence for the WT

N=size(sequences,1); % num of sequences
n=length(WTSeq); % num of AAs
numNbs=cellfun(@length,al);

% Determine the neighbourhood residues
if depth>1
    alit=cell(n,1);
    for d=1:depth
        for i=1:n
            ali=al{i};
            tmp=al(ali);
            alit{i}=unique([tmp{:,:}]);
        end
        al=alit;
    end
end

K = zeros(N,N);
neighpos = find(numNbs > 0); % no lonely residues
for i=1:length(neighpos)
    r=neighpos(i);
    zs = sequences(:,r); % selectors
    ns = sequences(:,al{r});
    zscores = S(zs,zs);
    nscores = squeeze( sum( bsxfun(@(i,j) S(i,j), ns, permute(ns, [3 2 1])), 2) );
    K = K + zscores .* nscores;
end

if normalize
    K=K./sqrt(diag(K)*diag(K)'); % normalize
end

function sequencesNew=constructSequences(sequenceWT, mutations1, includeWT)
%% CONSTRUCTSEQUENCES creates a matrix of sequences 
% based on the given WT sequence and mutations
%
% INPUT: sequenceWT = sequence of the WT protein
%        mutations = cell array of mutations. One cell contains all
%           mutations that are done to one protein.
%
% OUTPUT: sequences = matrix with one sequence on every row. The first
%           one is the WT protein which is followed by the mutated proteins
%           in the same order as given in mutations
% 

if isletter(sequenceWT(1))
    sequenceWT=double(aa2int(sequenceWT));
end

numMut=numel(mutations1);
if includeWT; wt=1; else; wt=0; end
sequencesNew=repmat(sequenceWT,numMut+wt,1);

for i=1:numMut
    mut=mutations1{i};
    letter_locs=isletter(mut);
    if mod(sum(letter_locs),2)~=0 || any(~ismember(mut,'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
        error('Invalid mutation number %d: %s',i,mut)
    end
    letter_inds=find(letter_locs);
    for j=1:2:sum(letter_locs)
        num=str2double(mut(letter_inds(j)+1:letter_inds(j+1)-1));
        if isnan(num) 
            warning('Mutation %s at index %d is not known. The WT sequence is left at this location.',mut,i);
            break
        elseif sequenceWT(num)~=aa2int(mut(letter_inds(j)))
            warning('Mutation %s at index %d does not match the WT, the original AA is %c',mut,i, int2aa(sequenceWT(num)));
        end
        sequencesNew(i+wt,num)=double(aa2int(mut(letter_inds(j+1))));
    end
end
"""
with open("kernel_script.m", "w") as outfile:
    outfile.write(K_script)
eng = matlab.engine.start_matlab()
matlab_m = matlab.double(m[0].tolist())
matlab_contacts = [matlab.int8(contacts.tolist()) for contacts in ref_contact_graph]

def test_normalized_kernel():
    """
    Test computed (normalized) matrix kernel values against reference K values
    """
    for i, m in enumerate(matrices):
        kernel = MatrixKernel(matrix=m[0], matrix_id=None)
        k = kernel.k(sequences=X, adjacencies=ref_contact_graph)
        print(mut_S_exp)
        ref_K = eng.kernel_script(sequence_WT, mut_S_exp, matlab_contacts, 
                    1, True, matlab_m, nargout=0)
        #np.testing.assert_almost_equal(k.detach().numpy(), ref_K_list[i][0])
        np.testing.assert_almost_equal(k.detach().numpy(), ref_K)