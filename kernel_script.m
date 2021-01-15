
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
