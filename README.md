# **pro_tooling** a Protein Tooling and Modeling Library
### Tooling Lib for Protein Interaction, Mutation Modeling and Visualization.

Efficient covariance matrix calculations have been implemented with numpy. GP Regression modeling has been implemented in PyTorch and can be utilized for predictive protein stability values.

See reference paper [by Jokinen et al. - mGPfusion](https://academic.oup.com/bioinformatics/article/34/13/i274/5045756) .

### Module Overview
+ `contact_mapper.py` with its `ContactMapper` class parses pdb files and computes distance matrices and contact-maps
+ `data_scaler.py` contains the `BayesScaler` used for in-silico transformation 
+ `gp_regreesion.py` contains the `GPRegression`, which includes MKL optimization (using `torch.optim`) as well as GP fitting using PyTorch, also implements mutation-lvl and position-lvl CV for GP training and assessment
+ `graphkernel.py` implements the `KernelLoader` as interface to `MatrixKernel`, which computes kernel values, given input-sequences and their adjacencies
+ `report.ipynb` is a Notebook to use all modules to create output results from the input-data
+ `run_experiments.py` is a script run to create results from input data
+ `utility.py` contains all required utility functions, from mutation and matlab parsing, sequence conversions, etc. both from the `gp_modeling` project as well from this; also contains the `Variable` class used for unconstrained optimization in GP regression MKL.
+ `visualization.py` contains all subroutines related to plotting and generating tabular result outputs.

`requirements.txt` contains Python environment required packages to run the code.

### File Structure
+ `./data/` contains reference matlab files for all proteins presented in the reference paper, as well as experimental and in-silico data. `ddg_protherm.mat` contains the experimental observations and associated mutations with their ΔΔG values, while `rosetta_multi.mat` and `rosetta_single.mat` contain in-silico mutations with associated ΔΔG values.
For VAE related analysis: focus are the `blat`, `pga`, `ubq` directories. Each subdirectory contains EVcoupling derived MSA - each with different bit-values from the alignment. These (_ALL suffix) were used to fit the VAEs. 
Additionally the subdirectories contain the csv of the DeepSequence publication, if available under the name as has been published in the supplementary material. Further a .csv file with a protabank ID
is contained. This is generally used for validation and testing against the MSA-VAEs. 
+ For MSA files ensure that the WT sequence is the first entry in the parsed list of strings!
+ `./pdb/` pdb files for the listed proteins have been directly downloaded from the [RCSB Protein Data Bank](https://www.rcsb.org/)
+ `./fig/` contains all figures created from running the `main.py` function
+ `./results/` contains all results computed from `main.py` runs as well as `run_experiments.py` - `./results/hyper/` contains MKL results for e.g. mean weights from MKL, while other subdirectories are method specific results
+ `./test/` contains test modules against the reference implementation that can be run with `pytest ${modulename}`


### Experiments
The original data was run on Protherm and Rosetta simulations and in reference to the mgpfusion publication the data can be found in data/mgp.

To test against other data-sets the beta-lactamase dataset was analyzed with reference to previous publications.

(internal VAE work uses Ranganathan2015, whereas ddG values are available in Palzkill2012)


### Workflow to generate the MSAs
1. Collect all sequences sequenced by Novozymes that matches the internally used domain identier annotation for Thermomyces lanuginosus.
2. Collect all BFD sequences using HHBlits against Thermomyces lanuginosus with a sequence identity of 40%.
3. Remove duplicates.
4. Remove singletons by clustering remaining sequences by 70% sequence identity and subsequently removing clusters of single sequences. Such singletons would likely exacerbate following sequence alignmentsRemove sequences shorter than 200 residues and longer than 500 residues.
5. Construct a multiple sequence alignment using the alignment tool FAMSA
6. Remove flanks of query protein (TLL)
7. any column corresponding to an introduced gap in Thermomyces lanuginosus lipase is also removed
8. Any sequence containing special characters not corresponding to natural amino acids or gaps were removed as well as any gappy sequences consisting of 50% gaps or more
9. also removed were sequences with more than 50% gaps





LIPASE v1 from "./data/tll/seqs_in_int_nogaps_sp400_Mar14_data_all_jaks_Apr3_trimmed.pkl"