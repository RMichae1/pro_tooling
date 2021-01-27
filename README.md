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
+ `./pdb/` pdb files for the listed proteins have been directly downloaded from the [RCSB Protein Data Bank](https://www.rcsb.org/)
+ `./fig/` contains all figures created from running the `main.py` function
+ `./results/` contains all results computed from `main.py` runs as well as `run_experiments.py` - `./results/hyper/` contains MKL results for e.g. mean weights from MKL, while other subdirectories are method specific results
+ `./test/` contains test modules against the reference implementation that can be run with `pytest ${modulename}`