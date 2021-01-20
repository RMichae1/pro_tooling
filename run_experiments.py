from scipy.io import loadmat
import os
from contact_mapper import ContactMapper
from protein_representation import ProteinCollection
from data_scaler import BayesScaler
from gp_regression import GPRegression

from utility import parse_matlab_mutation_file, parse_mutations, convert_graph_from_matlab_file
from utility import convert_aa_sequence, preprocess_observations

import numpy as np
import pickle

todos = ["2RN2", "4LYZ","2LZM", "1RTB", "1BVC"]
pdbs = ["1PGA", "1CSP", "1BPI", "1RGG"]
mutlvl_pdbs = ["1PGA"]
buggy = ["1BNI", "1VQB", "1LZI", "2CI2","1RN1", "1PIN"]

# get mutations for pdb
exp_mutations = parse_matlab_mutation_file("./data/ddg_protherm.mat", query="ddg_protherm")
sim_mutations = parse_matlab_mutation_file("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")

for pdb in pdbs:
    # load data from reference files
    filename = f"./pdb/{pdb.lower()}.pdb"
    save_fig = os.path.join(os.path.abspath(''), "fig/")
    ref_file = loadmat(f"./data/{pdb.upper()}.mat")
    ref_adj = convert_graph_from_matlab_file(ref_file["contact_map"])
    
    # generate contactmaps, distance matrix plots
    cm1 = ContactMapper(pdb_file=filename, tri_dist=True)
    #cm1.plot_distance_matrix(save_fig=save_fig)
    #cm1.plot_contact_map(save_fig=save_fig)
    
    # generate sub_matrices and representations ...
    pcol = ProteinCollection(cm1, pdb_ID=pdb, mutations_exp=exp_mutations, mutations_sim=sim_mutations, TESTING=False)
    #pcol.plot_sub_matrices(savefig=save_fig)
   
    # TODO implement Simons parsing here
    # extract mutations
    mut_S_exp, _, ΔΔg_exp, mut_ids_exp = parse_mutations(mutation_dict=exp_mutations.get(pcol.pdb_ID), 
                                                    sequence=pcol.sequence, adjacency=ref_adj)
    mut_S_is, _, ΔΔg_is, mut_ids_is = parse_mutations(mutation_dict=sim_mutations.get(pcol.pdb_ID), 
                                                    sequence=pcol.sequence, adjacency=ref_adj)
    X_exp, X_is = convert_aa_sequence(mut_S_exp), convert_aa_sequence(mut_S_is)
    y_wt = np.array([0])[:, np.newaxis]
    X_wt = convert_aa_sequence([pcol.sequence])
    
    # Bayesian Scaling
    bs = BayesScaler(is_mutations=mut_ids_is, ΔΔg=pcol.ΔΔg_is, exp_mutations=mut_ids_exp, 
                        experimentally_observed_ΔΔg=pcol.ΔΔg_exp, TESTING=False, pdb_ID=pdb.upper())
    ΔΔg_exp = ΔΔg_exp[:, np.newaxis]
    ΔΔg_is_scaled = bs.transform(ΔΔg_is)[:, np.newaxis]
    bs.plot_scaling(save_fig=save_fig)
    
    # Scale y-values as done in the implementation by normalizing with mean and max
    mean_y, max_y, y_wt, ΔΔg_exp, ΔΔg_is_scaled = preprocess_observations(y_wt, ΔΔg_exp, ΔΔg_is_scaled)

    gpr = GPRegression(protein_representation=pcol, X_wt=X_wt, X_exp=X_exp, X_is=X_is, 
                         y_wt=y_wt, y_exp=ΔΔg_exp, y_is=ΔΔg_is_scaled, adjacencies=ref_adj, 
                         σ_T=bs.σ_T, y_max=max_y, y_mean=mean_y)
    # get optimized values over all data
    start_nll = gpr.neg_ll()
    try:
        gpr.parameter_optimization()
    except:
        print(f"Optimization broke {pdb}")
        gpr.reset_trainable_parameters()
    weights = gpr.weights.get_value()
    sigma_s = gpr.σ_S.get_value()
    sigma_e = gpr.σ_E.get_value()
    t = gpr.t.get_value()
    end_nll = gpr.neg_ll()
    hyper_dict = {"w": weights, "sigma_s": sigma_s, "sigma_e": sigma_e, "t": t}
    with open(f"./hyperparameters_X_all_{pdb}.pickle", "wb") as outfile:
        pickle.dump(hyper_dict, outfile)
    
    # as done in reference (2xsigma and position level naive)
    ref_results = gpr.position_level_CV_reference(ref=True)
    with open(f"./results_REFERENCE_pos_lvl_{pdb}.pickle", "wb") as outfile:
        pickle.dump(ref_results, outfile)

    # position level naive corrected
    corrected_res = gpr.position_level_CV(ref=False)
    with open(f"./results_NO_ERROR_REFERENCE_pos_lvl_{pdb}.pickle", "wb") as outfile:
        pickle.dump(corrected_res, outfile)
        
    # experimentals in testing
    pos_results = gpr.position_level_CV()
    with open(f"./results_pos_lvl_{pdb}.pickle", "wb") as outfile:
        pickle.dump(pos_results, outfile)
    
    # mut_results = gpr.mutation_level_CV()
    # with open(f"./results_mut_lvl_{pdb}.pickle", "wb") as outfile:
    #     pickle.dump(mut_results, outfile)