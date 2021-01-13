from contact_mapper import ContactMapper
from protein_representation import ProteinCollection
from graphkernel import MatrixKernel
from utility import parse_mutations
from data_scaler import BayesScaler
from gp_regression import GPRegression

if __name__ == "__main__":

    # Create and test Contact Mapper
    # # example case 1PGA - CA-distance

    # example case 1PGA - residue distance
    cm_tri = ContactMapper(pdb_file="./pdb/1pga.pdb", tri_dist=True)
    
    mutational_dict_exp = parse_mutations("./data/ddg_protherm.mat", query="ddg_protherm")
    mutational_dict_is = parse_mutations("./data/ddg_rosetta_single.mat", query="ddg_rosetta_single")

    pcol = ProteinCollection(cm_tri, pdb_ID="1PGA", mutations_exp=mutational_dict_exp, mutations_sim=mutational_dict_is,
                    TESTING=True)
    print(pcol.matrices_df)
    # print(pcol.plot_sub_matrices())

    # instantiate and parse IS data

    # scale using Bayesian Scaling
    exp_mutation_ids = pcol.mutation_ids[:len(pcol.mut_S_exp)] # includes WT??
    is_mutation_ids = pcol.mutation_ids[len(pcol.mut_S_exp):]
    # bs_rosetta = BayesScaler(is_mutations=is_mutation_ids, ΔΔg=pcol.ΔΔg_is, exp_mutations=exp_mutation_ids, experimentally_observed_ΔΔg=pcol.ΔΔg_exp, 
    #                      TESTING=True, pdb_ID="1PGA")

    # print("theta")
    # print(bs_rosetta.θ)
    # print("sigma")
    # print(bs_rosetta.σ_T)
    # print("sigma sampled")
    # print(bs_rosetta.σ_T_sampled)
    # print(bs_rosetta.θ)
    # print(bs_rosetta.print_summary())
    # print(bs_rosetta.plot_scaling())

    gpr = GPRegression(protein_representation=pcol)
    print(gpr.neg_ll())
    gpr.parameter_optimization()
    print(gpr.neg_ll())
    # gpr.plot_log_prob()
    # gpr.plot()