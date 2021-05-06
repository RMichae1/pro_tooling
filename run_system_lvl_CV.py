from scipy.io import loadmat
import os
import warnings
import subprocess
import mlflow
from contact_mapper import ContactMapper
from utility import parse_matlab_mutation_file

todos = []
done = []
pdbs = ["1BVC"] #, "2RN2", "4LYZ", "2LZM", "1RTB"] #,"1BVC", "1PGA", "1CSP", "1BPI", "1RGG"]

EXPERIMENTAL_DATA = {"mgpf": "./data/mgp/ddg_protherm.mat",
                    "tll": "./data/tll/lipase_variants_tll_tm_tapo_20nov2020.xlsx",
                    "blat": "./data/blat/BLAT_ECOLX_Ranganathan2015.csv"}

IN_SILICO_DATA = {"mgpf": "./data/mgp/ddg_rosetta_single.mat",
                "tll": "./data/tll/TLL_IS_closed_results.xlsx"}


def get_positions(pdb: str) -> str:
    cm_tri = ContactMapper(pdb_file=f"./pdb/{pdb.lower()}.pdb", tri_dist=True)
    return cm_tri.sequence


def run_sys_CV(pdb, idx, cv, experiment, run_id, data=None, ref=False, optim=True, no_fusion=False, verbose=False,
               ref_contact_map=False, vae_input=False, vae_kernel=False):
    command_lst = ["C:/Users/RCML/Anaconda3/envs/mgpfusion/python.exe", "//wsl$/Ubuntu/home/rcml/pro_tooling/run_experiments.py", "-p", f"{pdb}", "-i", f"{idx}",
                   "-r", f"{cv}", "--seed", "3032021", "--experiment", f"{experiment}", "--run_id", f"{run_id}",
                   "--data", data, "--experimental_data", f"{EXPERIMENTAL_DATA.get(data)}",
                   "--simulated_data", f"{IN_SILICO_DATA.get(data)}"]
    if optim:
        command_lst += ["-o"]
    if ref:
        command_lst += ["-m"]
    if verbose:
        command_lst += ["-v"]
    if no_fusion:
        command_lst += ["--no_fusion"]
    if ref_contact_map:
        command_lst += ["--ref_contact"]
    if vae_input:
        command_lst += ["--vae_input"]
    if vae_kernel:
        command_lst += ["--vae_kernel"]
    subprocess.run(command_lst)


def create_mlflow_run_pos_lvl(pdb: str, cv: str, optim: bool, ref: bool, no_fusion: bool = False, data: str = None,
                              ref_contact_map: bool = False, vae_input: bool = False, vae_kernel: bool=False) -> None:
    sequence = get_positions(pdb)
    experiment_name = f"{pdb}: {cv}"
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment) as run:
        exp_params = {"pdb": pdb, "cv": cv, "optimization": optim, "2σ": ref, "reference_contacts": ref_contact_map, 
                    "NO fusion": no_fusion, "vae": vae_input, "S_vae": vae_kernel}
        mlflow.log_params(exp_params)
        for idx, _ in enumerate(sequence):
            # run position lvl no optimization
            run_sys_CV(pdb, idx, cv=cv, ref=ref, optim=optim, experiment=experiment_name, run_id=run.info.run_id,
                       no_fusion=no_fusion, data=data, verbose=True, ref_contact_map=ref_contact_map,
                       vae_input=vae_input, vae_kernel=vae_kernel)
    mlflow.end_run()
    return None


def create_mlflow_run_mut_lvl(pdb: str, cv: str, optim: bool, ref: str, no_fusion: bool=False, data: str = None,
                              ref_contact_map: bool=False) -> None:
    exp_mutations = parse_matlab_mutation_file(f"./data/mgp/ddg_protherm.mat",
                                               query="ddg_protherm").get(pdb)
    experiment_name = f"{pdb}: {cv}"
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment) as run:
        exp_params = {"pdb": pdb, "cv": cv, "optimization": optim, "2σ": ref, "reference_contacts": ref_contact_map}
        mlflow.log_params(exp_params)
        for idx in range(1, len(exp_mutations)+1): # exclude WT zero round
            # run position lvl no optimization
            run_sys_CV(pdb, idx, cv=cv, ref=ref, optim=optim, experiment=experiment_name, run_id=run.info.run_id,
                       no_fusion=no_fusion, data=data, verbose=True, ref_contact_map=ref_contact_map)
    mlflow.end_run()
    return None


def run_MGPF() -> None:
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    for pdb in pdbs:
        # TODO rerun 1BVC position level - its overwritten with mutation level
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False,
        #                          no_fusion=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False,
        #                        no_fusion=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False, no_fusion=True,
        #                          ref_contact_map=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False, no_fusion=True,
        #                        ref_contact_map=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False,
        #                          ref_contact_map=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False,
        #                        ref_contact_map=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=True,
        #                        ref_contact_map=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=True,
        #                        ref_contact_map=True, data="mgpf")
        # create_mlflow_run_mut_lvl(pdb=pdb, cv="mut_lvl", optim=False, ref=False,
        #                          ref_contact_map=True, data="mgpf")
        # create_mlflow_run_mut_lvl(pdb=pdb, cv="mut_lvl", optim=True, ref=False,
        #                        ref_contact_map=True, data="mgpf")
        # create_mlflow_run_mut_lvl(pdb=pdb, cv="mut_lvl", optim=False, ref=True,
        #                        ref_contact_map=True, data="mgpf")
        # create_mlflow_run_mut_lvl(pdb=pdb, cv="mut_lvl", optim=True, ref=True,
        #                        ref_contact_map=True, data="mgpf")
        done.append(pdb)
    print(f"Done: {done}")


def run_TLL() -> None:
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    # create_mlflow_run_pos_lvl(pdb="1TIB", cv="pos_lvl", optim=False, ref=False, data="tll")
    # create_mlflow_run_pos_lvl(pdb="1TIB", cv="pos_lvl", optim=True, ref=False, data="tll")
    # create_mlflow_run_pos_lvl(pdb="1TIB", cv="pos_lvl", optim=False, ref=True, data="tll")
    # create_mlflow_run_pos_lvl(pdb="1TIB", cv="pos_lvl", optim=True, ref=True, data="tll")
    # create_mlflow_run_pos_lvl(pdb="1TIB", cv="pos_lvl", optim=False, ref=False, no_fusion=True, data="tll")
    # create_mlflow_run_pos_lvl(pdb="1TIB", cv="pos_lvl", optim=True, ref=False, no_fusion=True, data="tll")
    # create_mlflow_run_pos_lvl(pdb="1TIB", cv="pos_lvl", optim=False, ref=True, no_fusion=True, data="tll")
    # create_mlflow_run_pos_lvl(pdb="1TIB", cv="pos_lvl", optim=True, ref=True, no_fusion=True, data="tll")
    create_mlflow_run_pos_lvl(pdb="1TIB", cv="mut_lvl", optim=False, ref=False, data="tll")
    create_mlflow_run_pos_lvl(pdb="1TIB", cv="mut_lvl", optim=True, ref=False, data="tll")
    create_mlflow_run_pos_lvl(pdb="1TIB", cv="mut_lvl", optim=False, ref=True, data="tll")
    create_mlflow_run_pos_lvl(pdb="1TIB", cv="mut_lvl", optim=True, ref=True, data="tll")


def run_BLAT() -> None:
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    # create_mlflow_run_pos_lvl(pdb="1FQG", cv="pos_lvl", optim=False, ref=False, 
    #                           no_fusion=True, data="blat")
    # create_mlflow_run_pos_lvl(pdb="1FQG", cv="pos_lvl", optim=True, ref=False, 
    #                           no_fusion=True, data="blat")
    # create_mlflow_run_pos_lvl(pdb="1FQG", cv="pos_lvl", optim=False, ref=True, 
    #                           no_fusion=True, data="blat")
    # create_mlflow_run_pos_lvl(pdb="1FQG", cv="pos_lvl", optim=True, ref=True, 
    #                           no_fusion=True, data="blat")
    create_mlflow_run_pos_lvl(pdb="1FQG", cv="pos_lvl", optim=False, ref=False, 
                                vae_input=True, data="blat")
    create_mlflow_run_pos_lvl(pdb="1FQG", cv="pos_lvl", optim=True, ref=False, 
                                vae_input=True, data="blat")
    create_mlflow_run_pos_lvl(pdb="1FQG", cv="pos_lvl", optim=False, ref=False, 
                                vae_kernel=True, data="blat")
    create_mlflow_run_pos_lvl(pdb="1FQG", cv="pos_lvl", optim=True, ref=False, 
                                vae_kernel=True, data="blat")
    create_mlflow_run_pos_lvl(pdb="1FQG", cv="pos_lvl", optim=False, ref=False, 
                                vae_input=True, vae_kernel=True, data="blat")
    create_mlflow_run_pos_lvl(pdb="1FQG", cv="pos_lvl", optim=True, ref=False, 
                                vae_input=True, vae_kernel=True, data="blat")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_TLL()
    #run_MGPF()
    #run_BLAT()
    
