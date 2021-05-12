from scipy.io import loadmat
import os
import warnings
import subprocess
import mlflow
from contact_mapper import ContactMapper
from utility import parse_matlab_mutation_file

EXPERIMENTAL_DATA = {"mgpf": "./data/mgp/ddg_protherm.mat",
                     "tll": "./data/tll/lipase_variants_tll_tm_tapo_20nov2020.xlsx",
                     "blat": "./data/blat/BLAT_ECOLX_Ranganathan2015.csv",
                     "ubq": "./data/ubq/RL401_Bolon2013.csv",
                     "pga": "./data/pga/Nisthal_Mayo_2019_updated_3xESLyS9.csv",
                     "hex": "./data/hex/Hexosaminidase_SSL_data_simple.xlsx"}

IN_SILICO_DATA = {"mgpf": "./data/mgp/ddg_rosetta_single.mat",
                  "tll": "./data/tll/TLL_IS_closed_results.xlsx",
                  "ubq": "./data/ubq/vae_ubq_IS_samples.pkl",
                  "blat": "./data/blat/vae_blat_IS_samples.pkl",
                  "pga": "./data/pga/vae_pga_IS_samples.pkl",
                  "hex": "./data/hex/vae_hex_IS_samples.pkl"}


def get_positions(pdb: str) -> str:
    cm_tri = ContactMapper(pdb_file=f"./pdb/{pdb.lower()}.pdb", tri_dist=True)
    return cm_tri.sequence


def run_sys_CV(pdb, idx, cv, experiment, run_id, data=None, ref=False, optim=True, fusion=False, verbose=False,
               ref_contact_map=False, vae_input=False, vae_kernel=False, frac=1.):
    command_lst = ["/z/home/rcml/anaconda3/bin/python", "/z/home/rcml/pro_tooling/run_experiments.py", "-p",
                   f"{pdb}", "-i", f"{idx}",
                   "-r", f"{cv}", "--seed", "3032021", "--experiment", f"{experiment}", "--run_id", f"{run_id}",
                   "--data", data, "--experimental_data", f"{EXPERIMENTAL_DATA.get(data)}",
                   "--simulated_data", f"{IN_SILICO_DATA.get(data)}", "--fraction", f"{frac}"]
    if optim:
        command_lst += ["-o"]
    if ref:
        command_lst += ["-m"]
    if verbose:
        command_lst += ["-v"]
    if fusion:
        command_lst += ["--fusion"]
    if ref_contact_map:
        command_lst += ["--ref_contact"]
    if vae_input:
        command_lst += ["--vae_input"]
    if vae_kernel:
        command_lst += ["--vae_kernel"]
    subprocess.run(command_lst)


def create_mlflow_run_pos_lvl(pdb: str, cv: str, optim: bool, ref: bool, fusion: bool = False, data: str = None,
                              ref_contact_map: bool = False, vae_input: bool = False, vae_kernel: bool = False,
                              frac: float = 1.) -> None:
    sequence = get_positions(pdb)
    experiment_name = f"{pdb}: {cv}"
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment) as run:
        exp_params = {"pdb": pdb, "cv": cv, "optimization": optim, "2σ": ref, "reference_contacts": ref_contact_map,
                      "fusion": fusion, "vae_input": vae_input, "vae_kernel": vae_kernel, "data_train": frac}
        mlflow.log_params(exp_params)
        for idx, _ in enumerate(sequence):
            # run position lvl no optimization
            run_sys_CV(pdb, idx, cv=cv, ref=ref, optim=optim, experiment=experiment_name, run_id=run.info.run_id,
                       fusion=fusion, data=data, verbose=True, ref_contact_map=ref_contact_map,
                       vae_input=vae_input, vae_kernel=vae_kernel, frac=frac)
    mlflow.end_run()
    return None


def create_mlflow_run_mut_lvl(pdb: str, cv: str, optim: bool, ref: str, fusion: bool = False, data: str = None,
                              ref_contact_map: bool = False) -> None:
    exp_mutations = parse_matlab_mutation_file(f"./data/mgp/ddg_protherm.mat",
                                               query="ddg_protherm").get(pdb)
    experiment_name = f"{pdb}: {cv}"
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment) as run:
        exp_params = {"pdb": pdb, "cv": cv, "optimization": optim, "2σ": ref, "reference_contacts": ref_contact_map}
        mlflow.log_params(exp_params)
        for idx in range(1, len(exp_mutations) + 1):  # exclude WT zero round
            # run position lvl no optimization
            run_sys_CV(pdb, idx, cv=cv, ref=ref, optim=optim, experiment=experiment_name, run_id=run.info.run_id,
                       fusion=fusion, data=data, verbose=True, ref_contact_map=ref_contact_map)
    mlflow.end_run()
    return None


def run_MGPF() -> None:
    todos = []
    done = []
    pdbs = ["1BVC"]  # , "2RN2", "4LYZ", "2LZM", "1RTB"] #,"1BVC", "1PGA", "1CSP", "1BPI", "1RGG"]
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    for pdb in pdbs:
        # TODO rerun 1BVC position level - its overwritten with mutation level
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False,
        #                          fusion=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False,
        #                        fusion=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False, fusion=True,
        #                          ref_contact_map=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False, fusion=True,
        #                        ref_contact_map=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False,
        #                           ref_contact_map=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False,
        #                        ref_contact_map=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=True,
        #                        ref_contact_map=True, data="mgpf")
        # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=True,
        #                        ref_contact_map=True, data="mgpf")
        # create_mlflow_run_mut_lvl(pdb=pdb, cv="mut_lvl", optim=False, ref=False,
        #                           ref_contact_map=True, data="mgpf")
        # create_mlflow_run_mut_lvl(pdb=pdb, cv="mut_lvl", optim=True, ref=False,
        #                        ref_contact_map=True, data="mgpf")
        # create_mlflow_run_mut_lvl(pdb=pdb, cv="mut_lvl", optim=False, ref=True,
        #                        ref_contact_map=True, data="mgpf")
        # create_mlflow_run_mut_lvl(pdb=pdb, cv="mut_lvl", optim=True, ref=True,
        #                        ref_contact_map=True, data="mgpf")
        done.append(pdb)
    print(f"Done: {done}")


def run_mGP(pdb: str, data: str) -> None:
    # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False,
    #                           fusion=False, data=data, frac=0.3)
    create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False,
                              fusion=False, data=data, frac=0.3)
    # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False,
    #                           fusion=False, data=data, frac=0.5)
    create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False,
                              fusion=False, data=data, frac=0.5)
    # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False,
    #                           fusion=False, data=data, frac=0.9)
    create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False,
                              fusion=False, data=data, frac=0.9)


def run_DES_K(pdb: str, data: str) -> None:
    create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False, vae_kernel=True, data=data, frac=0.3)
    create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False,
                              vae_kernel=True, data=data, frac=0.5)
    create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False,
                              vae_kernel=True, data=data, frac=0.9)
    # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False,
    #                           vae_kernel=True, data=data, frac=0.3)
    # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False,
    #                           vae_kernel=True, data=data, frac=0.5)
    # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False,
    #                           vae_kernel=True, data=data, frac=0.9)


def run_VAE_in_silico(pdb: str, data: str) -> None:
    # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False, fusion=True,
    #                           vae_input=True, data=data, frac=0.3)
    create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False, fusion=True, vae_input=True, data=data, frac=0.3)
    # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False, fusion=True,
    #                           vae_input=True, data=data, frac=0.5)
    create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False, fusion=True,
                              vae_input=True, data=data, frac=0.5)
    # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False, fusion=True
    #                           vae_input=True, data=data, frac=0.9)
    create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False, fusion=True,
                              vae_input=True, data=data, frac=0.9)
    # create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=False, ref=False, fusion=True
    #                           vae_input=True, vae_kernel=True,
    #                           data=data)


def run_VAE_in_silico_DES_K(pdb: str, data: str) -> None:
    create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False, fusion=True, vae_input=True,
                              vae_kernel=True, data=data, frac=0.3)
    create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False, fusion=True,
                              vae_input=True, vae_kernel=True,
                              data=data, frac=0.5)
    create_mlflow_run_pos_lvl(pdb=pdb, cv="pos_lvl", optim=True, ref=False, fusion=True,
                              vae_input=True, vae_kernel=True,
                              data=data, frac=0.9)
    # TODO run without optimization


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # run_MGPF()
    #pdbs = ["1FQG", "1UBQ", "1PGA", "D45"]
    #data = ["blat", "ubq", "pga", "hexo"]
    pdbs = ["1UBQ", "1PGA"]
    data = ["ubq", "pga"]
    for pdb, d in zip(pdbs, data):
        # run_mGP(pdb=pdb, data=d)
        run_DES_K(pdb=pdb, data=d)
        # run_VAE_in_silico(pdb=pdb, data=d)
        # run_VAE_in_silico_DES_K(pdb=pdb, data=d)
