from scipy.io import loadmat
import os
import warnings
import subprocess
import mlflow
from contact_mapper import ContactMapper

todos = []
done = []
pdbs = ["1BVC", "2LZM", "1PGA", "1CSP", "1BPI", "1RGG", "1RTB", "2RN2", "4LYZ"]


def get_positions(pdb: str) -> str:
    cm_tri = ContactMapper(pdb_file=f"./pdb/{pdb.lower()}.pdb", tri_dist=True)
    return cm_tri.sequence


def run_sys_CV(pdb, idx, cv, experiment, run_id, data=None, ref=False, optim=True, no_fusion=False, verbose=False):
    command_lst = ["python", "/home/rimichael/pro_tooling/run_experiments.py", "-p", f"{pdb}", "-i", f"{idx}",
                   "-r", f"{cv}", "--seed", "3032021", "--experiment", f"{experiment}", "--run_id", f"{run_id}"]
    if optim:
        command_lst += ["-o"]
    if ref:
        command_lst += ["-m"]
    if verbose:
        command_lst += ["-v"]
    if no_fusion:
        command_lst += ["--no_fusion"]
    if data:
        command_lst += ["--data"]
        command_lst += [data]
    subprocess.run(command_lst)


def create_mlflow_run(pdb: str, cv: str, optim: bool, ref: str, no_fusion: bool, data: str = None) -> None:
    sequence = get_positions(pdb)
    experiment_name = f"{pdb}: {cv}"
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment) as run:
        exp_params = {"pdb": pdb, "cv": cv, "optimization": optim, "2σ": ref}
        mlflow.log_params(exp_params)
        for idx, _ in enumerate(sequence):
            # run position lvl no optimization
            run_sys_CV(pdb, idx, cv=cv, ref=ref, optim=optim, experiment=experiment_name, run_id=run.info.run_id,
                       no_fusion=no_fusion, data=data, verbose=True)
    # TODO compute overall stats for experiment
    mlflow.end_run()
    return None


def main() -> None:
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    for pdb in pdbs:
        create_mlflow_run(pdb=pdb, cv="pos_lvl", optim=False, ref=False)
        create_mlflow_run(pdb=pdb, cv="pos_lvl", optim=True, ref=False)
        create_mlflow_run(pdb=pdb, cv="pos_lvl", optim=False, ref=True)
        create_mlflow_run(pdb=pdb, cv="pos_lvl", optim=True, ref=True)
        done.append(pdb)
    print(f"Done: {done}")


def run_TLL() -> None:
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    create_mlflow_run(pdb="1TIB", cv="pos_lvl", optim=False, ref=False, no_fusion=True, data="tll")
    create_mlflow_run(pdb="1TIB", cv="pos_lvl", optim=True, ref=False, no_fusion=True, data="tll")
    create_mlflow_run(pdb="1TIB", cv="pos_lvl", optim=False, ref=True, no_fusion=True, data="tll")
    create_mlflow_run(pdb="1TIB", cv="pos_lvl", optim=True, ref=True, no_fusion=True, data="tll")


def run_BLAT() -> None:
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    create_mlflow_run(pdb="1FQG", cv="pos_lvl", optim=False, ref=False, no_fusion=True, data="blat")
    create_mlflow_run(pdb="1FQG", cv="pos_lvl", optim=True, ref=False, no_fusion=True, data="blat")
    create_mlflow_run(pdb="1FQG", cv="pos_lvl", optim=False, ref=True, no_fusion=True, data="blat")
    create_mlflow_run(pdb="1FQG", cv="pos_lvl", optim=True, ref=True, no_fusion=True, data="blat")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    #run_TLL()
    run_BLAT()
    #main()
    
