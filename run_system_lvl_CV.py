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
    pga_file = loadmat(os.path.join(os.path.dirname(__file__), os.path.join("data/mgp/", f"{pdb}.mat")))
    cm_tri = ContactMapper(pdb_file=f"./pdb/{pdb.lower()}.pdb", tri_dist=True)
    return cm_tri.sequence


def run_sys_CV(pdb, idx, cv, experiment, run_id, ref=False, optim=True, verbose=False):
    command_lst = ["python",
                    "/home/rimichael/pro_tooling/run_experiments.py",
                    "-p", f"{pdb}", "-i", f"{idx}", "-r", f"{cv}", "--seed", "3032021", "--experiment", f"{experiment}",
                   "--run_id", f"{run_id}"]
    if optim:
        command_lst += ["-o"]
    if ref:
        command_lst += ["-m"]
    if verbose:
        command_lst += ["-v"]
    subprocess.run(command_lst)


def create_mlflow_run(pdb: str, cv: str, optim: bool, ref: str) -> None:
    sequence = get_positions(pdb)
    experiment_name = f"{pdb}: {cv}"
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment) as run:
        exp_params = {"pdb": pdb, "cv": cv, "optimization": optim, "2σ": ref}
        mlflow.log_params(exp_params)
        for idx, _ in enumerate(sequence):
            # run position lvl no optimization
            run_sys_CV(pdb, idx, cv=cv, ref=ref, optim=optim, experiment=experiment_name, run_id=run.info.run_id,
                       verbose=True)
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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()