import Bio.PDB
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from os.path import basename 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform, pdist


class ContactMapper:
    def __init__(self, pdb_file: str, pdb_ID: str=None, angstrom_threshold: float=5.):
        self.pdb_file = pdb_file
        self.pdb_ID = basename(self.pdb_file).split(".")[0].upper() if not pdb_ID else pdb_ID
        self.angstrom_threshold = angstrom_threshold
        self.structure = Bio.PDB.PDBParser().get_structure(self.pdb_ID, self.pdb_file)
        self.model_obj: Model = self.structure[0]
        self.chains = [chain for chain in self.model_obj]
        self.dim = len(self.chains)
        self.distance_matrices: list = self.calc_distace_matrix()
        self.contact_maps: list = [distance_matrix < self.angstrom_threshold for distance_matrix in self.distance_matrices]

    @staticmethod
    def calc_residue_CA_distance(coord_X: np.ndarray, coord_Y: np.ndarray) -> np.float:
        diff_vector = coord_X - coord_Y
        return np.sqrt(np.sum(diff_vector*diff_vector))

    @staticmethod
    def get_CA_coords(res: Residue) -> np.ndarray:
        coord = None
        try:
            coord = res["CA"].coord
        except KeyError as e:
            # no C-alpha present, take mean of 3 residues instead
            coord = np.mean(np.array([a.coord for a in res]))
        return coord

    @staticmethod
    def calc_residue_tri_distance(res_X: Residue, res_Y: Residue) -> np.float:
        # distance min residues see mgpfusion/code/protein.m:494
        # TODO check how to apply pairwise distance given sequence residues
        pairwise_res_dist = pdist(res_X, res_Y)
        squareform_pair_res_dist = squareform(pairwise_res_dist)
        print(squareform_pair_res_dist)
        return min(min(squareform_pair_res_dist))
    
    def calculate_chain_distance(self, chain_X: Chain, chain_Y: Chain, tri_res_calculation=False) -> np.array:
        mat = np.zeros((len(chain_X), len(chain_Y)), np.float)
        for res_X_pos, res_X in enumerate(chain_X):
            for res_Y_pos, res_Y in enumerate(chain_Y):
                if tri_res_calculation:
                    mat[res_X_pos, res_Y_pos] = self.calc_residue_tri_distance(res_X, res_Y)
                else:
                    coord_X = self.get_CA_coords(res_X)
                    coord_Y = self.get_CA_coords(res_Y)
                    mat[res_X_pos, res_Y_pos] = self.calc_residue_CA_distance(coord_X, coord_Y)
        return mat

    def calc_distace_matrix(self) -> np.array:
        """
        Go through all sequences in all chains and build distance matrix by residue 
        :returns : distance matrix as array
        """
        # dist_matrix = np.zeros((self.dim, self.dim), np.float)
        dist_matrix = []
        for idx, chain_X in enumerate(self.chains):
            # calculate distance for each residue in all the given chains
            for idy, chain_Y in enumerate(self.chains[idx:]):
                # dist_matrices[idx, idy] = self.calculate_chain_distance(chain_X, chain_Y)
                dist_matrix.append(self.calculate_chain_distance(chain_X, chain_Y))
        return dist_matrix

    def get_min_distance(self):
        return min(min(mat for mat in self.distance_matrices))

    def get_max_distance(self):
        return max(max(mat for mat in self.distance_matrices))

    def plot_distance_matrix(self, save_fig=None):
        fig, ax = plt.subplots(self.dim, self.dim, figsize=(11, 9))
        for idx, d_mat in enumerate(self.distance_matrices):
            # handle 1x1 non-subscriptable axis vs. arrays for higher dim
            ax = ax[idx] if isinstance(ax, np.ndarray) else ax
            sns.heatmap(d_mat, cmap="coolwarm", ax=ax)
            ax.set_title(f"Distances for chain {idx}")
        plt.suptitle(f"Distance Map for {self.pdb_ID}")
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"{save_fig}/{self.pdb_ID}_dist.png")
        plt.show()

    def plot_contact_map(self, save_fig=None):
        fig, ax = plt.subplots(self.dim, self.dim, figsize=(11,9))
        for idx, c_mat in enumerate(self.contact_maps):
            ax = ax[idx] if isinstance(ax, np.ndarray) else ax
            sns.heatmap(c_mat, vmin=0, vmax=1, cmap=sns.cm.rocket_r, ax=ax)
            ax.set_title(f"Contacts for chain {idx}")
        plt.suptitle(f"Contact Map for {self.pdb_ID}")
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"{save_fig}/{self.pdb_ID}_cmap.png")
        plt.show()
        

if __name__ == "__main__":
    # example case 1PGA
    cm = ContactMapper(pdb_file="/home/rcml/pdb/1pga.pdb")
    print(cm.contact_maps)
    print(cm.distance_matrices)
    cm.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    cm.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")