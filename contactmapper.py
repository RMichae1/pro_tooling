import Bio.PDB
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from os.path import basename 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform, pdist, euclidean
from time import time


class ContactMapper:
    def __init__(self, pdb_file: str, pdb_ID: str=None, tri_dist: bool=False, angstrom_threshold: float=5., 
                check_AA: bool=True):
        self.pdb_file = pdb_file
        self.pdb_ID = basename(self.pdb_file).split(".")[0].upper() if not pdb_ID else pdb_ID
        self.tri_dist = tri_dist
        self.angstrom_threshold = angstrom_threshold
        self.check_AA = check_AA
        self.structure = Bio.PDB.PDBParser().get_structure(self.pdb_ID, self.pdb_file)
        self.model_obj: Model = self.structure[0]
        # self.chains = [chain for chain in self.model_obj]
        # only Chain-A is used
        self.chains = self.model_obj[0]
        # TODO: add Sequence information
        self.sequence = None
        self.dim = len(self.chains)
        self.centers: list = []
        self.all_coordinates: list = []
        self.distance_matrices: list = self.calc_distance_matrix(tri_dist=tri_dist)
        self.contact_maps: list = [distance_matrix < self.angstrom_threshold for distance_matrix in self.distance_matrices]

    @staticmethod
    def get_CA_coords(res: Residue) -> np.ndarray:
        try:
            coord = res["CA"].coord
        except KeyError as _:
            # no C-alpha present, take mean of 3 residues instead
            coord = np.mean(np.array([a.coord for a in res]))
        return coord

    @staticmethod
    def get_RES_coords(res: Residue) -> np.array:
        coord_vec = np.array([atom.coord for atom in res.get_atoms()])
        return coord_vec

    @staticmethod
    def calc_residue_tri_distance(coord_X: np.ndarray, coord_Y: np.ndarray) -> np.float:
        # take min atom distance between residues see mgpfusion/code/protein.m:494
        d_vec = np.array([[euclidean(c_X, c_Y) for c_Y in coord_Y] for c_X in coord_X])
        if len(d_vec) == 1:
            # last elem 0
            min_dist = np.min(d_vec)
        else:
            min_dist = np.min(d_vec[np.nonzero(d_vec)])
        return min_dist
    
    def calculate_chain_distance(self, chain_X: Chain, chain_Y: Chain, tri_res_calculation=False) -> np.array:
        mat = np.zeros((len(chain_X), len(chain_Y)), np.float)
        t_coord_X = []
        coord_X = []
        for res_X_pos, res_X in enumerate(chain_X):
            if not Bio.PDB.is_aa(res_X_pos) and self.check_AA:
                continue
            for res_Y_pos, res_Y in enumerate(chain_Y):
                if not Bio.PDB.is_aa(res_Y_pos) and self.check_AA:
                    continue
                if tri_res_calculation:
                    t_coord_X = self.get_RES_coords(res_X)
                    t_coord_Y = self.get_RES_coords(res_Y)
                    mat[res_X_pos, res_Y_pos] = self.calc_residue_tri_distance(t_coord_X, t_coord_Y)
                else: # calc using CA-centers
                    coord_X = self.get_CA_coords(res_X)
                    coord_Y = self.get_CA_coords(res_Y)
                    mat[res_X_pos, res_Y_pos] = euclidean(coord_X, coord_Y)
            self.centers.append(coord_X)
            self.all_coordinates.append(t_coord_X)
        return mat

    def calc_distance_matrix(self, tri_dist) -> np.array:
        """
        Go through all sequences in all chains and build distance matrix by residue 
        :returns : distance matrix as array
        """
        dist_matrix = []
        for idx, chain_X in enumerate(self.chains):
            # calculate distance for each residue in all the given chains
            for idy, chain_Y in enumerate(self.chains[idx:]):
                dist_matrix.append(self.calculate_chain_distance(chain_X, chain_Y,
                                    tri_res_calculation=tri_dist))
        return dist_matrix

    def get_min_distance(self):
        return min(min(mat for mat in self.distance_matrices))

    def get_max_distance(self):
        return max(max(mat for mat in self.distance_matrices))

    def plot_distance_matrix(self, save_fig=None):
        fig, ax = plt.subplots(self.dim, self.dim, figsize=(11, 9))
        for i in range(self.dim):
            for j in range(self.dim):
                d_mat = self.distance_matrices[i+j]
                ax_ = ax[i, j] if isinstance(ax, np.ndarray) else ax
                sns.heatmap(d_mat, cmap="coolwarm_r", ax=ax_)
                ax_.set_title(f"Distances for chain {i+j}")
        plt.suptitle(f"Distance Map for {self.pdb_ID}")
        plt.tight_layout()
        if save_fig:
            d_measure = "tri_dist" if self.tri_dist else "ca_dist"
            plt.savefig(f"{save_fig}/{self.pdb_ID}_dist_{d_measure}.png")
        plt.show()

    def plot_contact_map(self, save_fig=None):
        fig, ax = plt.subplots(self.dim, self.dim, figsize=(11,9))
        for i in range(self.dim):
            for j in range(self.dim):
                c_mat = self.contact_maps[i+j]
                ax_ = ax[i,j] if isinstance(ax, np.ndarray) else ax
                sns.heatmap(c_mat, vmin=0, vmax=1, cmap=sns.cm.rocket_r, ax=ax_)
                ax_.set_title(f"Contacts for chain {i} with {j}")       
        plt.suptitle(f"Contact Map for {self.pdb_ID}")
        plt.tight_layout()
        if save_fig:
            d_measure = "tri_dist" if self.tri_dist else "ca_dist"
            plt.savefig(f"{save_fig}/{self.pdb_ID}_cmap{d_measure}_{self.angstrom_threshold}.png")
        plt.show()
        

if __name__ == "__main__":
    # example case 1PGA - CA-distance
    cm = ContactMapper(pdb_file="/home/rcml/pdb/1pga.pdb")
    print(cm.contact_maps)
    print(cm.distance_matrices)
    cm.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    cm.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")
    # example case 1PGA - residue distance
    cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/1pga.pdb", tri_dist=True)
    cm_tri.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    cm_tri.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")

    # example case 1LZI - CA-distance
    cm = ContactMapper(pdb_file="/home/rcml/pdb/1lzi.pdb")
    print(cm.contact_maps)
    print(cm.distance_matrices)
    print(len(cm.distance_matrices))
    for d in cm.distance_matrices:
        print(len(d))
    cm.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    cm.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")
    # example case 1LZI - residue distance
    cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/1lzi.pdb", tri_dist=True)
    cm_tri.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    cm_tri.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")

    # example case 2LZM - CA-distance
    cm = ContactMapper(pdb_file="/home/rcml/pdb/2lzm.pdb")
    print(cm.contact_maps)
    print(cm.distance_matrices)
    print(len(cm.distance_matrices))
    for d in cm.distance_matrices:
        print(len(d))
    cm.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    cm.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")
    # example case 2LZM - residue distance
    cm_tri = ContactMapper(pdb_file="/home/rcml/pdb/2lzm.pdb", tri_dist=True)
    cm_tri.plot_distance_matrix(save_fig="/home/rcml/pro_tooling/fig/")
    cm_tri.plot_contact_map(save_fig="/home/rcml/pro_tooling/fig/")