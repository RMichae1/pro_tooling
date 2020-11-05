import Bio.PDB
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from os.path import basename 
import numpy as np
import matplotlib.pyplot as plt


class ContactMapper:
    def __init__(self, pdb_file: str, pdb_ID: str=None, angstrom_threshold: float=12.):
        self.pdb_file = pdb_file
        self.pdb_ID = basename(self.pdb_file).split(".")[0] if not pdb_ID else pdb_ID
        self.angstrom_threshold = angstrom_threshold
        self.structure = Bio.PDB.PDBParser().get_structure(self.pdb_ID, self.pdb_file)
        self.model_obj: Model = self.structure[0]
        self.chains = [chain for chain in self.model_obj]
        self.dim = len(chains)
        self.distance_matrix: np.array = self.calc_distace_matrix()
        self.contact_map: np.array = self.distance_matrix < self.angstrom_threshold

    @staticmethod
    def calc_residue_distance(res_X: Residue, res_Y: Residue) -> np.float:
        diff_vector = res_X["CA"].coord - res_Y["CA"].coord
        return np.sqrt(np.sum(diff_vector*diff_vector))

    @staticmethod
    def calculate_chain_distance(chain_X: Chain, chain_Y: Chain) -> np.array:
        mat = np.zeros((len(chain_X), len(chain_Y)), np.float)
        for res_X_pos, res_X: Residue in enumerate(chain_X):
            for res_Y_pos, res_Y: Residue in enumerate(chain_Y):
                mat[res_X_pos, res_Y_pos] = self.calc_residue_distance(res_X, res_Y)
        return mat

    def calc_distace_matrix(self) -> np.array:
        dist_matrix = np.zeros((self.dim, self.dim), np.float)
        for idx, chain_X: Chain in enumerate(chains):
            # calculate distance for each residue in all the given chains
            for idy, chain_Y: Chain in enumerate(chains[idx:]):
                dist_matrix[idx, idy] = self.calculate_chain_distance(chain_X, chain_Y)
        return dist_matrix

    def get_min_distance(self):
        return min(self.distance_matrix)

    def get_max_distance(self):
        return max(self.distance_matrix)

    def plot_distance_matrix(self):
        fig, ax = plt.subplots(self.dim, self.dim)
        for idx, _ in enumerate(self.distance_matrix):
            for idy, d in enumerate(self.distance_matrix[idx]):
                # skip empties  
                if not np.any(d):
                    continue
                ax[idx, idy].imshow(d)
        plt.show()

    def plot_contact_map(self):
        pass
        

if __name__ == "__main__":
    cm = ContactMapper(pdb_file="C:\Users\RCML\OneDrive - Novozymes A S\Documents\protein")
    print(cm.contact_map)
    print(cm.distance_matrix)
    cm.plot_distance_matrix()