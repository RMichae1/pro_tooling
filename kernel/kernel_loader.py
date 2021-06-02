from scipy.io import loadmat
from vae import VAE
from .vae_kernel import VaeKernel
from .s_kernel import MatrixKernel
from .linear_kernel import LinearKernel


class KernelLoader:
    def __init__(self, sub_matrices: list = [], sub_mat_ids: list = [], vae: VAE = None, lin: bool = False):
        """
        Interface to MatrixKernel that encapsulates the collection of substitution matrices
        used. 
        Has list of kernels as class property.
        sub_mat_ids takes IDs from SubMat Matlab
        """
        if isinstance(vae, VAE):
            self.kernels: list = [VaeKernel(vae)]
            s_mat_id = ["VAE-kernel"]
        elif lin:
            self.kernels: list = [LinearKernel()]
            s_mat_id = ["Linear-kernel"]
        else:
            s_mat, s_mat_id = self.load_sub_matrices(sub_matrices, sub_mat_ids)
            self.kernels: list = [MatrixKernel(matrix=s, matrix_id=m_id) for s, m_id in zip(s_mat, s_mat_id)]
        self.sub_matrices_ids = s_mat_id
        assert len(self.kernels) == len(self.sub_matrices_ids)

    def load_sub_matrices(self, sub_matrices, sub_mat_ids, ):
        matrices = loadmat(f"../data/mgp/subMats.mat").get('subMats')
        s_mat = []
        s_mat_id = []
        # check for provided sub_matrices in data subMat
        for m_vals, m_id, m_info in matrices:
            if sub_matrices or sub_mat_ids:
                if self.select_sub_matrices(m_id[0], m_info[0], sub_matrices, sub_mat_ids):
                    s_mat_id.append(m_id[0])
                    s_mat.append(m_vals)
                else:
                    continue
            else:
                s_mat_id.append(m_id[0])
                s_mat.append(m_vals)
        return s_mat, s_mat_id

    @staticmethod
    def select_sub_matrices(matrix_id, matrix_info, sub_matrices, s_mat_ids) -> bool:
        if matrix_id in s_mat_ids:  # check with IDs
            return True
        elif any([bool(s in matrix_info) for s in sub_matrices]):
            return True  # check with info IDs
        else:
            False
