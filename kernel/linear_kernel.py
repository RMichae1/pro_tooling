import numpy as np
import torch
from tqdm import tqdm
from torch.nn.functional import one_hot


class LinearKernel:
    def __init__(self) -> None:
        """
        Linear Kernel computes dot-product of input sequences.
        """
        self.num_classes = 21

    @staticmethod
    def linear_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.dot(x.T, y) + 1

    @torch.no_grad()
    def k(self, x_p, x_q=None, normalize_k=True) -> torch.Tensor:
        """
        Vectorized implementation of kernel for linear OH-sequence
        """
        # torch no-grad
        x_p = one_hot(x_p, num_classes=self.num_classes).flatten().float()
        N = x_p.shape[0]
        k = np.zeros([N, N])
        x_q = x_p if x_q is None else one_hot(x_q, num_classes=self.num_classes).flatten().float()
        for i in tqdm(range(N)):
            for j in tqdm(range(N)):
                k[i, j] = self.linear_k(x_p[i], x_q[j])
        if not normalize_k:
            return torch.Tensor(k).to(torch.float64)
        norm = np.sqrt(np.diag(k))[:, np.newaxis]
        k_hat = k / norm.dot(norm.T)
        return torch.Tensor(k_hat).to(torch.float64)